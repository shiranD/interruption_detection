from speechbrain.pretrained import VAD
from speechbrain.pretrained import EncoderClassifier
from collections import defaultdict
import operator
from torch import nn
import pdb
from fg_model import get_f_net

class VADer:

    def __init__(self, threshold):
        self.vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
        self.threshold = threshold
    def chunk(self, wav):

        # 1- Let's compute frame-level posteriors first
        prob_chunks = self.vad.get_speech_prob_file(wav)

        # 2- Let's apply a threshold on top of the posteriors
        prob_th = self.vad.apply_threshold(prob_chunks).float()

        # 3- Let's now derive the candidate speech segments
        boundaries = self.vad.get_boundaries(prob_th)

        # 4- Apply energy VAD within each candidate speech segment (optional)
        boundaries = self.vad.energy_VAD(wav,boundaries)

        # 5- Merge segments that are too close
        boundaries = self.vad.merge_close_segments(boundaries, close_th=0.250)

        # 6- Remove segments that are too short
        boundaries = self.vad.remove_short_segments(boundaries, len_th=0.250)

        # 7- Double-check speech segments (optional).
        boundaries = self.vad.double_check_speech_segments(boundaries, wav,  speech_th=0.5)
        return boundaries

    def process_vad(self, wav, fs, enroll_dict, begin_time, approach, kernel):
        """
        detects whether the are speech overlaps which indicate interruption
        a two second window is applied to every sample and 0.1 sec step size if possible
        """
        if approach == 'pyaudio':
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        elif approach == 'composition':
            f_net = get_f_net()
            f_net.eval()
        self.sp_dict = enroll_dict
        self.kernel = kernel
        self.window = 0.5 # 2 second window
        fixed_step = 0.1 # 0.1 step size
        i = 0
        real_speakers = []
        timing = []
        #pdb.set_trace()
        if len(wav) < (i+self.window)*fs:
            if approach == 'pyaudio':
                embedding = classifier.encode_batch(wav)
                embedding = embedding.squeeze(0)
            elif approach == 'composition':
                 wav = wav.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(wav)
            embedding = nn.functional.normalize(embedding, p=2)
            return self.process_speaker_sequence([self.speaker_verification(embedding)], [float(begin_time), float(begin_time+len(wav)/fs)]) 

        segment = wav[int(i*fs): int((i+self.window)*fs)]
        while len(segment) > 0.5*self.window*fs:
            segment = wav[int(i*fs): int((i+self.window)*fs)]
            print(len(wav), len(segment), int(i*fs), int((i+self.window)*fs))
            if approach == 'pyaudio':
                # convert to embedding
                embedding = classifier.encode_batch(segment) 
                embedding = embedding.squeeze(0)
            elif approach == 'composition':
                 segment = segment.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(segment)
            embedding = nn.functional.normalize(embedding, p=2)
            speaker = self.speaker_verification(embedding)
            # choose the real speaker(s) who was closer to dict 'speaker' (or is closer to 1)
            real_speakers.append(speaker)
            timing.append(float(begin_time)+i+self.window)
            # increment
            i+=fixed_step
            segment = wav[int(i*fs): int((i+self.window)*fs)]
        print(real_speakers)
        return self.process_speaker_sequence(real_speakers, timing)
        
    def speaker_verification(self, emb):
        """
        measures distance from the dictionary embeddings
        and returns the speaker(s) who is was closest
        imposes a threshold to assign a speaker, otherwise assigns background noise
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        distances = []
        speakers = []
        # which speaker is closer to a particular embedding
        for speaker in self.sp_dict.keys():
            distances.append(cos(self.sp_dict[speaker], emb))
            speakers.append(speaker)
        # choose the real speaker(s) who was closer to dict 'speaker' (or is closer to 1)
        # speakers are assigned only after crossing a threshold
        #print(distances)
        return speakers[distances.index(max(distances))] if distances.index(max(distances)) > self.threshold else "none" 
        #return speakers[distances.index(max(distances))]

    def process_speaker_sequence(self, speakers, times):
        """
        merges speaking time to determine speaking segments
        """        
        spk_seq = []
        # all seq is the avg of seq
        if len(speakers) < self.kernel:
            spk_d = defaultdict(int)
            for x in speakers:
                spk_d[x]+=1
            spk = max(spk_d.items(), key=operator.itemgetter(1))[0]
            for i in range(len(speakers)):
                spk_seq.append([spk, times[i]])
        else:
            start = self.kernel//2
            end = len(speakers)-self.kernel//2 
            for j, time in enumerate(times[start:end]):
                # avg the spekaers
                spk_d = defaultdict(int)
                for x in speakers[j:j+self.kernel]:
                    spk_d[x]+=1
                # avg over kernel number of sampels
                spk = max(spk_d.items(), key=operator.itemgetter(1))[0] 
                # time assignment
                spk_seq.append([spk, time])
            last = spk_seq[-1][0]
            first = spk_seq[0][0]
            for i in range(start):
                spk_seq.insert(0,[first, times[start-i-1]])
            for i in range(start):
                spk_seq.append([last, times[-(self.kernel//2)+i]])
        new_seq = self.process_windows(spk_seq)
        return new_seq

    def process_windows(self, spk_seq):
        #spk_seq = [['enrollment_student-036.wav_enrollment_student-034.wav', 0.6399999856948853]]
        print(spk_seq)

        # split speakers if they overlap
        n_spk_seq = []
        for speaker, time in spk_seq:
            if 'v_e' in speaker:
                spk = speaker.replace('v_e', 'v e')
                for s in spk.split():
                    n_spk_seq.append([s, time])
            else:  
                n_spk_seq.append([speaker, time])
                
        print(n_spk_seq)
        #pdb.set_trace()
        # sort by speaker 
        speaker_seg = defaultdict(list)
        for spk, time in n_spk_seq:
            speaker_seg[spk].append([spk, time, time+self.window])

        if len(n_spk_seq)==1: return [[n_spk_seq[-1][0], n_spk_seq[-1][1], n_spk_seq[-1][1]+self.window]] 
        # merge by speaker
        new_seq = []
        for speaker in self.sp_dict.keys():
           end_t = 0
           for spk, start, end in speaker_seg[speaker]:
               if not end_t:
                   end_t = end
                   new_seq.append([speaker, start, end])
               else:
                   if start == end_t:
                       new_seq[-1][2]=end
                       end_t=end
                     
        #pdb.set_trace()
        print(new_seq)
        # sort by time
        new_seq = sorted(new_seq, key=lambda x: x[1])
        print(new_seq)
        return new_seq
