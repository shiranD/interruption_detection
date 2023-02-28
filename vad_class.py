from speechbrain.pretrained import VAD
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition
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


    def process_vad_segment(self, wav, fs, enroll_dict, begin_time, approach, kernel, window):
        """
        detects whether the are speech overlaps which indicate interruption
        a two second window is applied to every sample and 0.1 sec step size if possible
        """
        self.approach = approach
        if 'pyaudio_ecapa' in self.approach:
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
            self.verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        if 'pyaudio_xvec' in self.approach:
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
            self.verifier = SpeakerRecognition.from_hparams(source="speechbrain/pretrained_models/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        elif 'composition' in self.approach:
            f_net = get_f_net()
            f_net.eval()
        self.sp_dict = enroll_dict
        self.kernel = kernel
        self.window = window # 2 second window
        self.fixed_step = 0.1 # 0.1 step size
        i = 0
        real_speakers = []
        timing = []
        embeddings = []
        if len(wav) < (i+self.window)*fs:
            if 'pyaudio' in self.approach:
                embedding = classifier.encode_batch(wav)
                embedding = embedding.squeeze(0)
            elif 'composition' in self.approach:
                 wav = wav.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(wav)
            embeddings.append(nn.functional.normalize(embedding, p=2))
            return self.speaker_verification_per_segment(embeddings, [[float(begin_time), float(begin_time+len(wav)/fs)]])

        segment = wav[int(i*fs): int((i+self.window)*fs)]
        while len(segment) > 0.5*self.window*fs:
            segment = wav[int(i*fs): int((i+self.window)*fs)]
            if 'pyaudio' in self.approach:
                # convert to embedding
                embedding = classifier.encode_batch(segment) 
                embedding = embedding.squeeze(0)
            elif 'composition' in self.approach:
                 segment = segment.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(segment)
            embeddings.append(nn.functional.normalize(embedding, p=2))
            timing.append([float(begin_time)+i, float(begin_time)+i+self.window])
            # increment
            i+=self.fixed_step
            segment = wav[int(i*fs): int((i+self.window)*fs)]
        return self.speaker_verification_per_segment(embeddings, timing)

    def speaker_verification_per_segment(self, embs, times):
        """
        measures distance from the dictionary embeddings
        and returns the speaker(s) who is was closest
        imposes a threshold to assign a speaker, otherwise assigns background noise
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        original = defaultdict(list)
        smooth = defaultdict(list)
        stretch = int(self.kernel/2)
        final_seq = []

        # table the speakers' distances
        for speaker in self.sp_dict.keys():
            for emb in embs:
                if 'composition' in self.approach:
                    original[speaker]+=cos(self.sp_dict[speaker], emb)
                if 'pyaudio' in self.approach:
#                    original[speaker]+=self.verifier.similarity(self.sp_dict[speaker], emb)
                    original[speaker]+=[float(self.verifier.similarity(self.sp_dict[speaker], emb))]
        # smoothing
        for speaker in self.sp_dict.keys():
            speaker_dists = original[speaker]
            for j, dist in enumerate(speaker_dists):
                #print("stretch", len(speaker_dists[j:]), stretch)
                if j < stretch or len(speaker_dists[j:])<=stretch:
                    smooth[speaker]+=[dist]
                else:
                #   print(-stretch+j)
                #   print(j, len(speaker_dists[j:]))
                   ker_sum = [speaker_dists[-stretch+m+j] for m in range(self.kernel)]
                   smooth[speaker]+=[float(sum(ker_sum)/self.kernel)]

        # choosing the closest speaker of the smooth outcome
        for n, (start_time, end_time) in enumerate(times):
            #choose max of speakers and seq it with corresponding time
            score = -1
            spk = ""
            for speaker in self.sp_dict.keys():
                if smooth[speaker][n] > score:
                    spk  = speaker
                    score = smooth[speaker][n]
            if final_seq and final_seq[-1][0]==spk:
                final_seq[-1][2]=end_time
            elif final_seq:
                final_seq[-1][2]=start_time
                final_seq.append([spk, start_time, end_time])
            else:
                final_seq.append([spk, start_time, end_time])
        #pdb.set_trace()
        # merging windows to segments
        return final_seq

    def process_vad(self, wav, fs, enroll_dict, begin_time, approach, kernel, window):
        """
        detects whether the are speech overlaps which indicate interruption
        a two second window is applied to every sample and 0.1 sec step size if possible
        """
        if 'pyaudio' in approach:
            #classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
            #self.verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
            self.verifier = SpeakerRecognition.from_hparams(source="speechbrain/pretrained_models/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        elif 'composition' in approach:
            f_net = get_f_net()
            f_net.eval()
        self.sp_dict = enroll_dict
        self.kernel = kernel
        self.window = window # 2 second window
        fixed_step = 0.1 # 0.1 step size
        i = 0
        real_speakers = []
        timing = []
        if len(wav) < (i+self.window)*fs:
            if 'pyaudio' in approach:
                embedding = classifier.encode_batch(wav)
                embedding = embedding.squeeze(0)
            elif 'composition' in approach:
                 wav = wav.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(wav)
            embedding = nn.functional.normalize(embedding, p=2)
            return self.process_speaker_sequence([self.speaker_verification(embedding)], [float(begin_time), float(begin_time+len(wav)/fs)]) 

        segment = wav[int(i*fs): int((i+self.window)*fs)]
        while len(segment) > 0.5*self.window*fs:
            segment = wav[int(i*fs): int((i+self.window)*fs)]
            #print(len(wav), len(segment), int(i*fs), int((i+self.window)*fs))
            if 'pyaudio' in approach:
                # convert to embedding
                embedding = classifier.encode_batch(segment) 
                embedding = embedding.squeeze(0)
            elif 'composition' in approach:
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
        #print(real_speakers)
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
            distances.append(self.verifier.similarity(self.sp_dict[speaker], emb))
            #print("dist",float(self.verifier.similarity(self.sp_dict[speaker], emb)))
            #distances.append(cos(self.sp_dict[speaker], emb))
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
        #print(spk_seq)

        # split speakers if they overlap
        n_spk_seq = []
        for speaker, time in spk_seq:
            if 'v_e' in speaker:
                spk = speaker.replace('v_e', 'v e')
                for s in spk.split():
                    n_spk_seq.append([s, time])
            else:  
                n_spk_seq.append([speaker, time])
                
        #print(n_spk_seq)
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
        #print(new_seq)
        # sort by time
        new_seq = sorted(new_seq, key=lambda x: x[1])
        #print(new_seq)
        return new_seq
