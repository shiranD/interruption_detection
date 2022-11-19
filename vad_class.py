from speechbrain.pretrained import VAD
from speechbrain.pretrained import EncoderClassifier
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

    def process_vad(self, wav, fs, enroll_dict, begin_time, approach):
        """
        detects whether the are speech overlaps which indicate interruption
        a two second window is applied to every sample and 0.1 sec step size if possible
        """
        if approach == 'pyaudio':
            classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        elif approach == 'composition':
            f_net = get_f_net()
            f_net.eval()

        window = 2 # 2 second window
        fixed_step = 0.1 # 0.1 step size
        i = 0
        real_speakers = []
        timing = []

        if len(wav) < (i+window)*fs:
            if approach == 'pyaudio':
                embedding = classifier.encode_batch(wav)
                embedding = embedding.squeeze(0)
            elif approach == 'composition':
                 wav = wav.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(wav)
            embedding = nn.functional.normalize(embedding, p=2)
            return [[self.speaker_verification(embedding, enroll_dict), begin_time, len(wav)/fs]] 

        while len(wav) > int(i+window)*fs:
            segment = wav[int(i*fs): int((i+window)*fs)]
            if approach == 'pyaudio':
                # convert to embedding
                embedding = classifier.encode_batch(segment) 
                embedding = embedding.squeeze(0)
            elif approach == 'composition':
                 segment = segment.unsqueeze(0).unsqueeze(2)
                 embedding = f_net(segment)
            embedding = nn.functional.normalize(embedding, p=2)
            speaker = self.speaker_verification(embedding, enroll_dict)
            # choose the real speaker(s) who was closer to dict 'speaker' (or is closer to 1)
            real_speakers.append(speaker)
            timing.append(begin_time+i+window)
            # increment
            i+=fixed_step
        return self.process_speaker_sequence(real_speakers, timing)
        
    def speaker_verification(self, emb, enroll_dict):
        """
        measures distance from the dictionary embeddings
        and returns the speaker(s) who is was closest
        imposes a threshold to assign a speaker, otherwise assigns background noise
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        distances = []
        speakers = []
        # which speaker is closer to a particular embedding
        for speaker in enroll_dict.keys():
            distances.append(cos(enroll_dict[speaker], emb))
            speakers.append(speaker)
        # choose the real speaker(s) who was closer to dict 'speaker' (or is closer to 1)
        # speakers are assigned only after crossing a threshold
        print(distances)
        return speakers[distances.index(max(distances))] if distances.index(max(distances)) > self.threshold else "none" 
        #return speakers[distances.index(max(distances))]

    def process_speaker_sequence(self, speakers, times):
        """
        merges speaking time to determine speaking segments
        """        
        old_spk = speakers[0]
        old_time = times[0]
        spk_seq = []
        for j, (speaker, time) in enumerate(zip(speakers[1:-1], times[1:-1])):
            if speaker != old_spk:
                if speakers[j+1] == speaker: # the speaker has changed
                    # mark the old speaker times
                    spk_seq.append([old_spk, old_time, time-old_time])
                    old_spk = speaker
                    old_time = time
        spk_seq.append([old_spk, old_time, times[-1]-old_time])
        return spk_seq
