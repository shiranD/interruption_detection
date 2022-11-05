from speechbrain.pretrained import VAD
from speechbrain.pretrained import EncoderClassifier
from torch import nn

class VADer:

    def __init__(self):
        self.vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
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

    def process_vad(self, wav, fs, enroll_dict):
        "detects whether the are speech overlaps which indicate interruption"
        "a two sec. window is applied to every sample, 0.1 sec step size"
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        window = 2 # 2 second window
        fixed_step = 0.1 # 0.1 step size
        spk_seq = 
        i = 0
        real_speakers = []
        timing = []
        while len(wav) > int(i+window)*fs:
       
            # convert to embedding
            embedding = classifier.encode_batch(wav[int(i)*fs: int(i+window)*fs)]) 
            # measure distance from the dictionary
            distances = []
            speakers = []
            # which speaker is closer to a particular embedding
            for speaker in enroll_dict.keys():
                distances.append(cos(enroll_dict[speaker][0], embedding[0]))
                speakers.append(speaker)
            # choose the real speaker(s) who was closer to dict 'speaker' (or is closer to 1)
            real_speakers.append(speakers[distances.index(max(distances))])
            timing.append(int(i+window)*fs)
            # increment
            i+=fixed_step
        #return process_speaker_sequences(real_speakers, timing)

    def process_speaker_sequence(self, speakers, times):
        pass
