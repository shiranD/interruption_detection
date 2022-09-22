import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier, VAD
import re
import os
from os.path import exists as file_exists
import pdb

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


wav_folder = "../wav/"
transcripts_folder = "../transcripts/"
for wav_file in os.listdir(wav_folder):
    print(wav_file)

    # replace .wav with .txt and find its transcript in transcripts folder
    txt_file = re.sub(".wav", ".txt", wav_file)
    txt = os.path.join(transcripts_folder, txt_file)
    wav = os.path.join(wav_folder, wav_file)

    # check if current path is a file
    if file_exists(txt):
        # apply VAD
        VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
        # 1- Let's compute frame-level posteriors first
        prob_chunks = VAD.get_speech_prob_file(wav)

        # 2- Let's apply a threshold on top of the posteriors
        prob_th = VAD.apply_threshold(prob_chunks).float()

        # 3- Let's now derive the candidate speech segments
        boundaries = VAD.get_boundaries(prob_th)
        
        # 4- Apply energy VAD within each candidate speech segment (optional)
        boundaries = VAD.energy_VAD(wav,boundaries)
        
        # 5- Merge segments that are too close
        boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)
        
        # 6- Remove segments that are too short
        boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)
        
        # 7- Double-check speech segments (optional).
        boundaries = VAD.double_check_speech_segments(boundaries, wav,  speech_th=0.5)

        # get the sample rate
        signal, fs = torchaudio.load(wav)
        for (begin1, end1), (begin2, end2)  in zip(boundaries, boundaries[1:]):
            # apply x-vector for diarization
            pdb.set_trace()
            # below a fixed threshold
            if begin2 - end1 < threshold:
                embeddings1 = classifier.encode_batch(signal[0][round(float(begin1))*fs: round(float(end1))*fs])
                embeddings2 = classifier.encode_batch(signal[0][round(float(begin2))*fs: round(float(end2))*fs])
                 # measure distance from the dictionary
                 sp1 = []
                 sp2 = []
                 for speaker in speakers:
                     sp1.append(cos(speaker, embeddings1))               
                     sp2.append(cos(speaker, embeddings2))
                 # if are not the same speaker
                 if sp1.index(min(sp1)) != sp2.index(min(sp2)):
                     print("{D[argmin(sp2)]} interrupted to {D[argmin(sp1)]} at {begin} second")

