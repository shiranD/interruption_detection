import torchaudio
from speechbrain.pretrained import EncoderClassifier, VAD
import re
import os
from os.path import exists as file_exists
import pdb

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")



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

        # apply x-vector for diarization
        signal, fs =torchaudio.load(wav)
        embeddings = classifier.encode_batch(signal)

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

     
