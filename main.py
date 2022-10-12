from torch import nn
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import re
import os
import sys
from os.path import exists as file_exists
import pdb
from vad_class import VADer
from enrollment import enroll_speakers
import math

def convert2ms(tensor):
    floats = float(tensor)
    splitter = math.modf(floats)
    return round(splitter[0]*100+splitter[1]*1000)

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

wav_folder = "../wav/"
transcripts_folder = "../transcripts/"
vad = VADer()
fixed_th = 1

for wav_file in os.listdir(wav_folder):
    print(wav_file)

    # replace .wav with .txt and find its transcript in transcripts folder
    txt_file = re.sub(".wav", ".txt", wav_file)
    txt = os.path.join(transcripts_folder, txt_file)

    # check if current path is a file
    if file_exists(txt):
        # enroll participants
        names = []
        with open(txt, "r") as f:
            for line in f.readlines():
               name = line.split('\t')[0].lower()
               if 'other' not in name and 'teacher' not in name:
                   name = name.replace(' ', '-')
                   name = 'enrollment_'+name+'.wav'
                   names.append(name)
            names = set(names)
            enroll_dict = enroll_speakers(names)
        # prep wav path 
        wav = os.path.join(wav_folder, wav_file)
        # apply vad
        boundaries = vad.chunk(wav)
        # get the sample rate
        signal, fs = torchaudio.load(wav)
        fs = fs//1000
        for (begin1, end1), (begin2, end2)  in zip(boundaries, boundaries[1:]):
            # below a fixed threshold
            if begin2 - end1 < fixed_th:
                begin1 = convert2ms(begin1)
                begin2 = convert2ms(begin2)
                end1 = convert2ms(end1)
                end2 = convert2ms(end2)
                # apply x-vector for diarization
                embeddings1 = classifier.encode_batch(signal[0][begin1*fs: end1*fs])
                embeddings2 = classifier.encode_batch(signal[0][begin2*fs: end2*fs])
                # measure distance from the dictionary
                real_speakers = []
                for emb in [embeddings1, embeddings2]:
                    time_step_emb = []
                    speakers = []
                    # which speaker is closer to a particular embedding
                    for speaker in enroll_dict.keys():
                        time_step_emb.append(cos(enroll_dict[speaker][0], emb[0]))
                        speakers.append(speaker)
                    # choose the real speaker who was closer to dict 'speaker' (or is closer to 1)
                    real_speakers.append(speakers[time_step_emb.index(max(time_step_emb))])
                    
                # if are not the same speaker
                if real_speakers[0] != real_speakers[1]:
                    print(f"{real_speakers[0]} interrupted to {real_speakers[1]} at {begin2} second")

