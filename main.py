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

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

wav_folder = "../wav/"
transcripts_folder = "../transcripts/"
vad = VADer()

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
        print(enroll_dict)
        sys.exit()
        # prep wav path 
        wav = os.path.join(wav_folder, wav_file)
        # apply vad
        boundaries = vad.chunk(wav)
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

