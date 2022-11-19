from speechbrain.pretrained import EncoderClassifier
import torchaudio
import os
from os.path import exists
from itertools import combinations
import pdb
from fg_model import get_f_net, get_g_net
from torch import nn
import pdb

def enroll_speakers(names, approach):
    local_download_path="../enr_files"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    speakers_dict = {}
    for spk1 in names:
        fname = os.path.join(local_download_path, spk1)
        if exists(fname):
            signal, fs = torchaudio.load(fname)
            if approach == 'pyaudio':
                embeddings = classifier.encode_batch(signal)
                embeddings = embeddings.squeeze(0)
            elif approach == 'composition':
                f_net = get_f_net()
                f_net.eval()
                signal = signal.transpose(1, 0).unsqueeze(0)
                embeddings =f_net(signal)
            embeddings = nn.functional.normalize(embeddings, p=2)
            speakers_dict[spk1]=embeddings
        else:
            print(f"can't find the speaker {spk1}")
    # enroll two speakers (I assume overlap happens between two for simplification)
    for (spk1, spk2) in list(combinations(names,2)):
        fname1 = os.path.join(local_download_path, spk1)
        fname2 = os.path.join(local_download_path, spk2)
        if exists(fname1) and exists(fname2):
            signal1, fs1 = torchaudio.load(fname1)
            signal2, fs2 = torchaudio.load(fname1)
            if fs1 == fs2:
                signal = 0.5*signal1+0.5*signal2
                if approach == 'pyaudio':
                    embeddings = classifier.encode_batch(signal)
                    embeddings = embeddings.squeeze(0)
                elif approach == 'composition':
                    g_net = get_g_net()
                    g_net.eval()
                    embeddings =g_net(speakers_dict[spk1], speakers_dict[spk2])
                embeddings = nn.functional.normalize(embeddings, p=2)
                speakers_dict[spk1+"_"+spk2]=embeddings
        else:
            print(f"can't find the speaker {fname1} or {fname2}")
    return speakers_dict
