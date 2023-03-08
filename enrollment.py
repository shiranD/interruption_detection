from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
import torchaudio
import os
from os.path import exists
from itertools import combinations
import pdb
from fg_model import get_f_net, get_g_net
from torch import nn
import pdb

def enroll_speakers(names, approach, enroll_path):
    if 'pyaudio_xvec' in approach:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    if 'pyaudio_ecapa' in approach:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    speakers_dict = {}
    for spk1 in names:
        fname = os.path.join(enroll_path, spk1)
        if exists(fname):
            signal, fs = torchaudio.load(fname)
            if 'pyaudio' in approach:
                #pdb.set_trace()
                embeddings = classifier.encode_batch(signal)
                embeddings = embeddings.squeeze(0)
            elif 'composition' in approach:
                f_net = get_f_net()
                f_net.eval()
                signal = signal.transpose(1, 0).unsqueeze(0)
                embeddings =f_net(signal)
            embeddings = nn.functional.normalize(embeddings, p=2)
            speakers_dict[spk1]=embeddings
        else:
            print(f"can't find the speaker {spk1}")
    if 'single' not in approach:
        # enroll two speakers (I assume overlap happens between two for simplification)
        for (spk1, spk2) in list(combinations(names,2)):
            fname1 = os.path.join(enroll_path, spk1)
            fname2 = os.path.join(enroll_path, spk2)
            if exists(fname1) and exists(fname2):
                signal1, fs1 = torchaudio.load(fname1)
                signal2, fs2 = torchaudio.load(fname2)
                max_len = max(signal1.size(dim=1), signal2.size(dim=1))
                signal1 = F.pad(signal1, (0, max_len-signal1.size(dim=1)), "constant", 0)
                signal2 = F.pad(signal2, (0, max_len-signal2.size(dim=1)), "constant", 0)
                if fs1 == fs2:
                    signal = 0.5*signal1+0.5*signal2
                    if 'pyaudio' in approach:
                        embeddings = classifier.encode_batch(signal)
                        embeddings = embeddings.squeeze(0)
                    elif 'composition_gf' in approach:
                        g_net = get_g_net()
                        g_net.eval()
                        embeddings =g_net(speakers_dict[spk1], speakers_dict[spk2])
                    elif 'composition_f' in approach:
                        f_net = get_f_net()
                        f_net.eval()
                        signal = signal.transpose(1, 0).unsqueeze(0)
                        embeddings =f_net(signal)
                    embeddings = nn.functional.normalize(embeddings, p=2)
                    speakers_dict[spk1+"_"+spk2]=embeddings
            else:
                print(f"can't find the speaker {fname1} or {fname2}")
    return speakers_dict
