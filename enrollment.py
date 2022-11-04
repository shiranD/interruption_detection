from speechbrain.pretrained import EncoderClassifier
import torchaudio
import os
from os.path import exists

def enroll_speakers(names):
    local_download_path="../enr_files"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    speakers_dict = {}
    for file0 in names:
        fname = os.path.join(local_download_path, file0)
        if exists(fname):
            signal, fs = torchaudio.load(fname)
            embeddings = classifier.encode_batch(signal)
            speakers_dict[file0]=embeddings
        else:
            print("can't find the speaker", file0)
    return speakers_dict
