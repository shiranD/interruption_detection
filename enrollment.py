from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import os

def enroll_speakers(names):
    # input speakers
    # output dict of speakers
    
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    #gauth.LocalWebserverAuth()
    # Creates local webserver and auto handles authentication
    drive = GoogleDrive(gauth)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    speakers_dict = {}
    #file0 = "enrollment_student-006.wav"
    for file0 in names:
        local_download_path="/Users/shdu9019/Documents/interruption_detection/interruption_detection/g_files"
        path = "https://drive.google.com/drive/u/1/folders/1vqIVEKdWVvZf3KBVgdOJ8_KbuzM_dndK/"
        # Auto-iterate through all files that matches this query
        file_list = drive.ListFile({'q': "'1vqIVEKdWVvZf3KBVgdOJ8_KbuzM_dndK' in parents"}).GetList()
        for file1 in file_list:
          #print('title: %s, id: %s' % (file1['title'], file1['id']))
          if file1['title'] == file0:
              fname = os.path.join(local_download_path, file1['title'])
              #print('downloading to {}'.format(fname))
              f_ = drive.CreateFile({'id': file1['id']})
              f_.GetContentFile(fname)
              signal, fs = torchaudio.load(fname)
              embeddings = classifier.encode_batch(signal)
              speakers_dict[file0]=embeddings
              #print("is", fs)
              break
    return speakers_dict
