import torch
from pyannote.audio.models import SincTDNN
from pyannote.audio.train.task import Task, TaskOutput, TaskType
import torchaudio
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 512

class GNet (nn.Module):
    def __init__ (self):
        super(GNet, self).__init__()
        self.linear1a = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.linear1b = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    def forward (self, X1, X2):
        linear = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear

def get_f_net():
    task = Task(TaskType.REPRESENTATION_LEARNING,TaskOutput.VECTOR)
    specifications = {'X':{'dimension': 1} ,'task': task}
    sincnet = {'instance_normalize': True, 'stride': [5, 1, 1], 'waveform_normalize': True}
    tdnn = {'embedding_dim': 512}
    embedding = {'batch_normalize': False, 'unit_normalize': False}
    f_net = SincTDNN(specifications=specifications, sincnet=sincnet, tdnn=tdnn, embedding=embedding)
    # .to(device)         
    f_net.load_state_dict(torch.load("checkpoints/best_f.pt", map_location=torch.device('cpu')))
    return f_net

def get_g_net():
    g_net = GNet()
    # .to(device)
    g_net.load_state_dict(torch.load("checkpoints/best_g.pt",map_location=torch.device('cpu')))
    return g_net

if __name__=="__main__":
    # dic = enroll_speakers()
    # print(dic)
    f_net = get_f_net()
    g_net = get_g_net()
    f_net.eval()
    g_net.eval()
    # file1
    file1 = "../enr_files/enrollment_student-047.wav"
    signal47, fs = torchaudio.load(file1)
    signal47=signal47.transpose(1, 0).unsqueeze(0)
    embeddings47 =f_net(signal47)
    # file2
    file2 = "../enr_files/enrollment_student-048.wav"
    signal48, fs = torchaudio.load(file2)
    signal48=signal48.transpose(1, 0).unsqueeze(0)
    embeddings48 =f_net(signal48)
    # compositional
    embedding_com = g_net(embeddings47,embeddings48)
    pdb.set_trace()
    print(embeddings47.shape)
    print(embeddings47)
    print(embeddings48.shape)
    print(embeddings48)
    print(embedding_com.shape)
    print(embedding_com)


