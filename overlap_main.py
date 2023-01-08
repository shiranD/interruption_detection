import torchaudio
import re
import os
import sys
from os.path import exists as file_exists
import pdb
from vad_class import VADer
from enrollment import enroll_speakers
from process_xl import processXL
from matplotlib import pyplot as plt
from retrieve_matching_files import file_matcher
import torch
from csv import reader

wav_folder = "../wav/"
txt_folder = "../n_transcripts/"
speaker_threshold = 0.99
vad = VADer(speaker_threshold)
window = 0.5
positive_th = 0.3
kernel=5

xl_folder = "../Finalized CPS Annotations/"
b_len = []
b_ons = []
b_inter = []

def process(time):
    hr, mn, sc = time.split(":")
    time = int(hr)*60*60+int(mn)*60+float(sc)
    return time

results_file = str(window)+"_"+str(kernel)+"_toy_"+str(speaker_threshold)+".txt"
f = open(results_file, "w")
for xl_file in os.listdir(xl_folder):
    #xl_file="Beck-White_CPS_Annotated_Crystal_21-11-04_si_l2_p2_video-yeti5_5min.xlsx"
    #xl_file="Beck-White_CPS_Annotated_Crystal_21-11-04_si_l2_p4_yeti5_5min.xlsx"
    print(xl_file)
#    xl_file = "Beck-White_CPS_Annotated_Crystal_21-11-04_si_l2_p3_video_yeti5_5min.xlsx"
    # extract the name
    base_name = xl_file.split("Beck-White_CPS_Annotated_")[1] 
    base_name = base_name.split(".xlsx")[0] 
    # for every annotated file grab annotations by time and compute length
    lens, interruptors, onsets = processXL(os.path.join(xl_folder, xl_file))
    # find wav and transcript
    wav_file = base_name+".wav"
    # search for it through gdrive
    wav_file = file_matcher(wav_folder, wav_file)
    if not type(wav_file) == str:
       continue
    # replace .wav with .txt and find its transcript in transcripts folder
    txt_file = base_name+".txt"
    # search for it through gdrive
    txt_file = file_matcher(txt_folder, txt_file)
#    # check if current path is a file
    if not type(txt_file) == str:
       continue
    elif file_exists(os.path.join(txt_folder, txt_file)):
        # process for overlapping times based on transcription
        f.write("\n")
        f.write("gold human transcripts "+base_name)
        f.write("\n")
        
        with open(os.path.join(txt_folder, txt_file), "r") as g:
            prev = 0
            for line in g.readlines():
               student, t1, t2, _ = line.split("\t")
               t1 = process(t1)
               t2 = process(t2)
               if student == 'Teacher' or 'other' in student: continue
               if not prev:
                   prev = [t1, t2]
                   continue
               else:
                   if t1-prev[1] <= 0:
                       f.write(student+"\t"+"overlap\t"+str(t1)+"\t"+str(prev[1])+"\t"+str(t1-prev[1]))
                       b_len.append(t1-prev[1])
                       if t1-prev[1] < -4:
                           print(base_name, t1-prev[1])
                       f.write("\n")
                   elif t1-prev[1] < positive_th:
                       f.write(student+"\t"+"shorturn\t"+str(prev[1])+"\t"+str(t1)+"\t"+str(t1-prev[1]))
                       b_len.append(t1-prev[1])
                       f.write("\n")
               prev = [t1, t2]
        names = []
        with open(os.path.join(txt_folder, txt_file), "r") as g:
            for line in g.readlines():
               line = line.replace("Student ", "student-")
               line = line.replace("student ", "student-")
               name = line.split()[0] 
               if 'other' not in name and 'teacher' not in name:
                   name = 'enrollment_'+name+'.wav'
                   names.append(name)
            names = set(names)
        for approach in ['composition', 'pyaudio']:
#        for approach in ['pyaudio']:
        #for approach in ['composition']:
            enroll_dict = enroll_speakers(names, approach)
            f.write(approach+" annotations")
            f.write("\n")
            wav = os.path.join(wav_folder, wav_file)
            #scd = torch.hub.load("pyannote/pyannote-audio", "scd")
            #scd = torch.hub.load("pyannote/pyannote-audio", "scd", trust_repo='check')
            #change_scores=scd({"audio": wav})
            # apply vad
            boundaries = vad.chunk(wav)
            # get the sample rate
            signal, fs = torchaudio.load(wav)
            # detect speech overlap
            #import sounddevice as sd
            with open(base_name+"_"+approach, "w") as d:
                for (begin, end) in boundaries:
                    chunk = signal[0][int(float(begin)*fs): int(float(end)*fs)]
                    #print(begin, end)   
                    #sd.play(chunk, fs)
                    interruption_details = vad.process_vad(chunk, fs, enroll_dict, begin, approach, kernel)
                    for (i, o, l) in interruption_details:
                        if i != 'none':
                            print(f"{i} {o} {l}\n")
                            d.write(f"{i}\t{o}\t{l}\n")
            #pdb.set_trace()
#            sys.exit()   
            with open(base_name+"_"+approach, "r") as d:
                prev = 0
                for line in d.readlines():
                   student, t1, t2 = line.split("\t")
                   t1=float(t1)
                   t2=float(t2)
                   if not prev:
                       prev = [t1, t2]
                       continue
                   else:
                       if t1-prev[1] <= 0:
                           f.write(student+"\t"+"overlap\t"+str(t1)+"\t"+str(prev[1])+"\t"+str(t1-prev[1]))
                           b_len.append(t1-prev[1])
                           if t1-prev[1] < -4:
                               print(base_name, t1-prev[1])
                           f.write("\n")
                       elif t1-prev[1] < positive_th:
                           f.write(student+"\t"+"shorturn\t"+str(prev[1])+"\t"+str(t1)+"\t"+str(t1-prev[1]))
                           b_len.append(t1-prev[1])
                           f.write("\n")
                   prev = [t1, t2]

            # process temporary file and write
#        sys.exit()   
#    sys.exit()
#import seaborn as sns
#import numpy as np
#import matplotlib as mpl
#label_size = 20
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size
#print(b_len)
#b_len = np.array(b_len)
##sns.histplot(data=b_len, x="interruption onset relative to first speaker [second]")
#sns.histplot(data=b_len)#, binwidth=2)
#w=1
#fig, ax = plt.subplots(figsize =(10, 7))
#ax.hist(b_len,color='orange', bins=np.arange(min(b_len),max(b_len)+w, w))
#ax.set_title("human-annotated interruption", fontsize=20)
#ax.set_xlabel("interruption onset relative to first speaker [second]", fontsize=20)
#ax.set_ylabel("counts", fontsize=20)
#plt.savefig("all_lens.eps")
#fig, ax = plt.subplots(figsize =(10, 7))
#ax.hist(b_ons)
#ax.set_title("interruption onset relative to first speaker [second]")
#plt.savefig("all_onsets")
#sys.exit()
