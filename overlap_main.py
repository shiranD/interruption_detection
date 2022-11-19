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
from csv import reader

wav_folder = "../wav/"
txt_folder = "../transcripts/"
speaker_threshold = 0.9
vad = VADer(speaker_threshold)
fixed_th = 0.5
approach='pyaudio'
#approach='composition'

xl_folder = "../Finalized CPS Annotations/"
b_len = []
b_ons = []
b_inter = []

results_file = "results.txt"
f = open(results_file, "w")
for xl_file in os.listdir(xl_folder):
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
    txt_file = "transcript_diarized_timestamped_"+re.sub(".wav", ".txt", wav_file)
    # search for it through gdrive
    txt_file = file_matcher(txt_folder, txt_file)
#    # check if current path is a file
    print(file_exists(os.path.join(txt_folder, txt_file)))
    if not type(txt_file) == str:
       continue
    elif file_exists(os.path.join(txt_folder, txt_file)):
        print("found both")
        # append from gold standard annotations
        b_len.extend(lens)
        b_ons.extend(onsets)
        b_inter.extend(interruptors)
        f.write(base_name)
        f.write("\n")
        f.write("gold annotations")
        f.write("\n")
        for (l, o, i) in zip(lens, onsets, interruptors):
            f.write(f"{i} {o} {l}")
            f.write("\n")
        f.write("algorithm annotations")
        f.write("\n")
        # enroll participants
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
            enroll_dict = enroll_speakers(names, approach)
#            #print(enroll_dict)
        # prep wav path 
        wav = os.path.join(wav_folder, wav_file)
        # apply vad
        boundaries = vad.chunk(wav)
        # get the sample rate
        signal, fs = torchaudio.load(wav)
        # detect speech overlap
        #import sounddevice as sd
        for (begin, end) in boundaries:
            chunk = signal[0][int(float(begin)*fs): int(float(end)*fs)]
            #print(begin, end)   
            #sd.play(chunk, fs)
            interruption_details = vad.process_vad(chunk, fs, enroll_dict, begin, approach)
            if len(interruption_details) > 1:
                for (i, o, l) in interruption_details[1:]:
                    f.write(f"{i} {o} {l}\n")
    sys.exit()
#import seaborn as sns
#import numpy as np
#import matplotlib as mpl
#label_size = 20
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size
#print(b_len)
#b_len.remove(8)
#print(b_len)
#b_len = np.array(b_len)
#sns.histplot(data=b_len, x="interruption onset relative to first speaker [second]")
#sns.histplot(data=b_len, binwidth=2)
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
