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
from g_files import g_drive_access
from csv import reader

wav_folder = "../wav/"
txt_folder = "../transcripts/"
vad = VADer()
fixed_th = 0.5

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
    wav_file = g_drive_access(wav_folder, wav_file)
    if not type(wav_file) == str:
       continue
    # replace .wav with .txt and find its transcript in transcripts folder
    txt_file = "transcript_diarized_timestamped_"+re.sub(".wav", ".txt", wav_file)
    # search for it through gdrive
    txt_file = g_drive_access(txt_folder, txt_file)
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
            enroll_dict = enroll_speakers(names)
            #print(enroll_dict)
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
            interruption_details = vad.process_vad(chunk, fs, enroll_dict, begin)
            if len(interruption_details) > 1:
                for (i, o, l) in interruption_details[1:]:
                    f.write(f"{i} {o} {l}\n")

#fig, ax = plt.subplots(figsize =(10, 7))
#ax.hist(b_len)
#ax.set_title("interruption overlap")
#plt.savefig("all_lens")
#fig, ax = plt.subplots(figsize =(10, 7))
#ax.hist(b_ons)
#ax.set_title("interruption time")
#plt.savefig("all_onsets")
#sys.exit()
