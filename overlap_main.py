import torchaudio
import re
import os
import sys
from os.path import exists as file_exists
import pdb
from vad_class import VADer
from enrollment import enroll_speakers
from xl_file_processor import processXL
from matplotlib import pyplot as plt
from retrieve_matching_files import file_matcher
import torch
from csv import reader
import argparse

###############################################################################
# Parsing Arguments
###############################################################################
parser = argparse.ArgumentParser(description='Interruption Detection')
parser.add_argument('--outputdir', type=str, help='path to save the final model results')
parser.add_argument('--txt', type=str, help='folder of human transcripts')
parser.add_argument('--wav', type=str, help='folder of the wav files')
parser.add_argument('--enroll', type=str, help='folder of speakers enrollments types')
parser.add_argument('--xl', type=str, help='folder of TBD interruption types (not used)')
parser.add_argument('--speaker_similarity_th', type=float, default=0.99, nargs='?', help='speaker similarity threshold')
parser.add_argument('--short_turn_th', type=float, default=0.2, nargs='?', help='short turn threshold')
parser.add_argument('--window_size', type=float, default=0.5, nargs='?', help='processing window size')
parser.add_argument('--kernel_size', type=int, default=5, nargs='?', help='kernel size')
args = parser.parse_args()
print(args)
#sys.exit()
vad = VADer(args.speaker_similarity_th)

b_len = []
b_ons = []
b_inter = []

def process(time):
    hr, mn, sc = time.split(":")
    time = int(hr)*60*60+int(mn)*60+float(sc)
    return time
print("window size is ", args.window_size)
print("cos th is ", args.speaker_similarity_th)
print("kernel is ", args.kernel_size)

results_file = str(args.window_size)+"_"+str(args.kernel_size)+"_"+str(args.speaker_similarity_th)+".txt"
f = open(os.path.join(args.outputdir, results_file), "w")
for xl_file in os.listdir(args.xl):
    #xl_file="Beck-White_CPS_Annotated_Crystal_21-11-04_si_l2_p2_video-yeti5_5min.xlsx"
    #xl_file="Beck-White_CPS_Annotated_Crystal_21-11-04_si_l2_p4_yeti5_5min.xlsx"
    #print(xl_file)
#    xl_file = "Beck-White_CPS_Annotated_Crystal_21-11-04_si_l2_p3_video_yeti5_5min.xlsx"
    # extract the name
    base_name = xl_file.split("Beck-White_CPS_Annotated_")[1] 
    base_name = base_name.split(".xlsx")[0] 
    # for every annotated file grab annotations by time and compute length
    lens, interruptors, onsets = processXL(os.path.join(args.xl, xl_file))
    # find wav and transcript
    wav_file = base_name+".wav"
    # search for it through gdrive
    wav_file = file_matcher(args.wav, wav_file)
    if not type(wav_file) == str:
       continue
    # replace .wav with .txt and find its transcript in transcripts folder
    txt_file = base_name+".txt"
    # search for it through gdrive
    txt_file = file_matcher(args.txt, txt_file)
#    # check if current path is a file
    if not type(txt_file) == str:
       continue
    elif file_exists(os.path.join(args.txt, txt_file)):
        # process for overlapping times based on transcription
        f.write("\n")
        f.write("gold human transcripts "+base_name)
        f.write("\n")
        q = open(os.path.join(f"tmp_{args.kernel_size}/",results_file+"_reference_"+base_name), "w")
        with open(os.path.join(args.txt, txt_file), "r") as g:
            prev = 0
            for line in g.readlines():
               student, t1, t2, _ = line.split("\t")
               student = student.replace("Student ","s_")
               q.write(student+"\t"+t1+"\t"+t2+"\n")
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
                       #if t1-prev[1] < -4:
                           #print(base_name, t1-prev[1])
                       f.write("\n")
                   elif t1-prev[1] < args.short_turn_th:
                       f.write(student+"\t"+"shorturn\t"+str(prev[1])+"\t"+str(t1)+"\t"+str(t1-prev[1]))
                       b_len.append(t1-prev[1])
                       f.write("\n")
               prev = [t1, t2]
        names = []
        with open(os.path.join(args.txt, txt_file), "r") as g:
            for line in g.readlines():
               line = line.replace("Student ", "student-")
               line = line.replace("student ", "student-")
               name = line.split()[0] 
               if 'other' not in name and 'teacher' not in name:
                   name = 'enrollment_'+name+'.wav'
                   names.append(name)
            names = set(names)
#        for approach in ['composition_single', 'pyaudio_single', 'composition', 'pyaudio']:
#        for approach in ['pyaudio_xvec']:
        for approach in ['composition_single', 'composition_gf', 'composition_f', 'pyaudio_xvec', 'pyaudio_xvec_single']:
            enroll_dict = enroll_speakers(names, approach, args.enroll)
            f.write(approach+" annotations")
            f.write("\n")
            wav = os.path.join(args.wav, wav_file)
            #scd = torch.hub.load("pyannote/pyannote-audio", "scd")
            #scd = torch.hub.load("pyannote/pyannote-audio", "scd", trust_repo='check')
            #change_scores=scd({"audio": wav})
            # apply vad
            boundaries = vad.chunk(wav)
            # get the sample rate
            signal, fs = torchaudio.load(wav)
            # detect speech overlap
            #import sounddevice as sd
            #with open(base_name+"_"+approach, "w") as d:
            with open(os.path.join(f"tmp_{args.kernel_size}/",results_file+"_"+approach+"_"+base_name), "w") as d:
                for (begin, end) in boundaries:
                    chunk = signal[0][int(float(begin)*fs): int(float(end)*fs)]
                    #print(begin, end)   
                    #sd.play(chunk, fs)
                    interruption_details = vad.process_vad_segment(chunk, fs, enroll_dict, begin, approach, args.kernel_size, args.window_size)
                    for (i, o, l) in interruption_details:
                        if i != 'none':
                            i = i.replace("enrollment_student-","s_")
                            i = i.replace(".wav","")
                            #print(f"{i} {o} {l}\n")
                            d.write(f"{i}\t{o}\t{l}\n")
            if os.stat(os.path.join(f"tmp_{args.kernel_size}/",results_file+"_"+approach+"_"+base_name)).st_size == 0: continue
            with open(os.path.join(f"tmp_{args.kernel_size}/",results_file+"_"+approach+"_"+base_name), "r") as d:
                prev = 0
                for line in d.readlines():
                   student, t1, t2 = line.split("\t")
                   student = student.replace("enrollment_student-","s_")
                   student = student.replace(".wav","")
                   t1=float(t1)
                   t2=float(t2)
                   if not prev:
                       prev = [t1, t2]
                       continue
                   else:
                       if t1-prev[1] <= 0:
                           f.write(student+"\t"+"overlap\t"+str(t1)+"\t"+str(prev[1])+"\t"+str(t1-prev[1]))
                           b_len.append(t1-prev[1])
                           #if t1-prev[1] < -4:
                           #    print(base_name, t1-prev[1])
                           f.write("\n")
                       elif t1-prev[1] < args.short_turn_th:
                           f.write(student+"\t"+"shorturn\t"+str(prev[1])+"\t"+str(t1)+"\t"+str(t1-prev[1]))
                           b_len.append(t1-prev[1])
                           f.write("\n")
                   prev = [t1, t2]
