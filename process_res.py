from collections import defaultdict
import re
import pdb
import sys
import os
import argparse
fname="0.5_5_toy_0.99.txt"

###############################################################################
# Parsing Arguments
###############################################################################
parser = argparse.ArgumentParser(description='Interruption Detection')
parser.add_argument('--outputdir', type=str, help='path to retreive raw results')
parser.add_argument('--inputdir', type=str, help='path to save the stats results')
parser.add_argument('--speaker_similarity_th', type=float, default=0.2, nargs='?', help='speaker similarity threshold')
parser.add_argument('--window_size', type=float, default=0.5, nargs='?', help='processing window size')
parser.add_argument('--kernel_size', type=int, default=5, nargs='?', help='kernel size')
args = parser.parse_args()
print(args)

results_file = str(args.window_size)+"_"+str(args.kernel_size)+"_"+str(args.speaker_similarity_th)+".txt"

# stats file
f = open(os.path.join(args.outputdir, results_file), 'w')

# raw results file
with open(os.path.join(args.inputdir, results_file), 'r') as g:
    flag=0
    id_events = defaultdict(int) # counts events per approach
    id_events_p_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    # percision (session)
    id_p_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    type_p_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    speaker_p_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    type_spk_p_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    # recall (session)
    id_r_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    type_r_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    speaker_r_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    type_spk_r_session = defaultdict(lambda: defaultdict(int)) # events per approach per session
    # recall (approach)
    id_r = defaultdict(int) # counts events per approach
    types_r = defaultdict(int) # counts events per approach
    speaker_r = defaultdict(int) # counts events per approach
    type_spk_r = defaultdict(int) # counts events per approach
    # percision (approach)
    id_p = defaultdict(int) # counts events per approach
    types_p = defaultdict(int) # counts events per approach
    speaker_p = defaultdict(int) # counts events per approach
    type_spk_p = defaultdict(int) # counts events per approach

    # True Positive
    c_id_events=defaultdict(int)
    c_id_events_p_session=defaultdict(lambda: defaultdict(int))
    c_speaker=defaultdict(int)
    c_speaker_p_session=defaultdict(lambda: defaultdict(int))
    c_type=defaultdict(int)
    c_type_p_session=defaultdict(lambda: defaultdict(int))
    c_type_spk=defaultdict(int)
    c_type_spk_p_session=defaultdict(lambda: defaultdict(int))

    for line in g.readlines():
        #print(line)
        if 'Crystal' in line:
            line = line.strip()
            session = line.split("s C")[1]
            flag=1
            approach="gold"
            gold = {}
        elif "pyaudio" in line or "composition" in line:
            flag=2
            approach=line.split()[0]
            print(approach)
            continue
        elif line == "\n" and flag == 2:
            # percision
            gold_events = id_events_p_session["gold"][session]
            id_p_session[approach][session]=c_id_events_p_session[approach][session]/gold_events
            type_p_session[approach][session]=c_type_p_session[approach][session]/gold_events
            speaker_p_session[approach][session]=c_speaker_p_session[approach][session]/gold_events
            type_spk_p_session[approach][session]=c_type_spk_p_session[approach][session]/gold_events
            # recall
            approach_events = id_events_p_session[approach][session]
            if approach_events == 0:
                id_r_session[approach][session]=0
                type_r_session[approach][session]=0
                speaker_r_session[approach][session]=0
                type_spk_r_session[approach][session]=0
            else:
                id_r_session[approach][session]=c_id_events_p_session[approach][session]/approach_events
                type_r_session[approach][session]=c_type_p_session[approach][session]/approach_events
                speaker_r_session[approach][session]=c_speaker_p_session[approach][session]/approach_events
                type_spk_r_session[approach][session]=c_type_spk_p_session[approach][session]/approach_events
            flag=0
        elif flag==1: # processing a baseline
            a, b, c, d, e =line.split("\t")
            a = a.replace("Student ","s_")
            a = a.replace("student-","s_")
            a = a.replace("student-other", "other")
            a = a.replace("enrollment_student-","s_")
            a = a.replace(".wav","")
            a = a.replace("none", "other")
            if "other" in a or "Teacher" in a: continue
            id_events[approach]+=1 # counts all events per approach
            id_events_p_session[approach][session]+=1
            gold[round(float(c))] = a + "\t" + b+ "\t" + d 
        elif flag==2:
            a, b, c, d, e =line.split("\t")
            a = a.replace("Student ","s_")
            a = a.replace("student-","s_")
            a = a.replace("student-other", "other")
            a = a.replace("enrollment_student-","s_")
            a = a.replace(".wav","")
            a = a.replace("none", "other")
            if "other" in a or "Teacher" in a: continue
            id_events[approach]+=1 # counts all events per approach
            id_events_p_session[approach][session]+=1
           # if  "rystal_21-11-05_SI_L2_p3_Video_Yeti5_5min" in session and approach=="composition":
           #     pdb.set_trace()
            # True Positive
            for i in [-1, 0, 1]:
                #print(round(float(c))+i)
                if round(float(c))+i in gold: # correct detection
                    c_id_events[approach]+=1
                    c_id_events_p_session[approach][session]+=1
                    info = gold[round(float(c))+i]
                    a_c, b_c, d_c = info.split("\t")
                    C = 0
                    if a_c == a: # correct speaker
                       print("in")
                       c_speaker[approach]+=1
                       c_speaker_p_session[approach][session]+=1
                       C = 1
                    if b_c == b: # correct interruption type
                       c_type[approach]+=1
                       c_type_p_session[approach][session]+=1
                       if C==1:
                           c_type_spk[approach]+=1
                           c_type_spk_p_session[approach][session]+=1 
#for approach in ["composition_f", "composition_gf", "composition_single", "pyaudio_xvec", "pyaudio_xvec_single"]:
for approach in ["pyaudio_ecapa_single", "pyaudio_ecapa"]:
    # percision
    print(approach)
    gold_events = id_events["gold"]
    id_p[approach]=c_id_events[approach]/gold_events
    types_p[approach]=c_type[approach]/gold_events
    speaker_p[approach]=c_speaker[approach]/gold_events
    type_spk_p[approach]=c_type_spk[approach]/gold_events
    # recall
    approach_events = id_events[approach]
    if approach_events:
        id_r[approach]=c_id_events[approach]/approach_events
        types_r[approach]=c_type[approach]/approach_events
        speaker_r[approach]=c_speaker[approach]/approach_events
        type_spk_r[approach]=c_type_spk[approach]/approach_events
    f.write("\n\n"+approach+" approach\n")
    #f.write("recall of interruption events:\t"+str(id_p[approach]))
    #f.write("\nrecall of types:\t"+str(types_p[approach]))
    #f.write("\nrecall of speakers:\t"+str(speaker_p[approach]))
    #f.write("\nrecall of types and speakers:\t"+str(type_spk_p[approach]))
    #f.write("\nprecision of interruption events:\t"+str(id_r[approach]))
    #f.write("\nprecision of types:\t"+str(types_r[approach]))
    #f.write("\nprecision of speakers:\t"+str(speaker_r[approach]))
    #f.write("\nprecision of types and speakers:\t"+str(type_spk_r[approach]))
    # per session
    sessions = c_id_events_p_session[approach].keys()
    n_sessions = len(sessions)
    print(n_sessions, "sessions")
    c_id_s = 0
    c_spk_s = 0
    c_type_s = 0
    c_type_spk_s = 0
    rc_s_id = 0
    rc_s_type = 0
    rc_s_spk = 0
    rc_s_type_spk = 0
    pr_s_id = 0
    pr_s_type = 0
    pr_s_spk = 0
    pr_s_type_spk = 0
    for session in sessions:
        c_id_s+= c_id_events_p_session[approach][session]
        c_spk_s+=c_speaker_p_session[approach][session]
        c_type_s+=c_type_p_session[approach][session]
        c_type_spk_s+=c_type_spk_p_session[approach][session]
        rc_s_id+=c_id_events_p_session[approach][session]/id_events_p_session["gold"][session]
        rc_s_type+=c_type_p_session[approach][session]/id_events_p_session["gold"][session]
        rc_s_spk+=c_speaker_p_session[approach][session]/id_events_p_session["gold"][session]
        rc_s_type_spk+=c_type_spk_p_session[approach][session]/id_events_p_session["gold"][session]
        if id_events_p_session[approach][session]:
            pr_s_id+=c_id_events_p_session[approach][session]/id_events_p_session[approach][session]
            pr_s_type+=c_type_p_session[approach][session]/id_events_p_session[approach][session]
            pr_s_spk+=c_speaker_p_session[approach][session]/id_events_p_session[approach][session]
            pr_s_type_spk+=c_type_spk_p_session[approach][session]/id_events_p_session[approach][session]
        print("SPEAKERS",session, c_speaker_p_session[approach][session])     
    f.write("\n\nmicro f-score (TFs/all)\n")
    # percision
   # f.write("\nrecall of interruption:\t" +str(c_id_s/gold_events))
   # f.write("\nrecall of types:\t"+str(c_type_s/gold_events))
   # f.write("\nrecall of speaker:\t"+str(c_spk_s/gold_events))
   # f.write("\nrecall of types and speaker:\t"+str(c_type_spk_s/gold_events))
    # recall
    if approach_events:
   #     f.write("\npercision of interruption:\t" +str(c_id_s/approach_events))
   #     f.write("\npercision of types:\t"+str(c_type_s/approach_events))
   #     f.write("\npercision of speaker:\t"+str(c_spk_s/approach_events))
   #     f.write("\npercision of types and speaker:\t"+str(c_type_spk_s/approach_events))
        f.write("\ninterruption:\t" +str(2*(c_id_s/approach_events)*(c_id_s/gold_events)/((c_id_s/approach_events)+(c_id_s/gold_events))))
        f.write("\ntypes:\t"+str(2*(c_type_s/approach_events)*(c_type_s/gold_events)/((c_type_s/approach_events)+(c_type_s/gold_events))))
        f.write("\nspeaker:\t"+str(2*(c_spk_s/approach_events)*(c_spk_s/gold_events)/((c_spk_s/approach_events)+(c_spk_s/gold_events))))
        f.write("\ntypes_speaker:\t"+str(2*(c_type_spk_s/approach_events)*(c_type_spk_s/gold_events)/((c_type_spk_s/approach_events)+(c_type_spk_s/gold_events))))
    else:
        f.write("no approach events were detected")
    f.write("\n")
    f.write("\nmacro f-score (Pr or RCs/number of sessions)\n")
   # f.write("\navg recalls of sessions of interruption:\t"+str(rc_s_id/n_sessions))
   # f.write("\navg recalls of sessions of types:\t"+str(rc_s_type/n_sessions))
   # f.write("\navg recalls of sessions of speaker:\t"+str(rc_s_spk/n_sessions))
   # f.write("\navg recalls of sessions of type spk:\t"+str(rc_s_type_spk/n_sessions))
   # f.write("\navg percision of sessions of interruption:\t"+str(pr_s_id/n_sessions))
   # f.write("\navg percision of sessions of types:\t"+str(pr_s_type/n_sessions))
   # f.write("\navg percision of sessions of speaker:\t"+str(pr_s_spk/n_sessions))
   # f.write("\navg percision of sessions of type spk:\t"+str(pr_s_type_spk/n_sessions))
    f.write("\navg_sessions interruption:\t"+str((pr_s_id/n_sessions)*(rc_s_id/n_sessions)/((pr_s_id/n_sessions)+(rc_s_id/n_sessions))))
    f.write("\navg_sessions types:\t"+str((pr_s_type/n_sessions)*(rc_s_type/n_sessions)/((pr_s_type/n_sessions)+(rc_s_type/n_sessions))))
    f.write("\navg_ sessions speaker:\t"+str((pr_s_spk/n_sessions)*(rc_s_spk/n_sessions)/((pr_s_spk/n_sessions)+(rc_s_spk/n_sessions))))
    f.write("\navg_ sessions types_speaker:\t"+str((pr_s_type_spk/n_sessions)*(rc_s_type_spk/n_sessions)/((pr_s_type_spk/n_sessions)+(rc_s_type_spk/n_sessions))))
