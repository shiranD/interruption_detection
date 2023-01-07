from collections import defaultdict
import re
import pdb
import sys
fname="0.5_5_toy_0.99.txt"
with open(fname,'r') as g:
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
        if 'Crystal' in line:
            line = line.strip()
            session = line.split("s C")[1]
            flag=1
            approach="gold"
            gold = {}
            continue
        elif "composition" in line: # finishing to process gold
            approach="composition"
            flag=2
            continue
        elif "pyaudio" in line: # finishing to process composition
            # percision
            gold_events = id_events_p_session["gold"][session]
            id_p_session[approach][session]=c_id_events_p_session[approach][session]/gold_events
            type_p_session[approach][session]=c_type_p_session[approach][session]/gold_events
            speaker_p_session[approach][session]=c_speaker_p_session[approach][session]/gold_events
            type_spk_p_session[approach][session]=c_type_spk_p_session[approach][session]/gold_events
            # recall
            approach_events = id_events_p_session[approach][session]
            id_r_session[approach][session]=c_id_events_p_session[approach][session]/approach_events
            type_r_session[approach][session]=c_type_p_session[approach][session]/approach_events
            speaker_r_session[approach][session]=c_speaker_p_session[approach][session]/approach_events
            type_spk_r_session[approach][session]=c_type_spk_p_session[approach][session]/approach_events
            approach="pyaudio"
            #pdb.set_trace()
            continue
        elif line == "\n": # between sessions
            if flag: # finishing to process pyaudio
                # percision
                gold_events = id_events_p_session["gold"][session]
                id_p_session[approach][session]=c_id_events_p_session[approach][session]/gold_events
                type_p_session[approach][session]=c_type_p_session[approach][session]/gold_events
                speaker_p_session[approach][session]=c_speaker_p_session[approach][session]/gold_events
                type_spk_p_session[approach][session]=c_type_spk_p_session[approach][session]/gold_events
                # recall
                approach_events = id_events_p_session[approach][session]
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
            # True Positive
            for i in [-1, 0, 1]:
                if round(float(c))+i in gold: # correct detection
                    c_id_events[approach]+=1
                    c_id_events_p_session[approach][session]+=1
                    info = gold[round(float(c))+i]
                    a_c, b_c, d_c = info.split("\t")
                    C = 0
                    if a_c == a: # correct speaker
                       c_speaker[approach]+=1
                       c_speaker_p_session[approach][session]+=1
                       C = 1
                    if b_c == b: # correct interruption type
                       c_type[approach]+=1
                       c_type_p_session[approach][session]+=1
                       if C==1:
                           c_type_spk[approach]+=1
                           c_type_spk_p_session[approach][session]+=1 
for approach in ["composition", "pyaudio"]:
    # percision
    gold_events = id_events["gold"]
    id_p[approach]=c_id_events[approach]/gold_events
    types_p[approach]=c_type[approach]/gold_events
    speaker_p[approach]=c_speaker[approach]/gold_events
    type_spk_p[approach]=c_type_spk[approach]/gold_events
    # recall
    approach_events = id_events[approach]
    id_r[approach]=c_id_events[approach]/approach_events
    types_r[approach]=c_type[approach]/approach_events
    speaker_r[approach]=c_speaker[approach]/approach_events
    type_spk_r[approach]=c_type_spk[approach]/approach_events
    print(approach+" approach\n")
    print("recall of interruption events:\t"+str(id_p[approach]))
    print("recall of types:\t"+str(types_p[approach]))
    print("recall of speakers:\t"+str(speaker_p[approach]))
    print("recall of types and speakers:\t"+str(type_spk_p[approach]))
    print("precision of interruption events:\t"+str(id_r[approach]))
    print("precision of types:\t"+str(types_r[approach]))
    print("precision of speakers:\t"+str(speaker_r[approach]))
    print("precision of types and speakers:\t"+str(type_spk_r[approach]))
    # per session
   # pdb.set_trace()
    sessions = c_id_events_p_session[approach].keys()
    n_sessions = len(sessions)
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
        pr_s_id+=c_id_events_p_session[approach][session]/id_events_p_session[approach][session]
        pr_s_type+=c_type_p_session[approach][session]/id_events_p_session[approach][session]
        pr_s_spk+=c_speaker_p_session[approach][session]/id_events_p_session[approach][session]
        pr_s_type_spk+=c_type_spk_p_session[approach][session]/id_events_p_session[approach][session]
    
    print("\nmicro (TFs/all)\n")
    # percision
    print("recall of interruption:\t" +str(c_id_s/gold_events))
    print("recall of types:\t"+str(c_type_s/gold_events))
    print("recall of speaker:\t"+str(c_spk_s/gold_events))
    print("recall of types and speaker:\t"+str(c_type_spk_s/gold_events))
    # recall
    print("percision of interruption:\t" +str(c_id_s/approach_events))
    print("percision of types:\t"+str(c_type_s/approach_events))
    print("percision of speaker:\t"+str(c_spk_s/approach_events))
    print("percision of types and speaker:\t"+str(c_type_spk_s/approach_events))
    print("\n")
    print("macro (Pr or RCs/number of sessions)\n")
    print("avg recalls of sessions of interruption:\t"+str(rc_s_id/n_sessions))
    print("avg recalls of sessions of types:\t"+str(rc_s_type/n_sessions))
    print("avg recalls of sessions of speaker:\t"+str(rc_s_spk/n_sessions))
    print("avg recalls of sessions of type spk:\t"+str(rc_s_type_spk/n_sessions))
    print("avg percision of sessions of interruption:\t"+str(pr_s_id/n_sessions))
    print("avg percision of sessions of types:\t"+str(pr_s_type/n_sessions))
    print("avg percision of sessions of speaker:\t"+str(pr_s_spk/n_sessions))
    print("avg percision of sessions of type spk:\t"+str(pr_s_type_spk/n_sessions))
