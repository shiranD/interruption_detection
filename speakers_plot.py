import pdb
from pyannote.core import Annotation, Segment
from pyannote.core import notebook
from matplotlib import pyplot as pet
import os

fname="results/kernel/raw/"
window=0.5
kernel=3
cos=0.99
fname=fname+str(window)+"_"+str(kernel)+"_"+str(cos)
a = 0
b = 200
#session="Crystal_21-11-04_si_l2_p3_video_yeti5_5min"
session="Crystal_21-11-04_si_l2_p3_video_yeti5_5min"
session="Crystal_21-11-04_si_l2_p2_video-yeti5_5min"
session="Crystal_21-11-04_si_l2_p4_yeti5_5min"
session="Crystal_21-11-05_SI_L2_P3_Video_Yeti5_5min"
session="Crystal_21-11-04_si_l2_p2_video-yeti5_5min"
session="Crystal_21-11-11_SI_L3_p3_video_yeti5_5min"
with open(fname,'r') as g:
    # search for res of specific session
    #pdb.set_trace()
    flag=0
    for line in g.readlines():
        print(len(line))
        if 'Crystal' in line:
            line = line.strip()
            session = line.split("s C")[1]
            flag=2
            annotation = Annotation()
            continue
        elif "composition" in line:
            figure, ax = pet.subplots(3, sharex=True)
            notebook.plot_annotation(annotation, ax=ax[0], time=True, legend=True)
            ax[0].set_ylabel("reference")
            ax[0].set_ylim(0,0.8)
            ax[0].set_yticks([])
            #ax[0].set_xlim(a,b)
            flag=1
            annotation = Annotation()
            continue
        elif "pyaudio" in line:
            notebook.plot_annotation(annotation, ax=ax[1], time=True, legend=True)
            ax[1].set_ylabel("composition")
            ax[1].set_ylim(0,0.8)
            #ax[1].set_xlim(a,b)
            ax[1].set_yticks([])
            annotation = Annotation()
            continue
        elif len(line) == 2:
            if flag:
                notebook.plot_annotation(annotation, ax=ax[2], time=True, legend=True)
                ax[2].set(ylabel="pyaudio")
                ax[2].set_ylim(0,0.8)
                ax[2].set_yticks([])
                #ax[2].set_xlim(a,b)
                figure.tight_layout()
                figure.savefig('C'+session+'.png')
          #  if flag == 1: break
        
        elif flag == 2:
            a, b, c, d, e =line.split("\t")
            a = a.replace("Student ","s_")
            a = a.replace("student-","s_")
            a = a.replace("student-other", "other")
            annotation[Segment(float(c), float(d))] = a
        elif flag==1:
            a, b, c, d, e =line.split()
            a = a.replace("enrollment_student-","s_")
            a = a.replace(".wav","")
            a = a.replace("none", "other")
            print(c)
            annotation[Segment(float(c), float(d))] = a
