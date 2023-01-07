import pdb
from pyannote.core import Annotation, Segment
from pyannote.core import notebook
from matplotlib import pyplot as pet
import os

fname="figures/window/"
window=0.5
kernel=5
cos=0.99
fname=fname+window+"_"+kernel+"_"+cos
os.makdir(fname)

#session="Crystal_21-11-04_si_l2_p3_video_yeti5_5min"
session="Crystal_21-11-04_si_l2_p3_video_yeti5_5min"
session="Crystal_21-11-04_si_l2_p2_video-yeti5_5min"
session="Crystal_21-11-04_si_l2_p4_yeti5_5min"
with open(fname,'r') as g:
    # search for res of specific session
    pdb.set_trace()
    flag=0
    for line in g.readlines():
        if 'Crystal' in line:
            line = line.strip()
            session = line.split("s C")[1]
            flag=2
            annotation = Annotation()
            continue
        elif "composition" in line:
            figure, ax = pet.subplots(3, sharex=True)
            notebook.plot_annotation(annotation, ax=ax[0], time=True, legend=True)
            ax[0].set(ylabel="reference")
            ax[0].set_ylim(0,0.8)
            ax[0].set_yticks([])
            flag=1
            annotation = Annotation()
            continue
        elif "pyaudio" in line:
            notebook.plot_annotation(annotation, ax=ax[1], time=True, legend=True)
            ax[1].set(ylabel="composition")
            ax[1].set_ylim(0,0.8)
            ax[1].set_yticks([])
            annotation = Annotation()
            continue
        
        elif flag == 2:
            a, b, c, d, e =line.split("\t")
            a = a.replace("Student ","s_")
            a = a.replace("student-","s_")
            a = a.replace("student-other", "other")
            annotation[Segment(float(c), float(d))] = a
        elif line == "\n":
            if flag:
                notebook.plot_annotation(annotation, ax=ax[2], time=True, legend=True)
                ax[2].set(ylabel="pyaudio")
                ax[2].set_ylim(0,0.8)
                ax[2].set_yticks([])
                figure.tight_layout()
                figure.savefig(fname+"/"+session+'.png')
          #  if flag == 1: break
        if flag==1:
            a, b, c, d, e =line.split()
            a = a.replace("enrollment_student-","s_")
            a = a.replace(".wav","")
            a = a.replace("none", "other")
            annotation[Segment(float(c), float(d))] = a
