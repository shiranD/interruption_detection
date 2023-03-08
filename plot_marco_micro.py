from collections import defaultdict
from matplotlib import pyplot as plt
import os
import pdb
import matplotlib as mpl
              
def process_vars(var_dir, ax1, ax2):

    macro_var=defaultdict(lambda: defaultdict(list))
    micro_var=defaultdict(lambda: defaultdict(list))
    for fname in os.listdir(var_dir):
        with open(os.path.join(var_dir,fname), 'r') as g:
            if 'window' in var_dir:
                var = fname.split("_")[0]
            if 'kernel' in var_dir:
                var = fname.split("_")[1]
            if 'cos' in var_dir:
                #print(fname)
                fname=fname.replace(".txt", "")
                var = fname.split("_")[2]
            for line in g.readlines():
                if "approach" in line:
                    now = line.split()[0]
                if "avg recalls of sessions of interruption:" in line:
                    macro_rc = float(line.split("\t")[-1])
                if "avg percision of sessions of interruption:" in line:
                    macro_pr = float(line.split("\t")[-1])
                    macro_var[now][var] = [macro_rc, macro_pr]
                    micro_var[now][var] = [micro_rc, micro_pr]
                if "recall of interruption:" in line:
                    micro_rc = float(line.split("\t")[-1])
                if "percision of interruption:" in line:
                    micro_pr = float(line.split("\t")[-1])
    
    #pdb.set_trace()
    
    data_macro = defaultdict(list)
    data_micro = defaultdict(list)
    max_maa = 0
    max_mii = 0
    size=20
    color=['blue', 'red', 'orange', 'magenta'] 
    style=['o', 'v', 's', 'D']
    for apch in macro_var.keys():
        for var in macro_var[apch].keys():
           macro_rc, macro_pr = macro_var[apch][var]
           #print(macro_rc, macro_pr)
           micro_rc, micro_pr = micro_var[apch][var]
           #print(micro_rc, micro_pr)
           data_macro[apch].append((macro_rc, macro_pr))
           data_micro[apch].append((micro_rc, micro_pr))
        #   ax1.annotate(str(var), xy=(macro_rc, macro_pr))
        #   ax2.annotate(str(var), xy=(micro_rc, micro_pr))
           # optimize for std and avg
           if macro_rc+macro_pr>max_maa:
               max_maa=macro_rc+macro_pr
               max_ma=[macro_rc,macro_pr]
               var_ma=var 
           if micro_rc+micro_pr>max_mii:
               max_mii=micro_rc+micro_pr
               max_mi=[micro_rc,micro_pr]
               var_mi=var 
        #print(var_dir, apch, max_ma, var_ma, max_mi, var_mi)
        ax1.annotate(str(var_ma), xy=(max_ma[0], max_ma[1]), fontsize=size)
        ax2.annotate(str(var_mi), xy=(max_mi[0], max_ma[1]), fontsize=size)
    for j, apch in enumerate(data_macro.keys()):
        #print(apch)
        #print(data_macro[apch])
        print(apch)
        if 'composition' in apch:
            alabel='cmp'
        if 'pyaudio' in apch:
            alabel = 'pyd'
        if 'single' in apch:
            alabel+='_s'
        ax1.scatter(*zip(*data_macro[apch]),label=alabel, alpha=0.5, s=100, color=color[j], marker=style[j])
        ax2.scatter(*zip(*data_micro[apch]), label=alabel, alpha=0.5, s=100, color=color[j], marker=style[j])
       # print(apch)
       # if "single" in apch:
       #     ax1.scatter(*zip(*[[max_ma[0], max_ma[1]]]))
       #     ax2.scatter(*zip(*[[max_mi[0], max_mi[1]]]))
       # else:
       #     ax1.scatter(*zip(*data_macro[apch]))
       #     ax2.scatter(*zip(*data_micro[apch]))
    ax1.set_ylim([0.10,0.35])
    ax2.set_ylim([0.15,0.5])
    ax1.set_xlim([0.05,0.8])
    ax2.set_xlim([0.05,0.85])
    ax1.tick_params(axis='x', labelsize=size)
    ax1.tick_params(axis='y', labelsize=size)
    ax2.tick_params(axis='x', labelsize=size)
    ax2.tick_params(axis='y', labelsize=size)
    ax1.grid(True)
    ax2.grid(True)
#var_dirs = ["results/kernel/stats/", "results/cos/stats/"]
var_dirs = ["results/window/stats/", "results/kernel/stats/", "results/cos/stats/"]
size=25
figure, ax = plt.subplots(2,3, figsize=(15,10))
mpl.rcParams['xtick.labelsize'] = size
for i, var_dir in enumerate(var_dirs):
    print(i)
    process_vars(var_dir, ax[0,i], ax[1,i])
ax[0,0].set_title(r'window ($w$)'+'\n'+r'($k$=5,$c$=0.99)', size = size-3)
ax[0,1].set_title("kernel (k)\n(w=0.5,c=0.99)", size= size-3)
ax[0,2].set_title("cosine threshold (c)\n(k=5,w=0.5)",fontsize = size-3)
ax[0,0].set_ylabel("macro", fontsize = size)
ax[1,0].set_ylabel("micro", fontsize = size)
#ax[0,0].set_title("macro", size = size)
#ax[0,1].set_title("micro", size= size)
#ax[0,0].set_ylabel("window (w)\n(k=5,c=0.99)", fontsize = size-3)
#ax[1,0].set_ylabel("kernel (k)\n(w=0.5,c=0.99)", fontsize = size-3)
#ax[2,0].set_ylabel("cosine threshold (c)\n(k=5,w=0.5)",fontsize = size-3)
ax[0,1].set_ylabel("percision", fontsize = size-3)
ax[0,1].set_xlabel("recall", fontsize = size-3)
ax[1,0].legend(fontsize=size-5,loc='upper right')

# Shrink current axis's height by 10% on the bottom
#box = ax[2,1].get_position()
#ax[2,1].set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])

# Put a legend below current axis
#ax[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, borderaxespad=0, frameon=False, fontsize=size)

figure.tight_layout()
figure.savefig('mac_toy.pdf')

