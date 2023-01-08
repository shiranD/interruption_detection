import pandas as pd
import os
import pdb

def convertDatetime(dt):
    # convert to seconds
    return dt.minute*60+dt.second

def processXL(path):
    gold = pd.read_excel(path, header=1, usecols=['Speaker', 'TimstampStart', 'TimestampEnd', 'Sentence', 'CONST_EstablishesCG_Interrupts'])
    gold = gold.reset_index()
    # who interrupted to whom?
    all_lens = []
    disturbers = []
    dist_onset = []
    for index, row in gold.iterrows():
        if row['CONST_EstablishesCG_Interrupts']==1.0:
            if index:# just to be on the safe side
                try:
                    finishing = convertDatetime(old_row["TimestampEnd"])
#                    print("finishing", finishing)
                    starting = convertDatetime(row["TimstampStart"])
#                    print("starting", starting)
                    inter_len = starting-finishing
                    all_lens.append(inter_len)
                    disturbers.append(row["Speaker"])
                    dist_onset.append(starting)
#                    print("len", inter_len)
                except:
                    pass
        old_row = row
    return all_lens, disturbers, dist_onset
