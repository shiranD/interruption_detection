import os

def file_matcher(local_folder, fname):
    
    for file1 in os.listdir(local_folder):
        if file1 == fname:
            print("found ", fname)
            return fname
    fname_o = (fname.replace('-','_')).lower()
    fname_o = fname_o.replace('__','_')
    for file1 in os.listdir(local_folder):
        file1_o = (file1.replace('-','_')).lower()
        file1_o = file1_o.replace('__','_')
        if file1_o == fname_o:
            print("found ", file1)
            return file1
    return None

