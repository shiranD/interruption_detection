from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import pdb

def g_drive_access(local_folder, fname):
#    pdb.set_trace()
    fname = (fname.replace('-','_')).lower()
    fname = fname.replace('__','_')
    for file1 in os.listdir(local_folder):
        file1 = (file1.replace('-','_')).lower()
        file1 = file1.replace('__','_')
        if file1 == fname:
            return fname
    return None

def g_drive_all(local_download_path, flag):
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    #gauth.LocalWebserverAuth()
    # Creates local webserver and auto handles authentication
    drive = GoogleDrive(gauth)
    # Auto-iterate through all files that matches this query
    if flag:
        if flag==2:
            # enrol
            file_list = drive.ListFile({'q': "'1vqIVEKdWVvZf3KBVgdOJ8_KbuzM_dndK' in parents"}).GetList()
        else:
            file_list = drive.ListFile({'q': "'1kKwBJvAVvfde-H0JnKEtREpK2yBKtkT4' in parents"}).GetList()
    else:
        file_list = drive.ListFile({'q': "'1cQr-N09xkMr02_F98oO0lvzCb4F6XEME' in parents"}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))
        fname1 = os.path.join(local_download_path, file1['title'])
        f_ = drive.CreateFile({'id': file1['id']})
#        print('downloading to {}'.format(fname))
        f_.GetContentFile(fname1)

#g_drive_all("../transcripts", 0)
#g_drive_all("enr_files", 2)
#g_drive_all("../wav", 1)
