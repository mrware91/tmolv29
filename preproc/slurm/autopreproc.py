import subprocess
import requests
import time
import os
from cmdInput import *

if runStop == 1:
    runStop = run

def is_run_saved(run, exp=exp):
    try:
        location = "SLAC"
        url = f"https://pswww.slac.stanford.edu/ws/lgbk/lgbk/{exp}/ws/{run}/files_for_live_mode_at_location"
        r = requests.get(url, params={"location": location})
        data = r.json()
        if data['success'] and data['value']['all_present'] and data['value']['is_closed']:
            return True
        else:
            return False
    except: #if it doesn't exist it throws an error
        return False
    
def submit_bjob(exp, run, numcores=32,  nevent=100000, directory='.', queue='psfehq'):
    cmd = ['./slurmJob.sh', '--cores=%d' % numcores, '--run=%d'%run, '--exp='+exp, '--nevent=%d' % nevent,'--directory='+directory, '--python=preproc.py', '--queue=%s' % queue]
    call = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    print(call.stdout)
    

print("Now looking for run %d" % runNumber, end='')
while runNumber <= runStop:
    if is_run_saved(runNumber):
        submit_bjob(exp, runNumber, numcores=cores, nevent=nevents, directory=directory, queue=queue)
        runNumber += 1
        print("Now looking for run %d" % runNumber, end='', flush=True)
    else:
        print(".", end='', flush=True)
        time.sleep(10)
