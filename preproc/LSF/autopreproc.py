## Contributors: Taran Driver and Matt Ware
import subprocess
import requests
import time
import os

print("Start!")

exp="tmolv2918"
run = 185
run_stop = 191

sh_folder = "/cds/home/m/mrware/Workspace/Workspace/2021-02-tmolw56/"
sh_file = "bsub.sh"

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
    
def submit_bjob(run, numcores=32, sh_file=sh_file, sh_folder=sh_folder):
    
    cmd = ["sh", sh_folder+sh_file, str(run), str(numcores)]
    call = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, cwd=sh_folder)
    print(call.stdout)
    

print("Now looking for run %d" % run, end='')
while run <= run_stop:
    if is_run_saved(run):
#         print('run %i good!\n'%run, flush=True)
        submit_bjob(run)
        run += 1
        print("Now looking for run %d" % run, end='', flush=True)
    else:
        print(".", end='', flush=True)
        time.sleep(10)
