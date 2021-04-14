# Contributors: Matt Ware
# Standard Python imports
import time
import psana
import numpy as np
import sys

import os
os.environ['PS_SRV_NODES']='1'
 
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'


import time

import loop
import setup_v2 as setup
import plotting_v2 as plotting
                


def update(evt, detectors, analysis, iread):
    for key in detectors.keys():
        get = detectors[key]['get']
        try:
            retrieveData = (iread % detectors[key]['gatherEveryNthShot'] == 0)
        except KeyError as ke:
            if 'gatherEveryNthShot' in str(ke):
                retrieveData = True
                print(f'User did not specify detectors[{key}][\'gatherEveryNthShot\'] defaulting to 1')
                detectors[key]['gatherEveryNthShot'] = 1
            else:
                raise ke
        if retrieveData:
            detectors[key]['shotData'] = get(detectors[key]['det'])(evt)
        else:
            detectors[key]['shotData'] = None
            
        if detectors[key]['shotData'] is None:
            detectors[key]['shotData'] = np.nan
                
    detData = {key: detectors[key]['shotData'] for key in detectors}
    for key, item in analysis.items():
        try:
            updateAnalysis = (iread % item['updateEveryNthShot'] == 0)
        except KeyError as ke:
            if 'updateEveryNthShot' in str(ke):
                updateAnalysis = True
                print(f'User did not specify analysis[{key}][\'updateEveryNthShot\'] defaulting to 1')
                analysis[key]['updateEveryNthShot'] = 1
            else:
                raise ke
        
        if updateAnalysis:
            item['data'] = item['update'](detData, item['data'])

def detectorSetup(run, detectors):
    pskeys = set([detectors[key]['pskey'] for key in detectors.keys()])
    psDetectors = {}
    for pskey in pskeys:
        try:
            psDetectors[pskey] = run.Detector(pskey)
        except psana.psexp.run.DetectorNameError as de:
            print('%s is not implemented, dropping' % pskey)
            psDetectors[pskey] = None

    for key in detectors.keys():
        if psDetectors[detectors[key]['pskey']] is not None:
            detectors[key]['det'] = psDetectors[detectors[key]['pskey']]
        else:
            detectors[key]['get'] = lambda x: lambda y: 0
            detectors[key]['det'] = None


def destination(evt):
    # Note that run, step, and det can be accessed
    # the same way as shown in filter_fn
    n_bd_nodes = size - 3 # for mpirun -n 6, 3 ranks are reserved so there are 3 bd ranks left
    dest = (evt.timestamp % n_bd_nodes) + 1
    return dest


defaultLoopStyle = lambda iterator: loop.timeIt(iterator, printEverySec=10)

def shmemReader(exp,run, detectors, analysisDict, plots, loopStyle=defaultLoopStyle):
    psDetectors = list(set([el['pskey'] for el in detectors.values()]))
    print(f'Using detectors ....{psDetectors}')
    if (exp is None) & (run is None):
        print('Starting in shmem ...')
        ds = psana.DataSource(shmem='tmo')#,
                            #   detectors=psDetectors, destination= destination, batch_size=1)#, batch_size=size-3)
    else:
        print('Running on XTC ...')
        ds=psana.DataSource(exp=exp, run=run,
                            detectors=psDetectors.append('epicsinfo'), destination= destination, batch_size=1)#, batch_size=size-3) #psdm
                              
    smd = ds.smalldata(callbacks=[plotting.callback], batch_size=1)

    ########################################################################
    # loop over runs
    ########################################################################
    for run in ds.runs():
        detectorSetup(run, detectors)
        
        # userAnalysis = analyses.analyses( analysisDict, nread )
        
        iread = 0
        # for nevt, evt in enumerate(loopStyle(run.events())): #loop over events
        for nevt, evt in enumerate((run.events())): #loop over events
            update(evt, detectors, analysisDict, iread)
            
            detData = {key: np.copy(detectors[key]['shotData']) for key in detectors}
            analysisData = {}
            for key, val in analysisDict.items():
                try:
                    postNow = (iread % val['post2MasterEveryNthShot']==0) | (iread<10)
                except KeyError as ke:
                    postNow = True
                if postNow:
                    try:
                        for subkey, subval in val['data'].items():
                            analysisData[f"{key}_{subkey}"] = np.copy(subval)
                    except AttributeError:
                        print(f'Skipping because of {key}')
                        continue
            # print(rank,iread)
            smd.event(evt, rank=rank, iread=iread,**analysisData)
            # callback({'iread':iread, **analysisData})
            iread += 1
            # time.sleep(0.1)
                      
    return None


if __name__ == "__main__":
    print(sys.argv)
    try:
        exp=str(sys.argv[1])
        run=int(sys.argv[2])
        runType='offline'
    except IndexError as ie:
        print(ie)
        if sys.argv[1] == 'shmem':
            runType='shmem'
        else:
            print('incorrect input')
            print('provide python TMOOnline.py exp run nevent or ...')
            print('python TMOOnline.py shmem nevent')
            exit
    
    # if rank==size-1:
    #     masterPlotter(numClients)
    # else:
    if runType == 'shmem':
        shmemReader(None,None, setup.detectors, setup.analysis, setup.plots,  loopStyle=defaultLoopStyle)
    else:
        shmemReader(exp,run, setup.detectors, setup.analysis, setup.plots, loopStyle=defaultLoopStyle)
    
    MPI.Finalize()