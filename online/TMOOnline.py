# Standard Python imports
import psana
import numpy as np
import time
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

# Imports for plotting
#from psmon.plots import XYPlot,Image,Hist,MultiPlot
# import psmon.plots
# #from psmon import publish
# import psmon.publish
# psmon.publish.local=False # changeme

# Double-ended queue
from collections import deque

# import online/offline library
import setup

import sys


########################################################################
# prepare data source
########################################################################
try:
    print(sys.argv[2])
    print('running in debug mode')
    ds=psana.DataSource(exp=str(sys.argv[1]), run=int(sys.argv[2])) #psdm
except IndexError as ie:
    if sys.argv[1] == 'shmem':
        print('running in shmem')
        ds=psana.DataSource(shmem='tmo') #shared memory
print('ds initialized')

detectors = setup.setupDetectors()
plots = setup.setupPlots()
pskeys = set([detectors[key]['pskey'] for key in detectors.keys()])

########################################################################
# loop over runs
########################################################################
tic=time.time()
plotEvery0 = 1
for run in ds.runs():
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

    for nevt, evt in enumerate(run.events()): #loop over events
        tic2=time.time()
        for key in detectors.keys():
#             print(key)
            get = detectors[key]['get']
            detectors[key]['shotData'] = get(detectors[key]['det'])(evt)

        for key in plots.keys():
            plots[key]['data']=plots[key]['updater'](plots[key]['data'], detectors)
            try:
                plotEvery = plots[key]['plotEvery']
            except KeyError as ke:
                plotEvery = plotEvery0
            if (nevt>0)&(nevt%plotEvery==0):
#                 print('Plotting '+key)
                plots[key]['data']=plots[key]['plotter']( plots[key]['data'], nevt )

        toc = time.time()
        print('full proc.: %.2f ms, event retrieval: %.2f ms'%(1e3*(toc - tic), 1e3*(tic2 - tic)))
        tic = toc
