## Contributors: Matt Ware
import numpy as np
import psana as ps
import sys
import os
import sys
sys.path.insert(1, '../../xtc')
import data
import loop


from cmdInput import *
from setup import *

filename = directory+'/run-%d.h5' % runNumber

if (os.path.isfile(filename) or os.path.isfile(filename.split('.')[0]+'_part0.h5')):
    raise ValueError('h5 files for run %d already exist! check folder: %s'%(run_number, preprocessed_folder))
    #not sure which one to check for so check for both

else:
    data.H5Writer(exp=exp,
                       runNumber=runNumber,
                       detectors=detectors,
                       analysisDict=analysis,
                       outputDir=directory,
                       ncores=cores,
                       nread=nevents,
                       loopStyle=lambda itr: loop.timeItNode(itr) )
                       
print('Execution done ... exiting')
exit()