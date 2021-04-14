## Contributors: Matt Ware
# Standard Python imports
import psana
import numpy as np
import time
import sys
import h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import os
os.environ['PS_SRV_NODES']='1'
import time

import analyses
import loop

def setTypes( adict ):
    types = {}
    for key, el in adict.items():
        types[key] = {'size':np.size(el), 'type':type(el)}
        
def checkTypes(types, adict):
    try:
        for key, el in adict.items():
            if 'StaleFlags' in key:
                print(np.size(el))
                print(type(el))
            if (np.size(el) == types[key]['size'])&( isinstance(el,types[key]['type']) ):
                continue
            else:
                printStr = '%s is of size %f and type ' % (key, float(np.size(el)))
                print(printStr+str(type(el)))
                printStr = 'expected size %f and type ' % float(types[key]['size'])
                print(printStr+str( types[key]['type'] ) )
                print('Skipping this event')
                return False
        return True
    except TypeError as te:
        if 'NoneType' in str(te):
            print('NoneType encountered in ',key,el)
            return False
        else:
            raise te
    
def generateDataset(raw, key, keyword, out):
    if keyword in key:
        detname = key.replace(keyword,'')
        subfolder = keyword.replace('_','')
        if subfolder not in out:
            out.create_group(subfolder)
        out.create_dataset(name = '/%s/%s' % (subfolder,detname), data = raw[ key ])
        return True
    else:
        return False
    
def reformatH5( origFile, outFile ):
    print(origFile)
    raw = h5py.File(origFile, 'r')
    with h5py.File(outFile,'w') as out:
        for key in raw.keys():
            keyword = 'epics_'
            placed = generateDataset(raw, key, keyword, out)
            keyword = 'summary_'
            placed = placed | generateDataset(raw, key, keyword, out)
            if not placed:
                out.create_dataset(name = key, data=raw[key])

def getEpics(run):
    epicsNames = []
    for key in run.epicsinfo:
        epicsNames.append( key[0] )
    return epicsNames

skipEpics = ['StaleFlags']
def makeEpicsDetectorDictionary(run):
    epicsNames = getEpics(run)
    detectors = {}
    for name in epicsNames:
        if name in skipEpics:
            print('Skipping %s' % name)
            continue
        detectors['epics_'+name] = {'pskey':name, 'get':lambda det: det}
    return detectors

def makeEpicsAnalysisDictionary(run):
    epicsNames = getEpics(run)
    analysis = {}
    for name in epicsNames:
        if name in skipEpics:
            print('Skipping %s' % name)
            continue
        analysis['epics_'+name] = {'function':lambda x: x, 'detectorKey':'epics_'+name}
    return analysis

def update(evt, detectors, analysis):
    for key in detectors.keys():
        get = detectors[key]['get']
        detectors[key]['shotData'] = get(detectors[key]['det'])(evt)

    analysis.update(detectors)

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


defaultLoopStyle = lambda iterator: loop.timeIt(iterator, printEverySec=10)

def XTCReader(exp,run, detectors, analysisDict, nread=1000, loopStyle=defaultLoopStyle):
    ds=psana.DataSource(exp=exp, run=run) #psdm
    
    ########################################################################
    # loop over runs
    ########################################################################
    for run in ds.runs():
        detectorSetup(run, detectors)
        theAnalysis = analyses.analyses( analysisDict, nread )
    
        epics = makeEpicsDetectorDictionary(run)
        epicsAnalysisDict = makeEpicsAnalysisDictionary(run)
        epicsAnalysis = analyses.analyses( epicsAnalysisDict, nread, printMode='quiet' )
        detectorSetup(run, epics)
        
        iread = 0
        for nevt, evt in enumerate(loopStyle(run.events())): #loop over events
            update(evt, detectors, theAnalysis)
            update(evt, epics, epicsAnalysis)
            
            iread += 1
            if iread >= nread:
                break
                      
    return theAnalysis.data, epicsAnalysis.data
    
def destination(evt, nodes):
    # Note that run, step, and det can be accessed
    # the same way as shown in filter_fn
    n_bd_nodes = nodes - 3 # for mpirun -n 6, 3 ranks are reserved so there are 3 bd ranks left
    if nodes == 1:
        n_bd_nodes = 1
    dest = (evt.timestamp % n_bd_nodes) + 1
    return dest
    
def H5Writer(exp, runNumber, detectors, analysisDict, outputDir, ncores, nread=1000, loopStyle=defaultLoopStyle, summaryData={}):
    if ncores == 1:
        ncoresUsed = 1
    elif ncores > 3:
        ncoresUsed = ncores - 3
    else:
        raise ValueError('Number of cores must be greater than 3 for MPI')
    
    tempout = outputDir + '/temp-run-%d_.h5' % runNumber
    truetempout = outputDir + '/temp-run-%d__part0.h5' % runNumber
    out = outputDir + '/run-%d.h5' % runNumber
    
    if ncores > 1:
        ds=psana.DataSource(exp=exp, run=runNumber, max_events=nread, batch_size=5, destination= lambda evt: destination(evt, ncores)) #psdm
    else:
        ds=psana.DataSource(exp=exp, run=runNumber, max_events=nread) #psdm
    print('Outputting to %s' % tempout)
    smd = ds.smalldata(filename=tempout)
    run = next(ds.runs())

    detectorSetup(run, detectors)
    theAnalysis = analyses.analyses( analysisDict, 1, printMode='quiet' )

    epics = makeEpicsDetectorDictionary(run)
    epicsAnalysisDict = makeEpicsAnalysisDictionary(run)
    epicsAnalysis = analyses.analyses( epicsAnalysisDict, 1, printMode='quiet' )
    detectorSetup(run, epics)
    
    iread = 0
    nskip = 0
    types = None
    typesEpics = None
    for nevt, evt in enumerate(loopStyle(run.events())): #loop over events
        update(evt, detectors, theAnalysis)
        update(evt, epics, epicsAnalysis)
        
        H5Det = theAnalysis.H5out()
        H5Epics = epicsAnalysis.H5out()
        
        if (H5Det is not None) and (H5Epics is not None):
            if types is None:
                types = theAnalysis.outTypes
                typesEpics = epicsAnalysis.outTypes
#             checkPass = checkTypes(types, H5Det) & checkTypes(typesEpics, H5Epics)
#             if not checkPass:
#                 nskip += 1
#                 continue
            
            smd.event(evt, **H5Det, **H5Epics)
        else:
            nskip += 1
            continue
        
        iread += 1
        if iread >= nread/ncoresUsed:
            break
            
    print('%d rank is done, analyzed %d events' % (rank,iread))
    if smd.summary:
        print('Summarizing data and generating temp H5 ...')
        nskip = smd.sum(nskip)
        iread = smd.sum(iread)
        smd.save_summary(summary_nskip=nskip, summary_iread=iread)
    
    smd.done()
           
    # if rank==0:
    #     print('Generating final H5 ...')
    #     reformatH5( tempout, out )
               
    print('Done.')
    
    if (rank == size-1)&(size > 1):
        print('Watching file %s' % truetempout)
        currSize = os.path.getsize(truetempout)
        time.sleep(60)
        while( os.path.getsize(truetempout) > currSize ):
            currSize = os.path.getsize(truetempout)
            print('Waiting to modify H5 until file is done writing...')
            time.sleep(10)
        print(rank,size)
        print('Generating final H5 ...')
        reformatH5( truetempout, out )
    elif (size==1):
        print('Generating final H5 ...')
        reformatH5(tempout, out)
        
    
    return None
