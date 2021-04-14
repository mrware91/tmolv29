## Contributors: Elio Champenois and Matt Ware
import numpy as np
import psana as ps
import sys
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scipy.ndimage.filters import gaussian_filter

exp = 'tmolv2918' #
run_number = int(sys.argv[1])

preprocessed_folder = '/cds/home/m/mrware/Workspace/Workspace/2021-02-tmolw56/out'
filename = preprocessed_folder+'%d.h5' % run_number

if (os.path.isfile(filename) or os.path.isfile(filename.split('.')[0]+'_part0.h5')):
    raise ValueError('h5 files for run %d already exist! check folder: %s'%(run_number, preprocessed_folder))
    #not sure which one to check for so check for both

ds = ps.DataSource(exp=exp, run=run_number) #, dir='/ffb01/data/tmo/tmolv2918/xtc')
smd = ds.smalldata(filename=filename)

update = 50 # Print update (per core)
default_val = -9999.0

######### ion ToF settings ###############
rebin_factor = 10

t0 = 0.190
A = (13.211 - t0) / np.sqrt(100)

mpks = np.array([12, 13, 14, 15, 27, 29, 31, 39, 41, 42, 43, 85, 100])
mpks = np.arange(106)
##########################################

Nfound = 0
Nbad = 0
times = None
Ipk = np.zeros_like(mpks, dtype=float)

for run in ds.runs():
    
    timing = run.Detector("timing")
    hsd = run.Detector("hsd")
    #tmo_opal1 = run.Detector("tmo_opal1")
    tmo_atmopal = run.Detector("tmo_atmopal")
    andor = run.Detector("andor")
    gmd = run.Detector("gmd")
    xgmd = run.Detector("xgmd")
    ebeam = run.Detector("ebeam")
    
    if hasattr(run, 'epicsinfo'):
        epics_strs = [item[0] for item in run.epicsinfo.keys()][1:] # first one is weird
        epics_detectors = [run.Detector(item) for item in epics_strs]

    for nevent, event in enumerate(run.events()):
        
        if nevent%update==0: print("Event number: %d, Valid shots: %d" % (nevent, Nfound))
            
        data = {'epics_'+epic_str: epic_det(event) for epic_str, epic_det in zip(epics_strs, epics_detectors)}
        
        if any(type(val) not in [int, float] for val in data.values()):
            print("Bad EPICS: %d" % nevent)
            Nbad += 1
            continue
            
        hsd_data = hsd.raw.waveforms(event)
        if hsd_data is None:
            print("Bad HSD: %d" % nevent)
            Nbad += 1
            continue
        #im = tmo_opal1.raw.image(event)
        #if im is None:
        #    print("Bad OPAL: %d" % nevent)
        #    Nbad += 1
        #    continue
        im2 = tmo_atmopal.raw.image(event)
        if im2 is None:
            print("Bad OPAL2: %d" % nevent)
            Nbad += 1
            continue
        vls = andor.raw.value(event)
        if vls is None:
            print("Bad VLS: %d (using default vals)" % nevent)
            #Nbad += 1
            #continue
            data['vls'] = default_val * np.ones(2048)
        else:
            vls = vls.mean(0)
            data['vls'] = vls.copy()
            
        evrs = timing.raw.eventcodes(event)
        if evrs is None:
            print("Bad EVRs: %d" % nevent)
            Nbad += 1
            continue
        evrs = np.array(evrs)
        if evrs.dtype == int:
            data['evrs'] = evrs.copy()
        else:
            print("Bad EVRs: %d" % nevent)
        
        bad = False
        for (detname, method), attribs in run.detinfo.items():
            if bad: break
            if (detname not in ['timing', 'hsd', 'tmo_opal1', 'andor', 'epicsinfo']) and not (detname=='tmo_atmopal' and method=='raw'):
                for attrib in attribs:
                    if detname!='tmo_atmopal' or attrib not in ['proj_ref', 'proj_sig']:
                        val = getattr(getattr(locals()[detname], method), attrib)(event)
                        if val is None:
                            if detname in ['ebeam', 'gmd', 'xgmd'] and evrs[161]: # BYKIK gives None for these, but we still want to process the shots
                                val = default_val
                            else:
                                bad = True
                                print("Bad %s: %d" % (detname, nevent))
                                Nbad += 1
                                break
                        data[detname+'_'+attrib] = val
        if bad:
            continue
        
        # get ion Tof waveform data
        if times is None:
            times = hsd_data[0]['times'] * 1e6
            m_qs = ((times - t0)/A)**2
            
        wf = hsd_data[0][0].astype('float')
        wf_diode = hsd_data[9][0].astype('float')
        
        for i in range(4):
            wf[i::4] -= wf[i:250:4].mean(0)
            wf_diode[i::4] -= wf_diode[i:250:4].mean(0)
            
        data['iToF_wf'] = wf.reshape(-1, rebin_factor).mean(1)
        data['diode_wf'] = wf_diode[:5000].copy()
        
        for i, mpk in enumerate(mpks):
            Ipk[i] = wf[(mpk - 0.25 < m_qs) & (m_qs < mpk + 0.25)].mean()
            
        data['Ipk'] = Ipk.copy()
        
        #ATM opal:
        stp = np.sum(im2[360:500,:],axis=0)
        data['atm_proj'] = stp.copy()
        stp2 = np.sum(im2[0:240,:],axis=0)
        data['atm_proj2'] = stp.copy()
        stp3 = np.sum(im2[240:360,:],axis=0)
        data['atm_proj3'] = stp.copy()
        
        valid_data = True
        for key, val in data.items():
            if (type(val) not in [int, float]) and (not hasattr(val, 'dtype')):
                print("Bad data:", key)
                valid_data = False
                break
        
        if valid_data:
            smd.event(event, **data)
            Nfound += 1
        else:
            Nbad += 1
            continue
        
if smd.summary:
    Nbad = smd.sum(Nbad)
    Nfound = smd.sum(Nfound)
    smd.save_summary(Nfound=Nfound, Nbad=Nbad, rebin_factor=rebin_factor, mpks=mpks.copy())
    
smd.done()
    
#if rank == (size - 1):
#    perms = '444' # fo-fo-fo
#    for f in [filename.split('.')[0]+'_part0.h5', filename]:
#        os.chmod(f, int(perms, base=8))
