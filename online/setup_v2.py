# Contributors: Matt Ware
#psplot -s drp-neh-cmp007 pulseEnergy tofSS tofAvg tofHist numhits numhitsVxgmd numhitsVgmd ttfltpos ttfltposHist ttSSdiv ttAvgBG ttfltposfwhm ttfltposfwhmHist ttamplVgmd ttSS ttAvg pulseEnergyHist vls vlsSumVgmd vlsCoMVL3 vitaraDelay ttcompensated diodeSum diodeSumVttampl diodeAvg diodeSS evrsHistogram

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import sys
sys.path.insert(1, '../xtc')
import FFT_peakfinder

defaultNSAVE = int(3600/(size))

def goodIdx(arr):
    idxs = np.arange(arr.size)
    theIdx=idxs[~np.isnan(arr)][0]
    # print(theIdx)
    return theIdx
    
    
def timestampFormat(ts):
    # print(ts.size, np.unique(ts).size)
    # return np.argsort(ts)
    return (ts-np.nanmin(ts))*1e-9/4.
    
def checkNone(checkThese):
    for el in checkThese:
        if el is None:
            return True
    return False
    

# detectors['sample']={'pskey':'timing', 'get':lambda det: det}

# Analysis is of form {analysisKey: {'function': analysisFunction(), 'detectorKey': 'key', 'analyzeEvery':1}}
# Function element is optional. If not provided, raw data is returned.

###############################################################
### Specify detectors
###############################################################

detectors = {}
analysis = {}
plots = {}
# Fast detectors
# detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
# detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image, 'gatherEveryNthShot':10}
# detectors['vls']={'pskey':'andor', 'get':lambda det: det.raw.value}
# detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
# detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms, 'gatherEveryNthShot':10}
# detectors['photonEnergy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPhotonEnergy}

# # Important Epics
# detectors['vitaraDelay']={'pskey':'las_fs14_target_time', 'get':lambda det: det}

###############################################################
### Evrs Histogram
###############################################################



detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}

def evrsHistogram(dets, data, redfac=.9):
    if dets['evrs'] is None:
        return data
        
    if data is None:
        data={}
        data['count'] = 1
        data['evrsum'] = np.copy(dets['evrs'])
    else:
        data['count'] = data['count']*redfac + 1
        data['evrsum'] = data['evrsum']*redfac + dets['evrs']
        
    return data
        
analysis['evrsHistogram'] = {'update': evrsHistogram,'data':None,'updateEveryNthShot':1}

def xfuncEvrsHistogram(data):
    NC,NH = data['evrsHistogram_evrsum'].shape
    return np.arange(NH+1)
    
def yfuncEvrsHistogram(data):
    sumHist = np.nansum(data['evrsHistogram_evrsum'],0)
    sumCount = np.nansum(data['evrsHistogram_count'],0)
    if sumCount == 0:
        sumCount = 1
    return sumHist/sumCount

plots['evrsHistogram'] = {
    'type':'Histogram',
    'dataSource':'analysis',
    'xfunc': xfuncEvrsHistogram,
    'yfunc': yfuncEvrsHistogram,
    'xlabel': 'evr',
    'ylabel':'counts'
}

# analysis['photonEnergy'] = {'detectorKey':'photonEnergy'}

###############################################################
### pulseEnergy plots
###############################################################

detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['timestamp'] = {'pskey':'timing', 'get': lambda det: lambda evt: evt.timestamp}

def pulseEnergy(dets, data, nsave= defaultNSAVE, NH=100, redfac=.99 ):
    pulseEnergy = dets['gmd']
    timestamp = dets['timestamp']
    if (pulseEnergy is None) | (timestamp is None):
        return data
        
    if data is None:
        data={}
        data['pulseEnergy'] = np.zeros(nsave)*np.nan
        data['num'] = 0
        data['timestamps'] = np.zeros(nsave)*np.nan
        
        
        data['numHist'] = 0
        data['pulseEnergyHist'] = np.zeros(NH)
        data['pulseEnergyEdges'] = np.linspace(0,1,NH+1)
    
    data['pulseEnergy'][data['num']%nsave] = pulseEnergy
    data['timestamps'][data['num']%nsave] = timestamp
    
    data['num']+=1
    
    data['pulseEnergyHist'] = data['pulseEnergyHist']*redfac
    try:
        data['pulseEnergyHist'][int(pulseEnergy*float(NH))] += 1
        data['numHist'] = redfac*data['numHist'] + 1
    except:
        pass
        
    return data

analysis['pulseEnergy'] = {'update': pulseEnergy,'data':None}


plots['pulseEnergy'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['pulseEnergy_timestamps'][:]),
    'yfunc': lambda data: data['pulseEnergy_pulseEnergy'][:],
    'xlabel': 'timestamps',
    'ylabel':'Pulse Energy (mJ)','formats':'.'
}


plots['pulseEnergyHist'] = {
    'type':'Histogram',
    'xfunc': lambda data: np.nanmean(data['pulseEnergy_pulseEnergyEdges'],0),
    'yfunc': lambda data: np.nansum(data['pulseEnergy_pulseEnergyHist'],0)/np.nansum(data['pulseEnergy_numHist']),
    'ylabel': 'counts',
    'xlabel':'Pulse Energy (mJ)'
}


###############################################################
### hsd plots
###############################################################
# detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms, 'gatherEveryNthShot':10}

# resample = lambda x, rebin_factor: x.reshape(-1, rebin_factor).mean(1)

# def hsdFunc(dets, data, nsave=int(100/(size-3)), redfac=0.9 ):
#     hsd = dets['hsd']
    
#     if (hsd is None):
#         return data
        
#     try:
#         hsd.keys()
#     except AttributeError as ae:
#         if 'float' in str(ae):
#             return data
#         else:
#             raise ae
        
#     itofWaveform = resample(hsd[0][0],10)
#     diodeWaveform = resample(hsd[9][0][:5000],10)
        
#     if data is None:
#         data={}
#         data['diodeTime'] = resample(hsd[9]['times'][:5000],10)
#         data['itofTime'] = resample(hsd[0]['times'],10)
#         data['num'] = 0
        
#         idims = np.shape(itofWaveform)
#         ddims = np.shape(diodeWaveform)
#         data['itofSS'] = np.zeros((*idims))
#         data['diodeSS'] = np.zeros((*ddims))
#         data['itofAvg'] = np.zeros((*idims))
#         data['diodeAvg'] = np.zeros((*ddims))
#         # data['itofWaveforms'] = np.zeros((nsave,*idims))*np.nan
#         # data['diodeWaveforms'] = np.zeros((nsave,*ddims))*np.nan
    
#     # data['itofWaveforms'][data['num']%nsave,] = itofWaveform
#     # data['diodeWaveforms'][data['num']%nsave,] = diodeWaveform
    
    
#     data['itofSS'] = itofWaveform
#     data['diodeSS'] = diodeWaveform
#     data['itofAvg'] = itofWaveform + redfac * data['itofAvg']
#     data['diodeAvg'] = diodeWaveform + redfac * data['diodeAvg']
    
#     data['num'] = 1 + redfac*data['num']
        
#     return data

# analysis['hsd'] = {'update': hsdFunc,'data':None}

# plots['diodeSS'] = {
#     'type':'XY',
#     'xfunc': lambda data: data['hsd_diodeTime'][0,:],
#     'yfunc': lambda data: data['hsd_diodeSS'][0,:],
#     'xlabel': 'Time (us)',
#     'ylabel':'Waveform (arb)','formats':'-'
# }

# plots['diodeAvg'] = {
#     'type':'XY',
#     'xfunc': lambda data: data['hsd_diodeTime'][0,:],
#     'yfunc': lambda data: np.nansum(data['hsd_diodeAvg'],0)/np.nansum(data['hsd_num'],0),
#     'xlabel': 'Time (us)',
#     'ylabel':'Waveform (arb)','formats':'-'
# }

# plots['itofSS'] = {
#     'type':'XY',
#     'xfunc': lambda data: data['hsd_itofTime'][0,:],
#     'yfunc': lambda data: data['hsd_itofSS'][0,:],
#     'xlabel': 'Time (us)',
#     'ylabel':'Waveform (arb)','formats':'-'
# }

# plots['itofAvg'] = {
#     'type':'XY',
#     'xfunc': lambda data: data['hsd_itofTime'][0,:],
#     'yfunc': lambda data: np.nansum(data['hsd_itofAvg'],0)/np.nansum(data['hsd_num'],0),
#     'xlabel': 'Time (us)',
#     'ylabel':'Waveform (arb)','formats':'-'
# }



##############################################################
## timetool 2D plots
##############################################################
detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image, 'gatherEveryNthShot':1}
detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
def ttFunc(dets, data, redfac=0.99 ):
    evrs = dets['evrs']
    ttimg = dets['tmo_atmopal']
    gmd = dets['gmd']
    
    if (ttimg is None) | (evrs is None) | (gmd is None):
        return data
        
    
    if (np.nanmedian(ttimg[:]) < -40):
        return data
    
    xrayOff = bool(evrs[161])
    
    if data is None:
        data={}
        data['num'] = 0
        
        ddims = np.shape(ttimg)
        data['SS'] = np.zeros_like(ttimg)
        data['Avg'] = np.zeros_like(ttimg)
    
    if (~xrayOff)&(gmd>0.2):
        data['SS'] = ttimg
    if xrayOff:
        data['Avg'] = ttimg + redfac * data['Avg']
        data['num'] = 1 + redfac*data['num']
        
    return data
analyzeEvery = 1
plotEvery = 0
analysis['tt'] = {'update': ttFunc,'data':None, 'post2MasterEveryNthShot':analyzeEvery}

plots['ttSS'] = {
    'type':'Image',
    'imageFunc': lambda data: data['tt_SS'][goodIdx(data['tt_num']),:],
    'plotEveryNthSec':plotEvery
}

plots['ttSSdiv2D'] = {
    'type':'Image',
    'imageFunc': lambda data: data['tt_SS'][goodIdx(data['tt_num']),:]/(np.nansum(data['tt_Avg'],0)/np.nansum(data['tt_num'],0)),
    'plotEveryNthSec':plotEvery
}

plots['ttAvg'] = {
    'type':'Image',
    'imageFunc': lambda data: np.nansum(data['tt_Avg'],0)/np.nansum(data['tt_num'],0),
    'plotEveryNthSec':plotEvery
}


#############################################################
# timetool 1D plots
#############################################################
detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image, 'gatherEveryNthShot':1}
detectors['ttfltpos']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.fltpos, 'gatherEveryNthShot':1}
detectors['ttfltposfwhm']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.fltposfwhm, 'gatherEveryNthShot':1}
detectors['ttampl']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.ampl, 'gatherEveryNthShot':1}
detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['timestamp'] = {'pskey':'timing', 'get': lambda det: lambda evt: evt.timestamp}
detectors['vitaraDelay']={'pskey':'las_fs14_target_time', 'get':lambda det: det}
# detectors['vitaraDelay']={'pskey':'y_corr_und_40', 'get':lambda det: det}

def tt1dFunc(dets, data, redfac=0.99, redfacHist=.9999, nsave=defaultNSAVE,ttmax=1024):
    evrs = np.array(dets['evrs'],dtype=bool)
    ttimg = dets['tmo_atmopal']
    ttfltpos = dets['ttfltpos']
    ttfltposfwhm = dets['ttfltposfwhm']
    ttampl = dets['ttampl']
    timestamps = dets['timestamp']
    gmd = dets['gmd']
    vitaraDelay = dets['vitaraDelay']
    
    checkThese = [ttimg,evrs,timestamps,ttfltpos,ttfltposfwhm,ttampl,gmd,vitaraDelay]
    
    for el in checkThese:
        if el is None:
            return data
    
    if (np.nanmedian(ttimg[:]) < -40):
        return data
        
    xrayOff= evrs[161]
    tt1d = np.nansum(ttimg[10:400,:],0)
    # tt1d = np.sum(ttimg[:,:],0)
    if data is None:
        data={}
        data['num'] = 0
        
        ddims = np.shape(ttimg)
        data['SS'] = np.zeros_like(tt1d)*np.nan
        data['AvgBG'] = np.zeros_like(tt1d)*np.nan
        
        data['nevents'] = 0
        data['ttfltpos'] = np.zeros(nsave)*np.nan
        data['ttfltposfwhm'] = np.zeros(nsave)*np.nan
        data['ttampl'] = np.zeros(nsave)*np.nan
        data['timestamps'] = np.zeros(nsave)*np.nan
        data['gmd'] = np.zeros(nsave)*np.nan
        data['vitaraDelay'] = np.zeros(nsave)*np.nan
        
        data['ttfltposHist'] = np.zeros(ttmax)
        data['ttfltposfwhmHist'] = np.zeros(ttmax)
        data['ttfltposEdges'] = np.arange(0,ttmax+1)
        data['numOn'] = 0
    
    
    data['ttfltpos'][data['nevents']%nsave] = ttfltpos
    
    data['ttfltposfwhm'][data['nevents']%nsave] = ttfltposfwhm
    
    data['ttampl'][data['nevents']%nsave] = ttampl

    data['gmd'][data['nevents']%nsave] = gmd
    data['vitaraDelay'][data['nevents']%nsave] = vitaraDelay
    data['timestamps'][data['nevents']%nsave] = timestamps
    data['nevents'] += 1
            
    data['ttfltposHist'] = redfacHist*data['ttfltposHist']
    data['ttfltposfwhmHist'] = redfacHist*data['ttfltposfwhmHist']
    try:
        data['ttfltposHist'][int(ttfltpos)] += 1
        data['ttfltposfwhmHist'][int(ttfltposfwhm)] += 1
        data['numOn'] = 1 + redfacHist*data['numOn']
    except IndexError as ie:
        pass
    
    if ~xrayOff:
        data['SS'] = tt1d
    
    else:
        if data['num'] == 0:
            data['AvgBG'] = tt1d
        else:
            data['AvgBG'] = tt1d + redfac * data['AvgBG']
        data['num'] = 1 + redfac*data['num']
        
    return data

analysis['tt1d'] = {'update': tt1dFunc,'data':None, 'post2MasterEveryNthShot':1}

def yfuncttSSdiv(data):
    if np.nansum(data['tt1d_num'],0) <= 0:
        return np.zeros_like(data['tt1d_SS'][0,:])
    else:
        idx = goodIdx(data['tt1d_SS'][:,0])
        y=data['tt1d_SS'][idx,:]/(np.nansum(data['tt1d_AvgBG'],0)/np.nansum(data['tt1d_num'],0))
        return y/y[0]

plots['ttSSdiv'] = {
    'type':'XY',
    'xfunc': lambda data: np.arange(data['tt1d_SS'][0,:].size),
    'yfunc': yfuncttSSdiv,
    'plotEveryNthSec':.5
}

def yfuncttavgBG(data):
    if np.nansum(data['tt1d_num'],0) <= 0:
        return np.zeros_like(data['tt1d_SS'][0,:])
    else:
        return np.nansum(data['tt1d_AvgBG'],0)/np.nansum(data['tt1d_num'],0)

plots['ttAvgBG'] = {
    'type':'XY',
    'xfunc': lambda data: np.arange(data['tt1d_SS'][0,:].size),
    'yfunc': yfuncttavgBG,
    'plotEveryNthSec':.5
}

plots['ttfltpos'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['tt1d_timestamps'][:]),
    'yfunc': lambda data: data['tt1d_ttfltpos'][:],
    'xlabel': 'Time (s)',
    'ylabel':'ttfltpos','formats':'.'
}

plots['ttfltposHist'] = {
    'type':'Histogram',
    'xfunc': lambda data: np.nanmean(data['tt1d_ttfltposEdges'],0),
    'yfunc': lambda data: np.nansum(data['tt1d_ttfltposHist'],0)/np.nansum(data['tt1d_numOn'],0),
    'xlabel': 'ttfltpos',
    'ylabel':'counts'
}

plots['ttfltposfwhm'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['tt1d_timestamps'][:]),
    'yfunc': lambda data: data['tt1d_ttfltposfwhm'][:],
    'xlabel': 'Time (s)',
    'ylabel':'ttfltposfwhm','formats':'.'
}

plots['ttfltposfwhmHist'] = {
    'type':'Histogram',
    'xfunc': lambda data: np.nanmean(data['tt1d_ttfltposEdges'],0),
    'yfunc': lambda data: np.nansum(data['tt1d_ttfltposfwhmHist'],0)/np.nansum(data['tt1d_numOn'],0),
    'xlabel': 'ttfltpos',
    'ylabel':'counts'
}

plots['ttamplVgmd'] = {
    'type':'XY',
    'xfunc': lambda data: data['tt1d_gmd'][:],
    'yfunc': lambda data: data['tt1d_ttampl'][:],
    'xlabel': 'Pulse energy gmd (mJ)',
    'ylabel':'ttampl (adu)','formats':'.'
}


plots['vitaraDelay'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['tt1d_timestamps'][:]),
    'yfunc': lambda data: data['tt1d_vitaraDelay'][:],
    'xlabel': 'Time (s)',
    'ylabel':'target time (ns)','formats':'.'
}

plots['ttcompensated'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['tt1d_timestamps'][:]),
    'yfunc': lambda data: data['tt1d_ttfltpos'][:] + -3.49077300e5 * data['tt1d_vitaraDelay'][:],
    'xlabel': 'Time (s)',
    'ylabel':'ttfltpos - vitaraDelay_pixel','formats':'.'
}



    # delay_in_tt = -3.49077300e5 * np.array(data['delay']) + 1.89836311e9
    # delay_in_ns = -2.73063978e-6 * np.array(data['ttfltpos']) + 5.43823129e3
    # y_ns = np.array(data['delay']).astype(float) - delay_in_ns
    # y_tt = np.array(data['ttfltpos']).astype(float) - delay_in_tt

# ###############################################################
# ### MBES plots
# ###############################################################
detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['xgmd']={'pskey':'xgmd', 'get':lambda det: det.raw.energy}
detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms, 'gatherEveryNthShot':1}
detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
detectors['timestamp'] = {'pskey':'timing', 'get': lambda det: lambda evt: evt.timestamp}

resample = lambda x, rebin_factor: x.reshape(-1, rebin_factor).mean(1)

def fix_wf_baseline(hsd_in, bgfrom=500*64):
    hsd_out = np.copy(hsd_in)
    for i in range(4):
        hsd_out[i::4] -= hsd_out[bgfrom+i::4].mean()
    for i in (12, 13, 12+32, 12+32):
        hsd_out[i::64] -= hsd_out[bgfrom+i::64].mean()
    return hsd_out

# 1d hitfinding
# def cfd(x, y, shift=2e-3, threshold=7, deadtime=2e-3):
#     # Simple Constant Fraction Discriminator for hitfinding

#     pixel_shift = int(shift / np.diff(x).mean() / 2)
#     y1, y2 = y[:-2*pixel_shift], y[2*pixel_shift:]
#     x_, y_ = x[pixel_shift:-pixel_shift], y[pixel_shift:-pixel_shift]
#     y3 = y1 - y2
#     peak_idx = np.where((y3[:-1]<0)&(0<=y3[1:])&(y_[1:]>threshold))[0]
#     times, amplitudes = x_[1:][peak_idx], y_[1:][peak_idx]
#     if len(times)==0:
#         return [], []
#     else:
#         deadtime_filter = [0]
#         previous_time = times[0]
#         for i, time in enumerate(times[1:]):
#             if time - previous_time > deadtime:
#                 deadtime_filter.append(i+1)
#                 previous_time = time
#         return times[deadtime_filter], amplitudes[deadtime_filter]
        
def cfd(x, y, pixel_shift=int(2e0), threshold=7):
    # Simple Constant Fraction Discriminator for hit finding
#     pixel_shift = int(shift / np.diff(x).mean() / 2)
    # print(pixel_shift)
    y1, y2 = y[:-2*pixel_shift], y[2*pixel_shift:]
    x_, y_ = x[pixel_shift:-pixel_shift], y[pixel_shift:-pixel_shift]
    y3 = y1 - y2
#     peak_idx = np.where((y3[:-1]>0)&(y3[1:]<=0)&(y3[:-1]>threshold))[0]
    peak_idx = np.where((y3[:-1]>threshold)&(y3[1:]<=threshold))[0]

#     times, amplitudes = x_[:-1][peak_idx], y_[:-1][peak_idx]
    times, amplitudes = x_[1:][peak_idx], y_[1:][peak_idx]
    if len(times)==0:
        return [], []
    else:
        return times, amplitudes
        
# nsaveHere = defaultNSAVE
nsaveHere =10000
def mbesFunc(dets, data, nsave=nsaveHere, redfac=0.9999, beamOff=43 ):
    hsd = dets['hsd']
    evrs = np.array(dets['evrs'],dtype=bool)
    timestamps = dets['timestamp']
    gmd = dets['gmd']
    xgmd= dets['xgmd']
    vitaraDelay = dets['vitaraDelay']
    
    checkThese = [hsd,evrs,timestamps,gmd,xgmd,vitaraDelay]
    
    for el in checkThese:
        if el is None:
            return data
        
    if evrs[161] | (not evrs[70]):# | evrs[beamOff]:
        return data
        
    isGoose = bool(evrs[68])
    isDuck = bool(evrs[67])
        
    try:
        hsd.keys()
    except AttributeError as ae:
        if 'float' in str(ae):
            return data
        else:
            raise ae
        
    rebinFactor = 10
    timesFull = hsd[0]['times'].astype(float)*1e6
    tofWaveformFull = fix_wf_baseline(hsd[0][0].astype(float))
    # tofWaveform = resample(tofWaveformFull,rebinFactor)
    # tofTime = resample(hsd[0]['times'],rebinFactor)* 1e6
    tofWaveform = tofWaveformFull[:10000]
    tofTime = hsd[0]['times'][:10000]* 1e6
    
    hitsMBES=FFT_peakfinder.FFTfind_fixed(hsd, nmax=1000)
    # hitsMBES=cfd(timesFull,(tofWaveformFull))
    notnan=~np.isnan(hitsMBES[0])
    numhitsMBES=np.sum(notnan)
    
    mbes_bins = np.arange(0, 22e-6, 5e-9)
    # mbes_bins = np.arange(0, 22e-6, 200e-9)
    tofHitTimes = hitsMBES[0][notnan]
    hist,edges = np.histogram(tofHitTimes, mbes_bins)
    
    # numeInROI = np.sum( (tofHitTimes > 19.3e-6)&(tofHitTimes <19.7e-6) )
        
    numeInROI = np.sum( (tofHitTimes > .74e-6)&(tofHitTimes <.93e-6) )
        
    if data is None:
        data={}
        data['tofTime'] = tofTime
        data['num'] = 0
        
        idims = np.shape(tofWaveform)
        data['tofSS'] = np.zeros((*idims))
        data['tofAvg'] = np.zeros((*idims))
        data['tofHFAvg'] = np.zeros((*idims))
        
        hdims = np.shape(hist)
        data['tofHist'] = np.zeros((*hdims))
        data['tofEdges'] = edges
        
        data['nevents'] = 0
        data['numhits'] = np.zeros(nsave)*np.nan
        data['gmd'] = np.zeros(nsave)*np.nan
        data['xgmd'] = np.zeros(nsave)*np.nan
        data['timestamps'] = np.zeros(nsave)*np.nan
        
        
        data['numeInROI'] = np.zeros(nsave)*np.nan
        data['numeInROIGoose'] = np.zeros(nsave)*np.nan
        data['vitaraDelay'] = np.zeros(nsave)*np.nan
        
        
        
        data['tofAvgGoose'] = np.zeros((*idims))
        data['tofHistGoose'] = np.zeros((*hdims))
        data['numGoose'] = 0
        data['tofAvgDuck'] = np.zeros((*idims))
        data['tofHistDuck'] = np.zeros((*hdims))
        data['numDuck'] = 0
        # data['itofWaveforms'] = np.zeros((nsave,*idims))*np.nan
        # data['diodeWaveforms'] = np.zeros((nsave,*ddims))*np.nan
    
    # data['itofWaveforms'][data['num']%nsave,] = itofWaveform
    # data['diodeWaveforms'][data['num']%nsave,] = diodeWaveform
    
    data['tofSS'] = tofWaveform
    data['tofAvg'] = tofWaveform + redfac * data['tofAvg']
    data['tofHist'] = hist + redfac * data['tofHist']
    data['num'] = 1 + redfac*data['num']
    
    if isGoose & (xgmd>3e-3):
        data['tofAvgGoose'] = tofWaveform + redfac * data['tofAvgGoose']
        data['tofHistGoose'] = hist + redfac * data['tofHistGoose']
        data['numGoose'] = 1 + redfac*data['numGoose']
        
    if isDuck & (xgmd>3e-3):
        data['tofAvgDuck'] = tofWaveform + redfac * data['tofAvgDuck']
        data['tofHistDuck'] = hist + redfac * data['tofHistDuck']
        data['numDuck'] = 1 + redfac*data['numDuck']
    
    data['numhits'][data['nevents']%nsave] = numhitsMBES
    data['gmd'][data['nevents']%nsave] = gmd
    data['xgmd'][data['nevents']%nsave] = xgmd
    data['timestamps'][data['nevents']%nsave] = timestamps
    
    
    data['vitaraDelay'][data['nevents']%nsave] = vitaraDelay
    if isDuck & (xgmd>3e-3):
        data['numeInROI'][data['nevents']%nsave] = numeInROI
    
    
    if isGoose & (xgmd>3e-3):
        data['numeInROIGoose'][data['nevents']%nsave] = numeInROI
    
    data['nevents'] += 1
        
    return data

analysis['mbes'] = {'update': mbesFunc,'data':None}

def goodIdx(arr):
    idx = np.arange(arr.size)
    return idx[~np.isnan(arr)][0]

plots['tofSS'] = {
    'type':'XY',
    'xfunc': lambda data: data['mbes_tofTime'][goodIdx(data['mbes_tofTime'][:,0]),:],
    'yfunc': lambda data: data['mbes_tofSS'][goodIdx(data['mbes_tofTime'][:,0]),:],
    'xlabel': 'Time (us)',
    'ylabel':'Waveform (arb)','formats':'-'
}

plots['tofAvg'] = {
    'type':'XY',
    'xfunc': lambda data: np.nanmean(data['mbes_tofTime'],0),
    'yfunc': lambda data: np.nansum(data['mbes_tofAvg'],0)/np.nansum(data['mbes_num'],0),
    'xlabel': 'Time (us)',
    'ylabel':'Waveform (arb)','formats':'-'
}

plots['tofDiff'] = {
    'type':'XY',
    'xfunc': lambda data: np.nanmean(data['mbes_tofTime'],0),
    'yfunc': lambda data: np.nansum(data['mbes_tofAvgDuck'],0)/np.nansum(data['mbes_numDuck'],0) - np.nansum(data['mbes_tofAvgGoose'],0)/np.nansum(data['mbes_numGoose'],0),
    'xlabel': 'Time (us)',
    'ylabel':'Duck-Goose Waveform (arb)','formats':'-'
}

plots['tofHistDiff'] = {
    'type':'Histogram',
    'xfunc': lambda data: np.nanmean(data['mbes_tofEdges'],0),
    'yfunc': lambda data: np.nansum(data['mbes_tofHistDuck'],0)/np.nansum(data['mbes_numDuck'],0) - np.nansum(data['mbes_tofHistGoose'],0)/np.nansum(data['mbes_numGoose'],0),
    'xlabel': 'Time (us)',
    'ylabel':'Duck-Goose Waveform (arb)','formats':'-'
}

plots['tofHist'] = {
    'type':'Histogram',
    'xfunc': lambda data: np.nanmean(data['mbes_tofEdges'],0),
    'yfunc': lambda data: np.nansum(data['mbes_tofHist'],0)/np.nansum(data['mbes_num'],0),
    'xlabel': 'Time (us)',
    'ylabel':'Electron count hist (arb)'
}

plots['numhits'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['mbes_timestamps'][:]),
    'yfunc': lambda data: data['mbes_numhits'][:],
    'xlabel': 'Time (s)',
    'ylabel':'Number of electron hits','formats':'.'
}

def nhfunc(data,key=''):
    ROI = data[f'mbes_numeInROI{key}'][:]
    Full = data['mbes_numhits'][:]
    delays = data['mbes_vitaraDelay'][:]
    notnan = ~np.isnan(ROI)
    
    vdLo=np.nanmin(delays)-1e-4
    vdHi=np.nanmax(delays)+1e-4
    
    delaySum, delayE= np.histogram( delays[notnan], bins=200, range=(vdLo,vdHi), weights=ROI[notnan] );
    delayCount, delayE= np.histogram( delays[notnan], bins=200, range=(vdLo,vdHi), weights=Full[notnan] );
    delayNorm =delaySum/delayCount
    notnan = delayCount>0
    
    return delayE[1:][notnan], delayNorm[notnan]
    

def nhxfunc(data,key=''):
    return nhfunc(data,key=key)[0]

def nhyfunc(data,key=''):
    return nhfunc(data,key=key)[1]

key=''
plots['numHitsHistVvitaraDelay'] = {
    'type':'XY',
    'xfunc': lambda data: nhxfunc(data,key=''),
    'yfunc': lambda data: nhyfunc(data,key=''),
    'xlabel': 'Vitara Delay (ns)',
    'ylabel': 'Number of particles in ROI to total number','formats':'-'
}

key='Goose'
plots['numHitsHistVvitaraDelayGoose'] = {
    'type':'XY',
    'xfunc': lambda data: nhxfunc(data,key='Goose'),
    'yfunc': lambda data: nhyfunc(data,key='Goose'),
    'xlabel': 'Vitara Delay (ns)',
    'ylabel': 'Number of particles in ROI to total number','formats':'-'
}

plots['numhitsVxgmd'] = {
    'type':'XY',
    'xfunc': lambda data: data['mbes_xgmd'][:],
    'yfunc': lambda data: data['mbes_numhits'][:],
    'xlabel': 'Pulse energy (mJ)',
    'ylabel':'Number of electron hits','formats':'.'
}


plots['numHitsVvitaraDelay'] = {
    'type':'XY',
    'xfunc': lambda data: data['mbes_vitaraDelay'][:],
    'yfunc': lambda data: data['mbes_numeInROI'][:],
    'xlabel': 'Vitara Delay (ns)',
    'ylabel':'Number of particles in ROI','formats':'.'
}

plots['numhitsVxgmd'] = {
    'type':'XY',
    'xfunc': lambda data: data['mbes_xgmd'][:],
    'yfunc': lambda data: data['mbes_numhits'][:],
    'xlabel': 'Vitara delay (ns)',
    'ylabel':'Number of electron hits','formats':'.'
}

plots['numhitsVgmd'] = {
    'type':'XY',
    'xfunc': lambda data: data['mbes_gmd'][:],
    'yfunc': lambda data: data['mbes_numhits'][:],
    'xlabel': 'Pulse energy (mJ)',
    'ylabel':'Number of electron hits','formats':'.'
}

###############################################################
### vls plots
###############################################################
detectors['vls']={'pskey':'andor', 'get':lambda det: det.raw.value}
detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['timestamp'] = {'pskey':'timing', 'get': lambda det: lambda evt: evt.timestamp}
detectors['l3'] = {'pskey':'ebeam', 'get': lambda det: det.raw.ebeamL3Energy}

def vls(dets, data, nsave= defaultNSAVE, NH=100, redfac=.99 ):
    pulseEnergy = dets['gmd']
    timestamp = dets['timestamp']
    vls = dets['vls']
    l3 = dets['l3']
    try:
        vls.shape
    except AttributeError:
        return data
    vls=vls.flatten()
    
    checkThese = [pulseEnergy,timestamp,vls,l3]
    if checkNone(checkThese):
        return data
        
    if data is None:
        data={}
        data['pulseEnergy'] = np.zeros(nsave)*np.nan
        data['l3'] = np.zeros(nsave)*np.nan
        data['num'] = 0
        data['timestamps'] = np.zeros(nsave)*np.nan
        data['vls'] = np.zeros_like(vls)*np.nan
        data['vlsCoM'] = np.zeros(nsave)*np.nan
        data['vlsSum'] = np.zeros(nsave)*np.nan
    
    data['vls']=np.copy(vls)
    data['vlsSum'][data['num']%nsave] = np.nansum(vls)
    data['vlsCoM'][data['num']%nsave] = np.nansum(vls*np.arange(vls.size))/np.nansum(vls)
    data['pulseEnergy'][data['num']%nsave] = pulseEnergy
    data['timestamps'][data['num']%nsave] = timestamp
    data['l3'][data['num']%nsave] = l3
    
    data['num']+=1
        
    return data

analysis['vls'] = {'update': vls,'data':None}

def xfuncVLS(data):
    return np.arange(data['vls_vls'][0,:].size)
plots['vls'] = {
    'type':'XY',
    'xfunc': xfuncVLS,
    'yfunc': lambda data: data['vls_vls'][goodIdx(data['vls_num']),].flatten(),
    'xlabel': 'idx',
    'ylabel':'VLS spectrum (adu)','formats':'-'
}


plots['vlsSumVgmd'] = {
    'type':'XY',
    'xfunc': lambda data: data['vls_pulseEnergy'].flatten(),
    'yfunc': lambda data: data['vls_vlsSum'].flatten(),
    'ylabel': 'VLS sum',
    'xlabel':'Pulse Energy (mJ)','formats':'.'
}


plots['vlsCoMVL3'] = {
    'type':'XY',
    'xfunc': lambda data: data['vls_l3'].flatten(),
    'yfunc': lambda data: data['vls_vlsCoM'].flatten(),
    'ylabel': 'VLS sum',
    'xlabel':'L3 energy','formats':'.'
}


##############################################################
## hsd plots
##############################################################
detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms, 'gatherEveryNthShot':1}
detectors['ttampl']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.ampl, 'gatherEveryNthShot':1}
detectors['timestamp'] = {'pskey':'timing', 'get': lambda det: lambda evt: evt.timestamp}

resample = lambda x, rebin_factor: x.reshape(-1, rebin_factor).mean(1)

def hsdFunc(dets, data, nsave=defaultNSAVE, redfac=0.9 ):
    hsd = dets['hsd']
    ttampl = dets['ttampl']
    timestamps = dets['timestamp']
    
    if (hsd is None):
        return data
        
    try:
        hsd.keys()
    except AttributeError as ae:
        if 'float' in str(ae):
            return data
        else:
            raise ae
        
    diodeTime = resample(hsd[9]['times'],10)*1.0e6
    diodeWaveform = resample(hsd[9][0],10)
        
    if data is None:
        data={}
        data['diodeTime'] = diodeTime
        data['num'] = 0
        
        ddims = np.shape(diodeWaveform)
        data['diodeSS'] = np.zeros((*ddims))
        data['diodeAvg'] = np.zeros((*ddims))
        data['diodeSum'] = np.zeros(nsave)*np.nan
        data['timestamps']= np.zeros(nsave)*np.nan
        data['ttampl'] = np.zeros(nsave)*np.nan
        data['nevents'] = 0
        
    data['diodeSS'] = diodeWaveform
    data['diodeAvg'] = diodeWaveform + redfac * data['diodeAvg']
    # print(diodeWaveform)
    
    data['num'] = 1 + redfac*data['num']
    
    data['timestamps'][data['nevents']%nsave] = timestamps
    data['diodeSum'][data['nevents']%nsave] = np.nansum(diodeWaveform)
    data['ttampl'][data['nevents']%nsave] = ttampl
    data['nevents'] += 1
        
    return data

analysis['hsd'] = {'update': hsdFunc,'data':None}

plots['diodeSS'] = {
    'type':'XY',
    'xfunc': lambda data: data['hsd_diodeTime'][goodIdx(data['hsd_nevents']),:],
    'yfunc': lambda data: data['hsd_diodeSS'][goodIdx(data['hsd_nevents']),:],
    'xlabel': 'Time (us)',
    'ylabel':'Waveform (arb)','formats':'-'
}

plots['diodeAvg'] = {
    'type':'XY',
    'xfunc': lambda data: data['hsd_diodeTime'][goodIdx(data['hsd_nevents']),:],
    'yfunc': lambda data: np.nansum(data['hsd_diodeAvg'],0)/np.nansum(data['hsd_num'],0),
    'xlabel': 'Time (us)',
    'ylabel':'Waveform (arb)','formats':'-'
}

plots['diodeSum'] = {
    'type':'XY',
    'xfunc': lambda data: timestampFormat(data['hsd_timestamps'][:]),
    'yfunc': lambda data: data['hsd_diodeSum'][:],
    'xlabel': 'Time (s)',
    'ylabel':'Diode trace sum','formats':'.'
}

plots['diodeSumVttampl'] = {
    'type':'XY',
    'xfunc': lambda data: data['hsd_ttampl'][:],
    'yfunc': lambda data: data['hsd_diodeSum'][:],
    'xlabel': 'Timetool edge amplitude',
    'ylabel':'Diode trace sum','formats':'.'
}
