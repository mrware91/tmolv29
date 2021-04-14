## Contributors: Matt Ware
import numpy as np

###############################################################
### Specify detectors
###############################################################

detectors = {}
# Fast detectors
# detectors['sample']={'pskey':'timing', 'get':lambda det: det}
detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
# detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image}
# detectors['vls']={'pskey':'andor', 'get':lambda det: det.raw.value}
detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms}
detectors['photonEnergy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPhotonEnergy}
detectors['L3Energy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamL3Energy}

# Important Epics
# detectors['vitaraDelay']={'pskey':'las_fs14_target_time', 'get':lambda det: det}

###############################################################
### Specify analyses to perform and write to small data
###############################################################

# Analysis is of form {analysisKey: {'function': analysisFunction(), 'detectorKey': 'key', 'analyzeEvery':1}}
# Function element is optional. If not provided, raw data is returned.
analysis = {}
# analysis['vitaraDelay'] = {'function':lambda x: x, 'detectorKey':'vitaraDelay'}
analysis['evrs'] = {'detectorKey':'evrs'}
# analysis['vls1D'] = {'function': lambda x: np.sum(x,-1), 'detectorKey':'vls'}
analysis['pulseEnergy'] = {'detectorKey':'gmd'}
analysis['photonEnergy'] = {'detectorKey':'photonEnergy'}
analysis['L3Energy'] = {'detectorKey':'L3Energy'}
# analysis['wfTime'] = {'function': lambda x: x[0]['times'], 'detectorKey':'hsd'}

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
def fix_wf_baseline(hsd_in, bgfrom=500*64):
    hsd_out = np.copy(hsd_in)
    for i in range(4):
        hsd_out[i::4] -= hsd_out[bgfrom+i::4].mean()
    for i in (12, 13, 12+32, 12+32):
        hsd_out[i::64] -= hsd_out[bgfrom+i::64].mean()
    return hsd_out

def cfdFixed(hsd,nmax=1000):
    x= hsd[0]['times']
    y= fix_wf_baseline(hsd[0][0].astype(float))
    timesF = np.zeros(nmax)*np.nan
    amplitudesF = np.zeros(nmax)*np.nan
    times,amplitudes = cfd(x,y)
    
    if len(times) > 0:
        timesF[:times.size] = times
        amplitudesF[:times.size] = times
    
    return timesF, amplitudesF

analysis['MBt'] = {'function': lambda x: cfdFixed(x)[0]*1e6, 'detectorKey':'hsd'}

analysis['MBa'] = {'function': lambda x: cfdFixed(x)[1], 'detectorKey':'hsd'}


resample = lambda x, rebin_factor: x.reshape(-1, rebin_factor).mean(1)
analysis['mb-time'] = {'function': lambda x: x[0]['times'].astype(float)[:10000]*1e6, 'detectorKey':'hsd'}
analysis['mb-waveform'] = {'function': lambda x: fix_wf_baseline(x[0][0].astype(float))[:10000], 'detectorKey':'hsd'}
# analysis['diode-waveform'] = {'function': lambda x: x[9][0].astype(float)[:5000], 'detectorKey':'hsd'}

# analysis['atm-proj1'] =  {'function': lambda x: np.sum(x[360:500,:],axis=0), 'detectorKey':'tmo_atmopal'}
# analysis['atm-proj2'] =  {'function': lambda x: np.sum(x[0:240,:],axis=0), 'detectorKey':'tmo_atmopal'}
# analysis['atm-proj3'] =  {'function': lambda x: np.sum(x[240:360,:],axis=0), 'detectorKey':'tmo_atmopal'}