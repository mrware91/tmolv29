
## Contributors: Matt Ware
import numpy as np
import sys
sys.path.insert(1, '../../xtc')

###############################################################
### Specify detectors
###############################################################

detectors = {}
# Fast detectors
# detectors['sample']={'pskey':'timing', 'get':lambda det: det}
detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
detectors['vls']={'pskey':'andor', 'get':lambda det: det.raw.value}
detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['xgmd']={'pskey':'xgmd', 'get':lambda det: det.raw.energy}
detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms}

# Timetool
detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image}
detectors['ttfltpos']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.fltpos}
detectors['ttfltposfwhm']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.fltposfwhm}
detectors['ttampl']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.ampl}

# Ebeam parameters
detectors['photonEnergy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPhotonEnergy}
detectors['ebeamCharge']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamCharge}
detectors['ebeamDumpCharge']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamDumpCharge}
detectors['ebeamEnergyBC1']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamEnergyBC1}
detectors['ebeamEnergyBC2']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamEnergyBC2}
detectors['ebeamL3Energy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamL3Energy}
detectors['ebeamLTU250']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamLTU250}
detectors['ebeamLTU450']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamLTU450}
detectors['ebeamLTUAngY']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamLTUAngY}
detectors['ebeamLTUPosX']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamLTUPosX}
detectors['ebeamLTUPosY']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamLTUPosY}
detectors['ebeamLUTAngX']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamLUTAngX}
detectors['ebeamPkCurrBC1']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPkCurrBC1}
detectors['ebeamPkCurrBC2']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPkCurrBC2}
detectors['ebeamUndAngX']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamUndAngX}
detectors['ebeamUndAngY']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamUndAngY}
detectors['ebeamUndPosX']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamUndPosX}
detectors['ebeamUndPosY']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamUndPosY}
detectors['ebeamXTCAVAmpl']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamXTCAVAmpl}
detectors['ebeamXTCAVPhase']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamXTCAVPhase}


# Important Epics
detectors['vitaraDelay']={'pskey':'las_fs14_target_time', 'get':lambda det: det}
###############################################################
### Specify analyses to perform and write to small data
###############################################################

from FFT_peakfinder import FFTfind_fixed
# Analysis is of form {analysisKey: {'function': analysisFunction(), 'detectorKey': 'key', 'analyzeEvery':1}}
# Function element is optional. If not provided, raw data is returned.
analysis = {}
analysis['vitaraDelay'] = {'function':lambda x: x, 'detectorKey':'vitaraDelay'}
analysis['evrs'] = {'detectorKey':'evrs'}
analysis['vls1D'] = {'function': lambda x: x, 'detectorKey':'vls'}
analysis['pulseEnergy-gmd'] = {'detectorKey':'gmd'}
analysis['pulseEnergy-xgmd'] = {'detectorKey':'xgmd'}
analysis['photonEnergy'] = {'detectorKey':'photonEnergy'}

analysis['ttfltpos']={'detectorKey':'ttfltpos'}
analysis['ttfltposfwhm']={'detectorKey':'ttfltposfwhm'}
analysis['ttampl']={'detectorKey':'ttampl'}


analysis['ebeamCharge']={'detectorKey':'ebeamCharge'}
analysis['ebeamDumpCharge']={'detectorKey':'ebeamDumpCharge'}
analysis['ebeamEnergyBC1']={'detectorKey':'ebeamEnergyBC1'}
analysis['ebeamEnergyBC2']={'detectorKey':'ebeamEnergyBC2'}
analysis['ebeamL3Energy']={'detectorKey':'ebeamL3Energy'}
analysis['ebeamLTU250']={'detectorKey':'ebeamLTU250'}
analysis['ebeamLTU450']={'detectorKey':'ebeamLTU450'}
analysis['ebeamLTUAngY']={'detectorKey':'ebeamLTUAngY'}
analysis['ebeamLTUPosX']={'detectorKey':'ebeamLTUPosX'}
analysis['ebeamLTUPosY']={'detectorKey':'ebeamLTUPosY'}
analysis['ebeamLUTAngX']={'detectorKey':'ebeamLUTAngX'}
analysis['ebeamPkCurrBC1']={'detectorKey':'ebeamPkCurrBC1'}
analysis['ebeamPkCurrBC2']={'detectorKey':'ebeamPkCurrBC2'}
analysis['ebeamUndAngX']={'detectorKey':'ebeamUndAngX'}
analysis['ebeamUndAngY']={'detectorKey':'ebeamUndAngY'}
analysis['ebeamUndPosX']={'detectorKey':'ebeamUndPosX'}
analysis['ebeamUndPosY']={'detectorKey':'ebeamUndPosY'}
analysis['ebeamXTCAVAmpl']={'detectorKey':'ebeamXTCAVAmpl'}
analysis['ebeamXTCAVPhase']={'detectorKey':'ebeamXTCAVPhase'}


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
# FFTfind_fixed(hsd, nmax=1000)
analysis['mb-hitfinder-t'] = {'function': lambda x: cfdFixed(x)[0]*1e6, 'detectorKey':'hsd'}
analysis['mb-hitfinder-ampl'] = {'function': lambda x: cfdFixed(x)[1], 'detectorKey':'hsd'}
analysis['mb-FFT-hitfinder-t'] = {'function': lambda x: FFTfind_fixed(x)[0]*1e6, 'detectorKey':'hsd'}
analysis['mb-FFT-hitfinder-ampl'] = {'function': lambda x: FFTfind_fixed(x)[1], 'detectorKey':'hsd'}

analysis['mb-time-subset'] = {'function': lambda x: x[0]['times'].astype(float)[:10000]*1e6, 'detectorKey':'hsd'}
analysis['mb-waveform-subset'] = {'function': lambda x: fix_wf_baseline(x[0][0].astype(float))[:10000], 'detectorKey':'hsd'}

resample = lambda x, rebin_factor: x.reshape(-1, rebin_factor).mean(1)
analysis['mb-time-downsample'] = {'function': lambda x: resample(x[0]['times'],10)*1e6, 'detectorKey':'hsd'}
analysis['mb-waveform-downsample'] = {'function': lambda x: resample(x[0][0],10), 'detectorKey':'hsd'}
analysis['diode-waveform-subset'] = {'function': lambda x: x[9][0].astype(float)[:5000], 'detectorKey':'hsd'}
analysis['diode-time-subset'] = {'function': lambda x: x[9]['times'].astype(float)[:5000]*1e6, 'detectorKey':'hsd'}
analysis['diode-waveform-downsample'] = {'function': lambda x: resample(x[9][0],10), 'detectorKey':'hsd'}
analysis['diode-time-downsample'] = {'function': lambda x: resample(x[9]['times'],10)*1e6, 'detectorKey':'hsd'}

analysis['atm-proj1'] =  {'function': lambda x: np.sum(x[360:500,:],axis=0), 'detectorKey':'tmo_atmopal'}
analysis['atm-proj2'] =  {'function': lambda x: np.sum(x[0:240,:],axis=0), 'detectorKey':'tmo_atmopal'}
analysis['atm-proj3'] =  {'function': lambda x: np.sum(x[240:360,:],axis=0), 'detectorKey':'tmo_atmopal'}