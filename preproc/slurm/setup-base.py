## Contributors: Matt Ware
import numpy as np

###############################################################
### Specify detectors
###############################################################

detectors = {}
# Fast detectors
# detectors['sample']={'pskey':'timing', 'get':lambda det: det}
detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image}
detectors['vls']={'pskey':'andor', 'get':lambda det: det.raw.value}
detectors['gmd']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
detectors['hsd']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms}
detectors['photonEnergy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPhotonEnergy}

# Important Epics
detectors['vitaraDelay']={'pskey':'las_fs14_target_time', 'get':lambda det: det}

###############################################################
### Specify analyses to perform and write to small data
###############################################################

# Analysis is of form {analysisKey: {'function': analysisFunction(), 'detectorKey': 'key', 'analyzeEvery':1}}
# Function element is optional. If not provided, raw data is returned.
analysis = {}
analysis['vitaraDelay'] = {'function':lambda x: x, 'detectorKey':'vitaraDelay'}
analysis['evrs'] = {'detectorKey':'evrs'}
analysis['vls1D'] = {'function': lambda x: np.sum(x,-1), 'detectorKey':'vls'}
analysis['pulseEnergy'] = {'detectorKey':'gmd'}
analysis['photonEnergy'] = {'detectorKey':'photonEnergy'}

resample = lambda x, rebin_factor: x.reshape(-1, rebin_factor).mean(1)
analysis['itof-time'] = {'function': lambda x: resample(x[0]['times'],10), 'detectorKey':'hsd'}
analysis['itof-waveform'] = {'function': lambda x: resample(x[0][0],10), 'detectorKey':'hsd'}
analysis['diode-time'] = {'function': lambda x: x[0]['times'].astype(float)[:5000], 'detectorKey':'hsd'}
analysis['diode-waveform'] = {'function': lambda x: x[9][0].astype(float)[:5000], 'detectorKey':'hsd'}

analysis['atm-proj1'] =  {'function': lambda x: np.sum(x[360:500,:],axis=0), 'detectorKey':'tmo_atmopal'}
analysis['atm-proj2'] =  {'function': lambda x: np.sum(x[0:240,:],axis=0), 'detectorKey':'tmo_atmopal'}
analysis['atm-proj3'] =  {'function': lambda x: np.sum(x[240:360,:],axis=0), 'detectorKey':'tmo_atmopal'}