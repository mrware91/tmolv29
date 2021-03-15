import numpy as np


class analyses:
    def __init__(self, analysis, totalEvents,printMode='verbose'):
        self.analysis = analysis
        self.totalEvents = totalEvents
        self.events = 0
        self.printMode = printMode
        self.data = {}
        self.dataTypesFound = False
        self.outTypes = {}
        self.initialize()
        
    def initialize(self):
        self.events = 0
        self.data = {}
        for key in self.analysis:
            self.outTypes[key] = None
            self.analysis[key]['type'] = None
            self.analysis[key]['size'] = None
            self.data[key] = np.zeros(self.totalEvents)*np.nan
            
            self.setdefault(self.analysis[key],
                       'function',
                       '%s: No analysis function provided. Defaulting to return raw data.'%key,
                        lambda x: x)
            
            self.setdefault(self.analysis[key],
                       'analyzeEvery',
                       '%s: No modulo provided. Will analyze every shot.'%key,
                        1)
                
            
    def update(self, detectors):
        self.dataTypesFound = True
        for key in self.analysis:
            analyzeEvery = self.analysis[key]['analyzeEvery']
            if not ( self.events%analyzeEvery == 0):
                continue
                
            function = self.analysis[key]['function']
            detectorKey = self.analysis[key]['detectorKey']
            shotData = detectors[detectorKey]['shotData']
            if shotData is None:
                self.dataTypesFound = False
                continue
                
            result = function(shotData)
            if result is not None:
                if self.analysis[key]['type'] is None:
                    self.analysis[key]['type'] = type(result)
                    self.analysis[key]['size'] = np.size(result)
                    dims = np.shape(result)
                    self.data[key] = np.zeros((self.totalEvents,*dims))*np.nan
                self.data[key][self.events,] = result
                if self.outTypes[key] is None:
                    self.outTypes[key] = {}
                    self.outTypes[key]['type'] = type(self.data[key][self.events,])
                    self.outTypes[key]['size'] = np.size( self.data[key][self.events,] )
            elif (result is None) & (self.analysis[key]['type'] is None):
                self.dataTypesFound = False
                    
        self.events += 1
        if self.events >= self.totalEvents:
            self.cprint('Read events exceeds total expected. Resetting event count.')
            self.events = 0
            
    def setdefault(self, adict, key, response, default):
        try:
            adict[key]
        except KeyError as ke:
            allowedErrorStr = '\'%s\'' % key
            if allowedErrorStr == str(ke):
                self.cprint(response)
                adict[key] = default
            else:
                raise ke
                
    def cprint(self, aString):
        if self.printMode in 'verbose':
            print(aString)
        elif self.printMode in 'quiet':
            pass
        else:
            print('printMode is %s. Should be verbose or quiet. Defaulting to verbose.'%self.printMode)
            self.printMode = 'verbose'
            self.cprint(aString)
            
    def H5out(self):
        if self.dataTypesFound:
            outDict = {}
            for key in self.data:
                outDict[key] = self.data[key][0]
            return outDict
        else:
            return None