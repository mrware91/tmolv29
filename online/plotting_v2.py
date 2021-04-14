# Contributors: Matt Ware
import numpy as np
import time
import setup_v2 as setup
import psmon.plots
import psmon.publish
psmon.publish.local=False

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Plotter:
    def __init__(self, numclients):
        self.nupdate=0
        self.data = None
        self.numclients = numclients
        self.t0 = time.time()
        self.t1 = time.time()
        self.lastPlotted = {}
    def update(self,data):
        placeAt = data['rank']
        if placeAt>=self.numclients:
            return
        if self.data is None:
            self.data={}
            for key,val in data.items():
                arr = np.copy(val)
                dims = np.shape(arr)
                self.data[key] = np.zeros((self.numclients,*dims))*np.nan
                self.data[key][placeAt,] = arr
        else:
            self.nupdate+=1
            # placeAt = self.nupdate % self.numclients
            for key,val in data.items():
                arr = np.copy(val)
                self.data[key][placeAt,] = arr
    def plot(self):
        if time.time()-self.t1 < 0.1:
            return
        self.t1 = time.time()
        
        for key, val in setup.plots.items():
            try:
                plotEverySec = val['plotEveryNthSec']
            except KeyError as ke:
                plotEverySec = 0
                
            try:
                t2 = self.lastPlotted[key]
            except KeyError as ke:
                t2 = time.time()
                self.lastPlotted[key] = t2
                
            if time.time()-t2 < plotEverySec:
                continue
            self.lastPlotted[key] = time.time()
            try:
                plotElement( self.nupdate, key, val, self.data )
            except KeyError as ke:
                print(f'Error occured for {key}, skipping')
                # raise e
        
        
ncalls=0
masterPlotter = None
def callback(data):
    global masterPlotter, ncalls
    if ncalls == 0:
        psmon.publish.init()
        masterPlotter = Plotter(size)
    
    # print(data['iread'],data['rank'],rank)
    masterPlotter.update(data)
    masterPlotter.plot()
    ncalls+=1

def plotImage(nevt, plotName, plotTitle, im,aspect_ratio=1,**kwargs):
    # print('?')
    aplot = psmon.plots.Image(nevt, plotTitle, im, aspect_ratio=aspect_ratio)
    psmon.publish.send(plotName, aplot)
    return None

def plotXY(nevt, plotName, x,y, plotTitle='', xlabel='x',ylabel='y',formats='-',**kwargs):
    # print('?')
    x=np.array(x).astype(float)
    y=np.array(y).astype(float)
    idx = (~np.isnan(x))&(~np.isnan(y))
    aplot = psmon.plots.XYPlot(nevt, plotTitle,
                               x[idx], y[idx],
                               xlabel=xlabel,
                               ylabel=ylabel,
                               formats=formats)
    psmon.publish.send(plotName, aplot)
    
def plotHist(nevt, plotName, x,y, plotTitle='',xlabel='x',ylabel='y',formats='-',fills=True,**kwargs):
    aplot = psmon.plots.Hist(nevt, plotTitle,
                               x, y,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               formats=formats,fills=fills)
    psmon.publish.send(plotName, aplot)

def plotElement( nevent, name, plotDictionary, dataSource ):
    plotTitle = name
        
    
    if plotDictionary['type'] == 'XY':
        # print(dataSource.keys())
        x = plotDictionary['xfunc'](dataSource)
        y = plotDictionary['yfunc'](dataSource)
        plotXY( nevent, name, x, y, plotTitle=name, **plotDictionary )
    elif plotDictionary['type'] == 'Image':
        im = plotDictionary['imageFunc'](dataSource)
        plotImage( nevent, name, name, im, **plotDictionary )
        
    elif plotDictionary['type'] == 'Histogram':
        x = plotDictionary['xfunc'](dataSource)
        y = plotDictionary['yfunc'](dataSource)
        plotHist( nevent, name, x,y, plotTitle = name, **plotDictionary )