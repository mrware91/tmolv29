rememberShots = 1200


########################################################################
# Import libraries and setup
########################################################################
import numpy as np
from collections import deque
import psmon.plots
import psmon.publish
# psmon.publish.local=False

# imports for electron hit finding
from sys import path
hitfinder_folder = '/reg/neh/home/tdd14/modules/hitfinder_py3'
path.append(hitfinder_folder)
from hitFinder_peakvals import findHits, make_kernel

# import ktof extraction functions
path.append('/cds/home/j/jneal/Jordan_LCLS_scripts/tmolv4118/shmem_functions/')
from ktof_extract import extract_en_intens

# imports for VMI Abel transformation
path.append('/reg/neh/home/tdd14/modules/CPBASEX/pbasex-Python/pbasex')
from gData import loadG
from pbasex import pbasex
path.append('/reg/neh/home/tdd14/modules/quadrant/quadrant')
from quadrant import foldQuadrant, resizeFolded
gData = loadG('/cds/home/m/mrware/G_r512_k128_l4.h5', 0) #1 flags load array to make images
alpha = 5.39e-4 #for Xiang 60 eV setting
#alpha = 2.69e-4  # for Xiang 30 eV setting
#alpha = 3.59e-4 # for Xiang 40 eV setting

from scipy.ndimage.filters import gaussian_filter

########################################################################
# set parameters
########################################################################

### 2d detectors
# define 2d hit finding parameters
thresh = 38.2
mkr = 1
gkr = 5
sigma = 1.0
#initialize gaussian kernel for hit finding
gauss_kernel = make_kernel(gkr, sigma)

x0,y0=465,469 #center of VMI image for electrons
thresh_imsum=40 #threshold for VMI image sum

elecroi1_r,elecroi1_w=230,20 #radius, width in pix for roi1
elecroi2_r,elecroi2_w=190,20 #radius, width in pix for roi2

### 1d detectors
shift, threshold, deadtime = 2e-3, 3, 4e-3 # 1d cfd parameters
c, t0 =  1.12789, 0.19988 # ion ToF calibration

########################################################################
# analysis functions
########################################################################
# function to fix waveform baseline
def fix_wf_baseline(hsd_in, bgfrom=500*64):
    hsd_out = np.copy(hsd_in)
    for i in range(4):
        hsd_out[i::4] -= hsd_out[bgfrom+i::4].mean()
    for i in (12, 13, 12+32, 12+32):
        hsd_out[i::64] -= hsd_out[bgfrom+i::64].mean()
    return hsd_out


# 1d hitfinding
def cfd(x, y, shift=2e-3, threshold=7, deadtime=2e-3):
    # Simple Constant Fraction Discriminator for hitfinding

    pixel_shift = int(shift / np.diff(x).mean() / 2)
    y1, y2 = y[:-2*pixel_shift], y[2*pixel_shift:]
    x_, y_ = x[pixel_shift:-pixel_shift], y[pixel_shift:-pixel_shift]
    y3 = y1 - y2
    peak_idx = np.where((y3[:-1]<0)&(0<=y3[1:])&(y_[1:]>threshold))[0]
    times, amplitudes = x_[1:][peak_idx], y_[1:][peak_idx]
    if len(times)==0:
        return [], []
    else:
        deadtime_filter = [0]
        previous_time = times[0]
        for i, time in enumerate(times[1:]):
            if time - previous_time > deadtime:
                deadtime_filter.append(i+1)
                previous_time = time
        return times[deadtime_filter], amplitudes[deadtime_filter]

def bin2d(ra, bin_sizes=(4,4)):
    'Bin down in 2 dimensions'
    quot0,rem0=divmod(ra.shape[0], bin_sizes[0])
    quot1,rem1=divmod(ra.shape[1], bin_sizes[1])
    ra=ra[:bin_sizes[0]*quot0,:bin_sizes[1]*quot1] #'trim'so that reshaping works
    
    binoneway=ra.reshape(ra.shape[0], quot0, bin_sizes[0]).sum(axis=2)
    binotherway=binoneway.T.reshape(binoneway.shape[1], quot1, bin_sizes[1]).sum(axis=2)
    
    return binotherway.T

# for making shared memory plots more square
def binim(x):
    return np.average(np.reshape(x.astype(float)[:,:divmod(x.shape[1],3)[0]*3], (1024,-1,3)), axis=2)

# interleave two arrays
def interleave(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


########################################################################
# useful psmon wrappers
########################################################################

def plotImage(nevt, plotName, plotTitle, im,aspect_ratio=1):
    aplot = psmon.plots.Image(nevt, plotTitle, im, aspect_ratio=aspect_ratio)
    psmon.publish.send(plotName, aplot)
    return None

def plotXY(nevt, plotName, x,y, plotTitle='', xlabel='x',ylabel='y',formats='-'):
    x=np.array(x).astype(float)
    y=np.array(y).astype(float)
    idx = (~np.isnan(x))&(~np.isnan(y))
    aplot = psmon.plots.XYPlot(nevt, plotTitle,
                               x[idx], y[idx],
                               xlabel=xlabel,
                               ylabel=ylabel,
                               formats=formats)
    psmon.publish.send(plotName, aplot)
    
def plotHist(nevt, plotName, x,y, plotTitle='',xlabel='x',ylabel='y',formats='-',fills=True):
    aplot = psmon.plots.Hist(nevt, plotTitle,
                               x, y,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               formats=formats,fills=fills)
    psmon.publish.send(plotName, aplot)
    
########################################################################
# laser diode vs. labtime
########################################################################
cutoff=200
def updatetranslodiode(data,detectors,cutoff=cutoff,maxShots=1200):
    if data is None:
        data={}
        data['off']=0
        data['on']=deque(maxlen=maxShots)
        data['delay']=deque(maxlen=maxShots)
        data['xeng']=deque(maxlen=maxShots)
        
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    
    wf=fix_wf_baseline(detectors['wfs']['shotData'][9][0].astype(float))
    wfsum=np.sum(wf[1250:1400]) 
    if (~xrayOff)&(wfsum>cutoff):
        data['on'].append( wfsum / data['off'] )
        data['delay'].append(detectors['vitaraDelay']['shotData'])
        data['xeng'].append(detectors['pulseEnergy']['shotData'])
    elif (wfsum>cutoff):
        data['off'] = wfsum
#         if data['off'] == 0:
#             data['off'] = wfsum
#         else:
#             data['off'] = data['off']*.75 - wfsum*.25
            
    return data
    
def plottranslodiode(data,nevt):
    t = np.arange(len(data['on']))
    y = np.array(data['on']).astype(float)
    xeng=np.array(data['xeng']).astype(float)
    delay=np.array(data['delay']).astype(float)
    
    plotXY(nevt, plotName='translodiode', 
               x=t,y=y, 
               plotTitle='transmission on diode vs lab time', xlabel='shot #',ylabel='diode integrated / bkgd',formats='-')

    plotXY(nevt, plotName='takahiro',x=xeng,y=y,plotTitle='diode/bkgd vs. xray pulse energy',xlabel='xray pulse energy',ylabel='diode/bkgd',formats='.')
    plotXY(nevt, plotName='diodeVdelay',x=delay,y=y,plotTitle='diode/bkgd vs. delay',xlabel='delay',ylabel='diode/bkgd',formats='.')

    # histogram
    histy, histx = np.histogram(delay[~np.isnan(delay) & ~np.isnan(y)], 50, weights=y[~np.isnan(delay) & ~np.isnan(y)])
    histn = np.zeros_like(histy)
    histidx = np.digitize(delay[~np.isnan(delay) & ~np.isnan(y)], histx)
    for i in range(histy.size):
        histn[i] = np.sum(histidx == i)
    histy = histy / histn
    
    plotXY(nevt, plotName='diodeVdelayhist',x=histx[:-1],y=histy,plotTitle='diode/bkgd vs. delay hist',xlabel='delay',ylabel='diode/bkgd')
    
    return data


########################################################################
# timetool plots
########################################################################
def updatett(data,detectors,tt_veto=-4.5e7):
    im = np.copy(detectors['tmo_atmopal']['shotData']).T
    if im is None:
        return data
    
    if data is None:
        data = {'imsum':None,'imsumbg':None,'shots':0,'shotsbg':0,
               'imlast':None}
        data['delay'] = None
        
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    
    if (~xrayOff)& ( np.nanmedian(im[:]) > -40 ):
        if data['shots'] == 0:
            data['imsum'] =im
            data['imlast'] =im
        else:
            data['imsum'] += im
            data['imlast'] =im
        data['shots']+=1.0

        if detectors['vitaraDelay']['shotData'] is None:
            data['delay'] = -1.
        else:
            data['delay'] = detectors['vitaraDelay']['shotData']

    if (xrayOff) & ((im[:].sum()>tt_veto)):
        if data['shotsbg'] == 0:
            data['imsumbg'] =im
        else:
            data['imsumbg'] = 0.9*data['imsumbg'] + 0.1*im
        data['shotsbg']+=1.0
    return data
    
def plottt(data,nevt):
    stable_time_point = 5438.224
    delay_diff = 1 # ps

    if data is None:
        return data
    if data['imsumbg'] is not None:
        plotImage(nevt, plotName='tt_dark', plotTitle='timetool dark', im=data['imsumbg'],aspect_ratio=1)
    if (data['imlast'] is not None) &  (data['imsumbg'] is not None):
        implt = data['imlast']/data['imsumbg']
        projplt = implt[:,200:500].mean(1)
        projplt = projplt / np.nanmean(projplt)
        plotXY(nevt, plotName='tt_proj', 
               x=np.arange(projplt.size),y=projplt, 
               plotTitle='timetool bkdivproj', xlabel='pixel',ylabel='tt / bkgd',formats='-')
        
        implt = implt / np.nanmean(implt)
        plotImage(nevt, plotName='tt_bkdiv', plotTitle='timetool latest / timetool dark', im=implt,aspect_ratio=1)
    if (data['imlast'] is not None):
        plotImage(nevt, plotName='tt', plotTitle='timetool latest', im=data['imlast'],aspect_ratio=1)

        #if 1e3*(np.abs(stable_time_point - data['delay']) < delay_diff):
            #plotXY(nevt, plotName='timetool_stable',x=np.arange(projplt.size),y=projplt, plotTitle='timetool bkdiv projection at one delay', xlabel='pixel',ylabel='tt / bkgd',formats='-')
            



    return data 

########################################################################
# wflodiode - Single shot ion tof
########################################################################
def updatewflodiode(data,detectors):
    if data is None:
        data=fix_wf_baseline(detectors['wfs']['shotData'][9][0].astype(float))
    return data
    
def plotwflodiode(data,nevt):
    plotName = 'wflodiode'
    t = np.arange(data.size)
    y = data
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Single-shot laser diode waveform',
                               t[idx], y[idx],
                               xlabel='adu',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return None

########################################################################
# vmihits - accumulated VMI hits over all shots, gas-off background subtracted
########################################################################
def updatevmihits(data,detectors):
    im = np.copy(detectors['tmo_opal1']['shotData'])
    
    if im is None:
        return data
    
    yc, xc, pv = findHits(im.flatten(), gauss_kernel, thresh, gkr, mkr) #hit find on electron image
    
    if data is None:
        data = {'xhits':[],'yhits':[],
                'xhitsbg':[],'yhitsbg':[],
                'shots':0,'shotsbg':0,
               'vmihits':None,'vmihitsbg':None}
        
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    gasOn=evr[70]
    
    if True: #(~xrayOff)&(gasOn):
        data['xhits'] = np.copy(xc)
        data['yhits'] = np.copy(yc)
        data['shots']+=1
    if (~xrayOff)&(~gasOn):
        data['xhitsbg'] = np.copy(xc)
        data['yhitsbg'] = np.copy(yc)
        data['shotsbg']+=1
    return data


def plotvmihits(data,nevt):
    plotName = 'vmihits'
    vmihits_temp = np.rot90(np.histogram2d(np.array(data['yhits']), np.array(data['xhits']), bins=(np.arange(1024),np.arange(1024)))[0])
    vmihitsbg_temp = np.rot90(np.histogram2d(np.array(data['yhitsbg']), np.array(data['xhitsbg']), bins=(np.arange(1024),np.arange(1024)))[0])
    if data['vmihits'] is None:
        data['vmihits']=np.copy(vmihits_temp)
        data['vmihitsbg']=np.copy(vmihitsbg_temp)
    else:
        data['vmihits']+=vmihits_temp
        data['vmihitsbg']+=vmihitsbg_temp
    if data['shotsbg'] <= 1: data['shotsbg'] = 1.
    print(data['shots'])
    vmiim2plot=data['vmihits'].astype(float)/float(data['shots']) #- data['vmihitsbg'].astype(float)/float(data['shotsbg'])
    aplot = psmon.plots.Image(nevt, 'VMI hits', vmiim2plot,aspect_ratio=1)
    psmon.publish.send(plotName, aplot)
    return data
            
########################################################################
# pbasex
########################################################################
def updatepbasex(data,detectors):
    im = np.copy(detectors['tmo_opal1']['shotData'])
    if im is None:
        return data
    im[im<thresh_imsum]=0
    if data is None:
        data = {'imsum':None,'imsumbg':None,'shots':0,'shotsbg':0}
        
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    gasOn=evr[70]
    
    if (~xrayOff)&(gasOn):
        if data['shots'] == 0:
            data['imsum'] =im
        else:
            data['imsum'] += im
        data['shots']+=1
    if (~xrayOff)&(~gasOn):
        if data['shotsbg'] == 0:
            data['imsumbg'] =im
        else:
            data['imsumbg'] += im
        data['shotsbg']+=1
    return data
    
def plotpbasex(data,nevt):
    plotName = 'pbasex'     
    im2inv=data['imsum']/float(data['shots'])-data['imsumbg']/float(data['shotsbg']) #vmihits
    bs=2 #must be integer
    folded = resizeFolded(foldQuadrant(bin2d(im2inv, bin_sizes=(2,2)), int(x0/float(bs)), int(y0/float(bs)), [1,1,1,1]), 512)
    out = pbasex(folded, gData, make_images=False, alpha=4.1e-4)
            
    aplot = psmon.plots.XYPlot(nevt, 'pbasex',
                               alpha*np.arange(512)[::bs]**2,
                               out['IE'][:int(512/bs)])
    psmon.publish.send(plotName, aplot)
    return data    
   

########################################################################
# numehits - number of electron hits
########################################################################
def updatenumehits(data,detectors):
    im = detectors['tmo_opal1']['shotData']
    if im is not None:
        yc, xc, pv = findHits(im.flatten(), gauss_kernel, thresh, gkr, mkr) #hit find on electron image
    if data is None:
        data = deque(maxlen=rememberShots)
    data.append( len(xc) )
    return data

    
def plotnumehits(data,nevt):
    plotName = 'numehits'
    t = np.arange(len(data))
    y = np.array(data).astype(float)
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'number of electron hits',
                               t[idx], y[idx],
                               xlabel='shot #',
                               ylabel='# e',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return data


########################################################################
# lastvmi - last electron vmi image
########################################################################
def updatelastvmi(data,detectors):
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    gasOn=evr[70]
#     print(detectors['tmo_opal1']['shotData'].shape)
    #if (detectors['tmo_opal1']['shotData'] is not None)&(~xrayOff)&(gasOn):
    data=detectors['tmo_opal1']['shotData']
    return data

    
def plotlastvmi(data,nevt):
    plotName = 'c-VMI'
    aplot = psmon.plots.Image(nevt, 'Last electron VMI image', data,aspect_ratio=1)
    psmon.publish.send(plotName, aplot)
    return data

########################################################################
# gdhist - pulse energy histogram
########################################################################
def updategdhist(data,detectors):
    if data is None:
        data=deque(maxlen=rememberShots)
    data.append(detectors['pulseEnergy']['shotData'] )
    return data

    
def plotgdhist(data,nevt):
    plotName = 'gdhist'
    datanp = np.array(data).astype(float)
    bins, edges = np.histogram(datanp[~np.isnan(datanp)],bins=20, range=(0,np.nanmax(datanp)))
    
    aplot = psmon.plots.Hist(nevt, 'Pulse energy histogram',
                               edges, bins,
                               xlabel='pulse energy',
                               ylabel='counts',
                               formats='-',fills=True)
    psmon.publish.send(plotName, aplot)
    return data


########################################################################
# timetool_roisum - summed timetool image
########################################################################
def updatetimetool_roisum(data,detectors):
    if data is None:
        data={}
        data=deque(maxlen=rememberShots)
    data.append(np.nansum(detectors['tmo_atmopal']['shotData'][:]) )
    return data

    
def plottimetool_roisum(data,nevt):
    plotName = 'timetool_roisum'
    t = np.arange(len(data))
    y = np.array(data).astype(float)
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'timetool_roisum',
                               t[idx], y[idx],
                               xlabel='shot #',
                               ylabel='roi sum',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return data


########################################################################
# ktofrollave - rolling Averaged ion tof
########################################################################
def updatektofrollave(data,detectors):
    if data is None:
        if detectors['wfs']['shotData'] is not None:
            data={}
            data['itof']=fix_wf_baseline(-detectors['wfs']['shotData'][6][0].astype(float))
            data['count']=1.
    else:
        if data['count'] < 20:
            if detectors['wfs']['shotData'] is not None:
                data['itof']+=fix_wf_baseline(-detectors['wfs']['shotData'][6][0].astype(float))
                data['count']+=1.
    return data
    
def plotktofrollave(data,nevt):
    plotName = 'ktofrollave'
    t = np.arange(data['itof'].size)
    y = data['itof']/data['count']
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Rolling average ktof',
                               t[idx], y[idx],
                               xlabel='adu',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return None

########################################################################
# ktofave - Averaged ion tof
########################################################################
def updatektofave(data,detectors):
    if data is None:
        if detectors['wfs']['shotData'] is not None:
            data={}
            data['itof']=fix_wf_baseline(-detectors['wfs']['shotData'][6][0].astype(float))
            data['count']=1.
    else:
        if detectors['wfs']['shotData'] is not None:
            data['itof']+=fix_wf_baseline(-detectors['wfs']['shotData'][6][0].astype(float))
            data['count']+=1.
    return data
    
def plotktofave(data,nevt):
    plotName = 'ktofave'
    t = np.arange(data['itof'].size)
    y = data['itof']/data['count']
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Average ktof',
                               t[idx], y[idx],
                               xlabel='adu',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return data

########################################################################
# ktof - Single shot ion tof
########################################################################
def updatektof(data,detectors):
    if data is None:
        data=fix_wf_baseline(-detectors['wfs']['shotData'][6][0].astype(float))
    return data
    
def plotktof(data,nevt):
    plotName = 'ktof'
    t = np.arange(data.size)
    y = data
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Single-shot ktof',
                               t[idx], y[idx],
                               xlabel='adu',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return None


########################################################################
# wf0rollave - rolling Averaged ion tof
########################################################################
def updatewf0rollave(data,detectors):
    if data is None:
        data = {}
        if detectors['wfs']['shotData'] is not None:
            data['itof'] = deque(maxlen=10)  # update maxlen to change number of shots in the rolling average
            data['times'] = deque(maxlen=10)
            data['itof'].append( fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float)) )
            data['times'].append( np.array(detectors['wfs']['shotData'][0]['times']) )
    else:
        if detectors['wfs']['shotData'] is not None:
            data['itof'].append( fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float)) )
            data['times'].append( np.array(detectors['wfs']['shotData'][0]['times']) )
    return data
    
def plotwf0rollave(data,nevt):
    plotName = 'wf0rollave'
    t = np.mean(data['times'], 0)*10**6  # microseconds
    t0 = 0.196  # in us
    A = (5.119 - t0)/np.sqrt(14)
    MQ = (t - t0)**2/A**2  # in m/q

    y = np.mean(data['itof'], 0)
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Rolling average ion ToF from dVMI',
                               MQ[idx], y[idx],
                               xlabel='m/q',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return data

########################################################################
# wf0ave - Averaged ion tof
########################################################################
def updatewf0ave(data,detectors):
    if data is None:
        if detectors['wfs']['shotData'] is not None:
            data={}
            data['itof']=fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float))
            data['count']=1.
    else:
        if detectors['wfs']['shotData'] is not None:
            data['itof']+=fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float))
            data['count']+=1.
    return data
    
def plotwf0ave(data,nevt):
    plotName = 'wf0ave'
    t = np.arange(data['itof'].size)
    y = data['itof']/data['count']
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Average ion tof from dVMI',
                               t[idx], y[idx],
                               xlabel='adu',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)
    return data

########################################################################
# wf0 - Single shot ion tof
########################################################################
def updatewf0(data,detectors,nshots=1000,wf0roi_range=(20000,40000)):
    if data is None:
        data = {}
        data['wf0'] = None
        data['wf0roi'] = deque(maxlen=nshots)
        data['wf0off'] = None
        data['times'] = np.array(detectors['wfs']['shotData'][0]['times'])
    data['wf0']=fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float))
    data['wf0roi'].append(np.sum(data['wf0'][wf0roi_range[0]:wf0roi_range[1]]))
    return data
    
def plotwf0(data,nevt):
    plotName = 'wf0'
    t = data['times']*10**6  # convert from seconds to microseconds
    y = data['wf0']
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Single-shot ion ToF from dVMI',
                               t[idx], y[idx],
                               xlabel='microseconds',
                               ylabel='waveform',
                               formats='-')
    psmon.publish.send(plotName, aplot)

    t0 = 0.190 # in us
    A = (13.211 - t0)/np.sqrt(100)
    MQ = (t - t0)**2/A**2  # in m/q
    plotXY(nevt, plotName='wf0mq', x=MQ[idx], y=y[idx], xlabel='m/q', plotTitle='Single-shot ion ToF from dVMI', ylabel='waveform',formats='-')

    t = np.arange(len(data['wf0roi']))
    y = data['wf0roi']
    plotXY(nevt, plotName='wf0roi',x=t,y=y,plotTitle='sum of ROI on ion ToF',xlabel='shot #',ylabel='ROI sum')

    return data


#  if evr[67]: #not goose
#                 ionsrollave.append(wf0) #add to the deque
#                 evr[161]
#             if evr[68]: #goose
#                ionsgooserollave.append(wf0)
#             ktofrollave.append(wf6)

# ########################################################################
# # wfgoose 
# ########################################################################
def updatewfgoose(data,detectors,nshots=32000, ROI=(42.8,43.2)): # old roi: 5.5,6
    pe=detectors['pulseEnergy']['shotData']
    if pe is None:
        return data
    
    wf0 = fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float))
    if wf0 is None:
        return data
    
    if (pe < 0.1)|(pe>0.3):
        return data
    
    wf0=wf0 /pe
    
    if data is None:
        data={}
        data['t'] = np.array(detectors['wfs']['shotData'][0]['times'])*1.0e6
        t0=0.190
        A=(13.211 - t0)/np.sqrt(100.)
        data['mq'] = ( data['t']-t0 )**2/A**2
        data['notgoose']=deque(maxlen=nshots)
        data['notxray']=deque(maxlen=nshots)
        data['delays']=deque(maxlen=nshots)
        data['nxdelays']=deque(maxlen=nshots)
        data['lastgoose'] = None
        data['waveformnotgoose'] = None
        data['count'] = 1
        
        
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    gasOn=evr[70]
    notgoose=evr[67]
    goose=evr[68]
    
    if detectors['vitaraDelay']['shotData'] is None:
        return data
        
    if (~xrayOff)&(gasOn):
        if notgoose & (data['lastgoose'] is not None):
            wfdiff =( wf0 - data['lastgoose'])
            idxs = (data['mq']>ROI[0])&( data['mq']<ROI[1])
#             print(ROI)
#             print(data['mq'])
#             print(data['mq'][idxs])
            roi = np.nansum( (wfdiff[ (data['mq']>ROI[0])&( data['mq']<ROI[1]) ]) )
            data['notgoose'].append( roi )
            data['delays'].append( detectors['vitaraDelay']['shotData'] )

            # subtracting goose - notgoose
            if data['waveformnotgoose'] is None:
                data['waveformnotgoose'] = wfdiff
                data['count'] = 1
            else:
                data['waveformnotgoose'] += wfdiff
                data['count'] += 1
                
        elif goose:
            data['lastgoose'] = wf0
            
    if xrayOff & (data['lastgoose'] is not None) & gasOn:
        wfdiff =( wf0 - data['lastgoose'])
        roi = np.nansum( (wfdiff[ (data['mq']>ROI[0])&( data['mq']<ROI[1]) ]) )
        data['notxray'].append( roi )
        data['nxdelays'].append( detectors['vitaraDelay']['shotData'] )
        
    return data
    
def plotwfgoose(data,nevt, plotName='MASS6'):
    if data is None:
        return data
    if len(data['delays']) <= 1 :
        return data
    minDelay = np.nanmin( np.array(data['delays']).astype(float))
    maxDelay = np.nanmax( np.array(data['delays']).astype(float))
    ndelay = 40
    
    weights,edges = np.histogram( data['delays'], bins=ndelay, range=(minDelay,maxDelay), weights=data['notgoose'] )
    counts,edges = np.histogram( data['delays'], bins=ndelay, range=(minDelay,maxDelay) )
    
    nxweights,nxedges = np.histogram( data['nxdelays'], bins=ndelay, range=(minDelay,maxDelay), weights=data['notxray'] )
    nxcounts,nxedges = np.histogram( data['nxdelays'], bins=ndelay, range=(minDelay,maxDelay) )
    
    means = weights/counts
    means = means[counts>0]
    centers = edges[1:][counts>0]
    
    nxmeans = nxweights/nxcounts
    nxmeans = nxmeans[nxcounts>0]
    nxcenters = nxedges[1:][nxcounts>0]
    
    plotXY(nevt, plotName=plotName+'GOOSE',x=data['delays'],y=data['notgoose'],plotTitle='delta sum of ROI on ion ToF vs delay',xlabel='delay',ylabel='delta ROI sum / pe',formats='.')
    plotXY(nevt, plotName=plotName+'BYKIK',x=data['nxdelays'],y=data['notxray'],plotTitle='delta sum of ROI on ion ToF vs delay',xlabel='delay',ylabel='delta ROI sum / pe',formats='.')
    
    plotXY(nevt, plotName=plotName+'MGOOSE',x=centers,y=means,plotTitle='delta sum of ROI on ion ToF vs delay',xlabel='delay',ylabel='delta ROI sum / pe',formats='.')
    plotXY(nevt, plotName=plotName+'MBYKIK',x=nxcenters,y=nxmeans,plotTitle='delta sum of ROI on ion ToF vs delay',xlabel='delay',ylabel='delta ROI sum / pe',formats='.')

    # averaged
    if data['count'] > 1:
        plotXY(nevt, plotName='avgoosediff',x=data['mq'],y=gaussian_filter(data['waveformnotgoose'] / data['count'],10),plotTitle='averaged delta on ion ToF',xlabel='m/q')

    return data
########################################################################
# Pulse energy vs. labtime
########################################################################
def updatePEvLT(data,detectors):
    if data is None:
        data={}
        data['PE']=deque(maxlen=rememberShots)
    data['PE'].append(detectors['pulseEnergy']['shotData'])
    return data
    
def plotPEvLT(data,nevt):
    plotName = 'PEvLT'
    t = np.arange(len(data['PE']))
    y = np.array(data['PE']).astype(float)
    idx = ~np.isnan(y)
    aplot = psmon.plots.XYPlot(nevt, 'Pulse energy v. shot',
                               t[idx], y[idx],
                               xlabel='shot number',
                               ylabel='Pulse energy (mJ)',
                               formats='.')
    psmon.publish.send(plotName, aplot)
    return data

########################################################################
# evr histogram
########################################################################
def updateEvrHistogram(data,detectors,dep=0.99):
    if data is None:
        data={}
        data['shots']=1.0
        data['evrArray']=np.array(detectors['evrs']['shotData'])
    else:
        data['shots'] = data['shots']*dep + 1.0
        data['evrArray']=data['evrArray']*dep + np.array(detectors['evrs']['shotData'])
    return data
    
def plotEvrHistogram(data,nevt):
    plotName = 'evrhist'
    bins = np.arange(data['evrArray'].size+1)- 0.5
    aplot = psmon.plots.Hist(nevt, 'Evr Histogram',
                               bins, data['evrArray']/float(data['shots']),
                               xlabel='event code',
                               ylabel='counts',
                               formats='-',fills=True)
    psmon.publish.send(plotName, aplot)
    return data
    
########################################################################
# laser delay vs. labtime
########################################################################
def updatelaserdelay(data,detectors):
    if data is None:
        data=deque(maxlen=rememberShots)
    
    data.append(detectors['vitaraDelay']['shotData'])
    return data

def plotlaserdelay(data,nevt):
    t = np.arange(len(data))
    y = np.array(data).astype(float)

    plotXY(nevt, plotName='laserdelay',x=t,y=y,plotTitle='laser delay vs lab time',formats='-')

    return data
     
########################################################################
# VLS live
########################################################################
def updateVLS(data,detectors):
    if detectors['vls']['shotData'] is not None:
        data=np.sum(detectors['vls']['shotData'],0)
    return data

def plotVLS(data,nevt):
    if data is None:
        return data
    t = np.arange(data.size)
    L3 = 0.436404099*t + 5.15936638e3  # convert VLS pixel to accelerator gamma (L3) in MeV
    photEn = L3/10.713 # convert L3 to photon energy in eV
    y = np.array(data).astype(float)

    plotXY(nevt, plotName='VLS',x=t,y=y,plotTitle='live VLS',xlabel='pixel #',formats='-')
    plotXY(nevt, plotName='VLS_L3', x=L3, y=y, plotTitle='live VLS_L3',xlabel='L3 [MeV]', formats='-')
    plotXY(nevt, plotName='VLS_photEn', x=photEn, y=y, plotTitle='live VLS_photEn',xlabel='photon energy [eV]', formats='-')

    return data

########################################################################
# VLS Gaussian Fit
########################################################################
def gauss(x, mu, sigma, offset, ampl):
    return np.exp(-(x-mu)**2/(2*sigma)**2)*ampl + offset

def updateVLSgauss(data,detectors):
    if data is None:
        data=deque(maxlen=rememberShots)
    
    if detectors['vls']['shotData'] is not None:
        arr = np.sum(detectors['vls']['shotData'],0)
    else: 
        return data
    
    try:
        popt, pcov = curve_fit(gauss, np.arange(len(arr)), arr, p0=[arr.argmax(), 3, arr.min(), arr.max()-arr.min()])
        data.append(popt[1]*2.355)
    except:
        pass
    return data
    
def plotVLSgauss(data,nevt):    
    t = np.arange(len(data))
    y = np.array(data).astype(float)

    plotXY(nevt, plotName='vlsGauss',x=t,y=y,plotTitle='VLS FWHM vs lab time',formats='-')

    return data
    
########################################################################
# VLS CoM vs. normalized iTOF roi
########################################################################
def updateXAS(data,detectors, ROI=(42.2,43.5), phEngs=(490,600), rememberShots=50000):
    
    t = detectors['wfs']['shotData'][0]['times']*10**6  # microseconds
    t0=0.190
    A=(13.211 - t0)/np.sqrt(100.)
    MQ = (t - t0)**2/A**2  # in m/q
    itofROI = (MQ>ROI[0]) & (MQ<ROI[1])

    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    goose=bool(evr[68])
    gasoff=bool(evr[71])
    
    vls = np.array(np.sum(detectors['vls']['shotData'],0) ).astype(float)
    vls = vls - np.nanmedian(vls)
    vlspix = np.arange(vls.size)
    itof = fix_wf_baseline(detectors['wfs']['shotData'][0][0].astype(float))
    pe = detectors['pulseEnergy']['shotData']
    phe = detectors['photonEnergy']['shotData']
    
#     pe = 1man 
    
    testForNone = [xrayOff,vls,itof,pe,phe]
    for idx,arr in enumerate(testForNone):
        if arr is None:
            print(idx)
            return data
        
    if xrayOff:
        return data
    
    idxs=(vls>100)
    vlsCoM = np.nanmean(vls[idxs]*vlspix[idxs])/np.nanmean(vls[idxs])
    nITOF = np.sum(itof[itofROI] ) / np.nansum(vls)
    
    if np.isnan(vlsCoM) | np.isnan(nITOF):
        return data
    
    if data is None:
        data={}
        data['vlsCoM']=deque(maxlen=rememberShots)
        data['vlsSum']=deque(maxlen=rememberShots)
        data['pe']=deque(maxlen=rememberShots)
        data['nITOF']=deque(maxlen=rememberShots)
        data['phe']=deque(maxlen=rememberShots)
        data['pheArray'] = np.arange( phEngs[0],phEngs[1] )
        data['VLSvPHE'] = np.zeros( (vls.size,data['pheArray'].size ) )
        data['goose']=deque(maxlen=rememberShots)
        data['gasoff']=deque(maxlen=rememberShots)
        data['delay']=deque(maxlen=rememberShots)
    
    idx0 = int(vlsCoM)
    phRange = phEngs[1]-phEngs[0]
    idx1 = int(phe-phEngs[0])
    if idx1 < 0:
        idx1 = 0
    elif idx1 >= phRange:
        idx1=phRange-1
    
    data['VLSvPHE'][idx0,idx1] += 1
    data['vlsCoM'].append(vlsCoM)
    data['vlsSum'].append( np.nansum(vls) )
    data['pe'].append(pe)
    data['nITOF'].append(nITOF)
    data['phe'].append(phe)
    data['goose'].append(goose)
    data['gasoff'].append(gasoff)
    if detectors['vitaraDelay']['shotData'] is None:
        data['delay'].append(-1.)
    else:
        data['delay'].append(detectors['vitaraDelay']['shotData'])
    return data

def plotXAS(data,nevt):
    if data is None:
        return data
    
    x=np.array(data['vlsCoM']).astype(float)
    y=np.array(data['nITOF']).astype(float)
    plotXY(nevt, plotName='XASonVLS',x=x,y=y,ylabel='iToF ROI/pulse energy',xlabel='VLS CoM',plotTitle='XAS using VLS',formats='.')

    plotXY(nevt, plotName='vlsCoM', x=np.arange(len(data['vlsCoM'])), y=x, ylabel='VLS CoM [pixel #]', xlabel='time bins', plotTitle='VLS CoM against lab time', formats='-')

    plotXY(nevt, plotName='XASwPE',x=data['phe'],y=data['nITOF'],ylabel='iToF ROI/pulse energy',xlabel='Photon energy from system',plotTitle='XAS using photon energy',formats='.')
    plotXY(nevt, plotName='VLSvPE',x=data['pe'],y=data['vlsSum'],xlabel='pulse energy',ylabel='Integrated VLS signal',plotTitle='VLS, pulse energy correlation',formats='.')
    
    NX,NY = data['VLSvPHE'].shape 
    plotImage(nevt, plotName='VLSvPHE', plotTitle='2D histogram of VLS CoM vs. system photon energy', im=data['VLSvPHE'],aspect_ratio=float(NY)/float(NX))

    # histogram
    histy, histx = np.histogram(np.array(data['vlsCoM'])[np.array(data['nITOF'])>0], 10, weights=np.array(data['nITOF'])[np.array(data['nITOF'])>0])

    histn = np.zeros_like(histy)
    histidx = np.digitize(np.array(data['vlsCoM'])[np.array(data['nITOF'])>0], histx)
    for i in range(histy.size):
        histn[i] = np.sum(histidx == i)
    histy = histy / histn
    plotXY(nevt, plotName='XASonVLShist', x=histx[:-1], y=histy, ylabel='iToF ROI/pulse energy',xlabel='VLS CoM',plotTitle='XAS hist using VLS')
    
    # goose histogram difference
    time_zero = 5438.224
    
    filt = (np.array(data['nITOF'])>0) & (~np.array(data['goose'])) & (~np.array(data['gasoff'])) & (np.array(data['delay'])>0)
    filt_late = filt & (np.array(data['delay']) >= time_zero)
    filt_early = filt & (np.array(data['delay']) < time_zero)

    if (np.sum(filt_late)>0) & (np.sum(filt_early)>0):

        histy_l, histx_l = np.histogram(np.array(data['vlsCoM'])[filt_late], 10, weights=np.array(data['nITOF'])[filt_late])
        histy_e, histx_e = np.histogram(np.array(data['vlsCoM'])[filt_early], histx_l, weights=np.array(data['nITOF'])[filt_early])

        histn_l = np.zeros_like(histy_l)
        histidx_l = np.digitize(np.array(data['vlsCoM'])[filt_late], histx_l)
        for i in range(histy_l.size):
            histn_l[i] = np.sum(histidx_l == i)
        histy_l = histy_l / histn_l

        median_l = np.median(histn_l)
        histy_l[histn_l < 40] = None

        histn_e = np.zeros_like(histy_e)
        histidx_e = np.digitize(np.array(data['vlsCoM'])[filt_early], histx_e)
        for i in range(histy_e.size):
            histn_e[i] = np.sum(histidx_e == i)
        histy_e = histy_e / histn_e

        median_e = np.median(histn_e)
        histy_e[histn_e < 40] = None

        plotXY(nevt, plotName='XASonVLShistdiff', x=histx_l[:-1], y=2*(histy_e - histy_l)/(histy_e + histy_l), ylabel='percent difference', xlabel='VLS CoM',plotTitle='percent difference, time early - late, XAS hist using VLS')

        # early and late separate
        plotXY(nevt, plotName='XASonVLShistearly', x=histx_e[:-1], y=histy_e, ylabel='iToF ROI/pulse energy',xlabel='VLS CoM',plotTitle='XAS hist using VLS, laser early')

        plotXY(nevt, plotName='XASonVLShistearlyn', x=histx_e[:-1], y=histn_e, ylabel='# shots',xlabel='VLS CoM',plotTitle='XAS hist using VLS, laser early number of shots')
    
        plotXY(nevt, plotName='XASonVLShistlate', x=histx_l[:-1], y=histy_l, ylabel='iToF ROI/pulse energy',xlabel='VLS CoM',plotTitle='XAS hist using VLS, laser late')

        plotXY(nevt, plotName='XASonVLShistlaten', x=histx_l[:-1], y=histn_l, ylabel='# shots',xlabel='VLS CoM',plotTitle='XAS hist using VLS, laser late number of shots')
    
    
    return data

########################################################################
# Test plot
########################################################################
def updateTest(data,detectors,testKey='vitaraDelay'):
    if data is None:
        data={}
        data=deque(maxlen=rememberShots)
    print(detectors[testKey]['shotData'])
    data.append(detectors[testKey]['shotData'])
    return data
    
def plotTest(data,nevt):
    if data is not None:
        plotName = 'test'
        t = np.arange(len(data))
        y = np.array(data).astype(float)
        idx = ~np.isnan(y)
        aplot = psmon.plots.XYPlot(nevt, 'Test v shot',
                                   t[idx], y[idx],
                                   xlabel='shot number',
                                   ylabel='test',
                                   formats='.')
        psmon.publish.send(plotName, aplot)
    return data

########################################################################
# timetool plots
########################################################################
def updatetimetool(data,detectors,rememberShots=6000):
    if data is None:
        data={}
        data['ttfltpos']=deque(maxlen=rememberShots)
        data['ttbinned']=np.zeros((int(1024),))
        data['ttbins'] = np.arange(1025)-0.5
        data['ref']=None
        data['proj']=None
        data['nshots'] = 0
        data['delay']=deque(maxlen=rememberShots)
        
    evr=np.array(detectors['evrs']['shotData'])
    xrayOff=evr[161]
    gasOn=evr[70]
    notgoose=evr[67]
    goose=evr[68]
    if (xrayOff is None) | (goose is None):
        return data
    if xrayOff | goose:
        return data

    ttfltpos = detectors['ttfltpos']['shotData']
    if ttfltpos is None:
        return data

    if ttfltpos > 1023:
        ttfltpos = 1023
    data['ttfltpos'].append(ttfltpos)
    data['ttbinned'] = data['ttbinned']*.9998
    data['ttbinned'][int(ttfltpos)]+=1.0
    
    ref=detectors['ttref']['shotData']
    proj=detectors['ttproj']['shotData']
    if (ref is None)|(proj is None):
        return data
    data['ref']=ref
    data['proj']=proj
    data['nshots']+=1

    if detectors['vitaraDelay']['shotData'] is None:
        data['delay'].append(np.nan)
    else:
        data['delay'].append(detectors['vitaraDelay']['shotData'])

    return data

def plottimetool(data,nevt): 
    if data['nshots'] < 10:
        return data
    if data['ref'] is None:
        return data
    if data['proj'] is None:
        return data
    tts=np.array(data['ttfltpos']).astype(float)
    t=np.arange(tts.size)
    y=data['ttbinned'] / np.nansum(data['ttbinned'])
    plotHist(nevt, plotName='TTHIST', x=data['ttbins'],y=y, plotTitle='TTFLTPOS Histogram',xlabel='TTFLTPOS',ylabel='Normalized counts')
    plotXY(nevt, plotName='TTLAB',x=t,y=tts,plotTitle='TTFLTPOS vs lab time',formats='.')
    
    ref=data['ref']
    proj=data['proj']
    idx=np.arange(len(ref))
    diff=proj-ref
    plotXY(nevt, plotName='TTREF',x=idx,y=ref,plotTitle='TTREF',xlabel='idx',ylabel='REF',formats='-')
    plotXY(nevt, plotName='TTPROJ',x=idx,y=proj,plotTitle='TTPROJ',xlabel='idx',ylabel='PROJ',formats='-')
    plotXY(nevt, plotName='TTSUB',x=idx,y=diff,plotTitle='TTSUB',xlabel='idx',ylabel='PROJ-REF',formats='-')

    #if np.sum(data['delay']>0) > 0:
    delay_in_tt = -3.49077300e5 * np.array(data['delay']) + 1.89836311e9
    delay_in_ns = -2.73063978e-6 * np.array(data['ttfltpos']) + 5.43823129e3
    y_ns = np.array(data['delay']).astype(float) - delay_in_ns
    y_tt = np.array(data['ttfltpos']).astype(float) - delay_in_tt
        
    plotXY(nevt, plotName='TTLABCOMPENSATEDNS',x=t,y=y_ns,plotTitle='TTFLTPOS_NS-TARGETTIME_NS vs lab time, compensated for laser delay',xlabel='shot #', ylabel='diff in ns', formats='.')

    plotXY(nevt, plotName='TTLABCOMPENSATED',x=t,y=y_tt,plotTitle='TTFLTPOS-TARGETTIME_PIX vs lab time, compensated for laser delay',xlabel='shot #', ylabel='diff in pix', formats='.')


    return data

########################################################################
# knife edge plots
########################################################################
def updateknifeedge(data,detectors,rememberShots=1200):
    if data is None:
        data={}
        data['spx']=deque(maxlen=rememberShots)
        data['diode']=deque(maxlen=rememberShots)
    
    wf=fix_wf_baseline(detectors['wfs']['shotData'][9][0].astype(float))    
    diode = np.sum(wf[1290:2000]) 
    spx = detectors['samplePaddleX']['shotData']
    if (spx is None) | (diode is None):
        return data
    data['spx'].append(spx)
    data['diode'].append(diode)
    return data

def plotknifeedge(data,nevt):
    spx = np.array(data['spx']).astype(float)
    diode = np.array(data['diode']).astype(float)
    t = np.arange(spx.size)

    plotXY(nevt, plotName='knifeedge',x=spx,y=diode,plotTitle='diode sum vs. sample paddle x',xlabel='spx',ylabel='diode sum',formats='.')
    plotXY(nevt, plotName='samplePaddleX',x=t,y=spx,plotTitle='sample paddle x vs lab time',xlabel='shot #',ylabel='spx',formats='-')
    return data

########################################################################
# Detector and plot setup
########################################################################
def setupDetectors():
    detectors = {}
    
    detectors['evrs'] = {'pskey':'timing', 'get':lambda det: det.raw.eventcodes}
    #detectors['pulseEnergy']={'pskey':'gmd', 'get':lambda det: det.raw.energy}
    #detectors['photonEnergy']={'pskey':'ebeam', 'get':lambda det: det.raw.ebeamPhotonEnergy}
    #detectors['wfs']={'pskey':'hsd', 'get':lambda det: det.raw.waveforms}
    detectors['tmo_opal1']={'pskey':'tmo_opal1', 'get':lambda det: det.raw.image}
    detectors['tmo_atmopal']={'pskey':'tmo_atmopal', 'get':lambda det: det.raw.image}
    detectors['tmo_opal2']={'pskey':'tmo_opal2', 'get':lambda det: det.raw.image}
    #detectors['phaseShift']={'pskey':'las_ph_shift', 'get':lambda det: det}
    detectors['vitaraDelay']={'pskey':'las_fs14_target_time', 'get':lambda det: det}
    detectors['vls']={'pskey':'andor', 'get':lambda det: det.raw.value}
    #detectors['ttfltpos']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.fltpos}
    #detectors['ttref']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.proj_ref}
    #detectors['ttproj']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.proj_sig}
#     detectors['ttfltpos']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.fltpos}
#     detectors['ttref']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.proj_ref}
#     detectors['ttproj']={'pskey':'tmo_atmopal', 'get':lambda det: det.ttfex.proj_sig}
    #detectors['samplePaddleX'] = {'pskey':'sample_paddle_x','get':lambda det: det}

    return detectors

def setupPlots():
    plots = {}
#     plots['testReadout']={'data':None,'updater':updateTest,'plotter':plotTest, 'plotEvery':1}

    # Nominal laser delay vs lab time (target position)
    #plots['laserdelay']={'data':None,'updater':updatelaserdelay,'plotter':plotlaserdelay, 'plotEvery':1}

    # Timetool plots: TTHIST, TTLAB, TTPROJ, TTREF, TTSUB
    #plots['timetool']={'data':None,'updater':updatetimetool,'plotter':plottimetool, 'plotEvery':1}
    
    
#     plots['photonEnergyVSlabtime']={'data':None,'updater':updatePEvLT,'plotter':plotPEvLT}
    plots['EvrHistogram']={'data':None, 'updater':updateEvrHistogram, 'plotter':plotEvrHistogram, 'plotEvery':1}
    #plots['wf0']={'data':None,'updater':updatewf0,'plotter':plotwf0}
    #plots['wfgoose']={'data':None,'updater':updatewfgoose,'plotter':plotwfgoose,'plotEvery':20}
    #plots['wf0ave']={'data':None,'updater':updatewf0ave, 'plotter':plotwf0ave}
    #plots['wf0rollave']={'data':None,'updater':updatewf0rollave,'plotter':plotwf0rollave,'plotEvery':1}
#     plots['ktof']={'data':None,'updater':updatektof,'plotter':plotktof}
#     plots['ktofave']={'data':None, 'updater':updatektofave,'plotter':plotktofave}
#     plots['ktofrollave']={'data':None,'updater':updatektofrollave, 'plotter':plotktofrollave}
    plots['lastvmi']={'data':None, 'updater':updatelastvmi,'plotter':plotlastvmi,'plotEvery':1}
    plots['vmihits']={'data':None,'updater':updatevmihits,'plotter':plotvmihits,'plotEvery':10}
    plots['numehits']={'data':None,'updater':updatenumehits,'plotter':plotnumehits}
#     plots['pbasex']={'data':None,'updater':updatepbasex,'plotter':plotpbasex,'plotEvery':10}
    #plots['gdhist']={'data':None,'updater':updategdhist,'plotter':plotgdhist}
    plots['tt']={'data':None,'updater':updatett,'plotter':plottt,'plotEvery':5}
    #plots['wflodiode']={'data':None,'updater':updatewflodiode,'plotter':plotwflodiode}
    #plots['translodiode']={'data':None,'updater':updatetranslodiode,'plotter':plottranslodiode}
    
#    # Live plot of the VLS
    plots['VLS']={'data':None,'updater':updateVLS,'plotter':plotVLS, 'plotEvery':1}
#    plots['vlsGauss']={'data':None, 'updater':updateVLSgauss, 'plotter':plotVLSgauss, 'plotEvery':1}
    
#    # Generates these plots: XASonVLS, XASwPE, VLSvPHE, VLSvPE
#    # XASonVLS shows the integrated itof/pulse energy vs. CoM on the VLS
#    # XASwPE shows integrated itof/pulse energy vs. system photon energy
#    # VLSvPHE shows the VLS CoM vs system photon energy
#    # VLSvPE shows the correlation between the integrated VLS and the xray pulse energy
    #plots['XAS']={'data':None,'updater':updateXAS,'plotter':plotXAS, 'plotEvery':1}

    #plots['knifeedge']={'data':None, 'updater':updateknifeedge, 'plotter':plotknifeedge, 'plotEvery':1}
    
    return plots
