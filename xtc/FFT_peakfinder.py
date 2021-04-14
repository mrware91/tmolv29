## Contributors: Thomas Wolf, Yusong Liu, and Matt Ware
from math import log, ceil, floor
import numpy as np

# raw_resp = np.load('/cds/data/psdm/tmo/tmolw5618/results/raw_resp.npy')
raw_resp = np.load('/cds/home/m/mrware/Workspace/2021-02-tmolw56/2021-02-preproc-git/xtc/raw_resp.npy')

def FFTfind_fixed(hsd, nmax=1000):
    """
    Wrapper function for FFT peakfinder to fit into preprocessing.
    Arguments:
    hsd : TMO Digitizer data
    nmax: Max number of hits per shot (length of output array).
    """
    x = hsd[0]['times']
    y = fix_wf_baseline(hsd[0][0].astype(float))
    timesF = np.zeros(nmax)*np.nan
    amplitudesF = np.zeros(nmax)*np.nan
    Peaklist = peakfinder(np.array([y]), 5, raw_resp, 7, 20)
    
    for i, peak in enumerate(Peaklist):
        if len(peak)!=0:
            amplitudesF[i] = 1
            timesF[i] = x[peak[1]]
    return timesF, amplitudesF

def fix_wf_baseline(hsd_in, bgfrom=500*64):
    hsd_out = np.copy(hsd_in)
    for i in range(4):
        hsd_out[i::4] -= hsd_out[bgfrom+i::4].mean()
    for i in (12, 13, 12+32, 12+32):
        hsd_out[i::64] -= hsd_out[bgfrom+i::64].mean()
    return hsd_out

def extract_electon_hits(dat):
    """
    This function takes in a number of ToF traces and returns the response function created from the electron hits
    in the traces.
    Arguments:
    dat:          2D numpy array with ToF traces in first dimension
    raw_resp:     response function in time representation.
    responseUN_f: response function in frequency representation.
    """
    
    # Invert data and set baseline to zero:
    # Make a histogram of the values and subtract the bin with the most entries from the data.
    med = np.median(dat.flatten())
    dat = med-dat
    
    # Identify the traces with electron hits:
    # Find traces in dat with values higher than 10 times the standard deviation.
    datstd = dat.std()
    hitlist = np.unique(np.where(dat>10*datstd)[0])
    print('Found '+str(len(hitlist))+' traces with hits.')
    
    #Identify and collect peaks:
    trace = np.zeros_like(dat[0,:])
    peaks = []
    for i in hitlist:
        trace[:] = dat[i,:]
        # Set all values below 2000 to zero.
        trace[np.where(trace<20)] = 0
        peakinds = []
        # Iterate, until traces doesn't contain values other than 0 anymore.
        while np.any(trace>0):
            # Identify indices in a range of -20 to 70 points around maximum value of the trace. Account for indices very
            # Close to the beginning or the end of the trace. Set trace in the range of the indices to zero
            maxind = trace.argmax()
            if maxind<=20:
                trace[:maxind+70] = 0
                peakinds.append([0,maxind+70])
            elif maxind>=len(trace)-70:
                trace[maxind-20:] = 0
                peakinds.append([maxind-20,len(trace)])
            else:
                trace[maxind-20:maxind+70] = 0
                peakinds.append([maxind-20,maxind+70])
        # Extract peaks according to indices into list.
        for ind in peakinds:
            peaks.append(dat[i,ind[0]:ind[1]])
    # Find maximum range of peak indices.
    peaklen = 0
    for peak in peaks:
        if len(peak)>peaklen: peaklen= len(peak)
    # Make 2D numpy array of peaks with uniform length.
    nppeaks = np.zeros((len(peaks),peaklen))
    for i,peak in enumerate(peaks):
        nppeaks[i,:len(peak)] = peak
        
    
    # Align x-axis for all peaks:
    inds = np.arange(len(nppeaks[0,:]))
    peaks_aligned = np.zeros_like(nppeaks)
    peaks_aligned[0,:] = nppeaks[0,:]
    for i in np.arange(1,len(nppeaks[:,0])):
        sums = np.zeros_like(inds)
        for j in inds:
            sums[j] = np.sum(nppeaks[0:i-1,:].sum(axis=0)*np.roll(nppeaks[1,:],j))
        rollind = inds[sums.argmax()]
        peaks_aligned[i,:] = np.roll(nppeaks[i,:],rollind)
    
    # Make response function:
    responseUN = np.zeros_like(peaks_aligned[0,:])
    responseUN_f = np.zeros_like(responseUN)

    for i in np.arange(len(peaks_aligned[:,0])):
        temp1 = peaks_aligned[i,:]
        responseUN += temp1
        temp3 = np.fft.fft(temp1)
        responseUN_f += abs(temp3)
    
    raw_resp = responseUN/len(peaks_aligned[:,0])
    return raw_resp, responseUN_f

def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return max(possible_results, key= lambda z: abs(x-2**z))

def BuildFilterFunction(npts,width,dt=0.5):
    """
    Function to build Fourier Filter
    npts is number of point in desired trace.
    width is the width of the filter function
    """
    dt=2 # time bins are seperated by 1 ns, dt=1 mean that f is in GHz.
    df = 2*np.pi/(npts*dt) # Frequency spacing due to time sampling.
    f = np.concatenate((np.arange(npts/2),np.arange(-npts/2,0))) * df # frequency for FFT (dt=1 => GHz)

    fdw = f/width # f/w
    retF = np.square(np.sin(fdw)/fdw) # filter function
    retF[0] = 1; # fix NaN at first index
    retF = retF*( abs(f) <= width*np.pi ) # set filter to zero after first minimum.
    return retF

def peakfinder(data,threshold,raw_resp,deadtime=20, nskip=20):
    """
    Deconvolution peakfinder function
    Arguments:
    data     : 2D numpy array with shots in first dimension and waveform in second dimension
    threshold: Threshold value for peak identification
    raw_resp : 1D numpy array containing the peak response function
    deadtime : Deadtime of the MCP in samples
    nskip    : Delay of onset of response function
    """
    # Invert data and set baseline to zero:
    med = np.median(data.flatten())
    data = med-data
    #Get data in shape of multiples of 2:
    lendata = 2**(closest_power(len(data[0,:]))+1)
    data2 = np.zeros((len(data[:,0]),lendata))
    data2[:,:len(data[0,:])] = data
    
    # Find possible peaks
    std = data2.std()
    inds = np.where(data2>std*threshold)
    tracelist = np.unique(inds[0])
    if len(tracelist)==0:
        return [[], []]
    data3 = data2[tracelist,:]
    
    # Filter data with response function:
    r = np.zeros((len(data3[0,:]),))
    r[:len(raw_resp)] = raw_resp
    R = np.fft.fft(r)
    
    w = 0.5
    F = BuildFilterFunction(lendata,w)
    
    R, a = np.meshgrid(R,tracelist)
    F, a = np.meshgrid(F,tracelist)
    s = data3
    S = np.fft.fft(s,len(s[0,:]),1)
    
    D = S/R
    D = D*F
    d = np.fft.ifft(D,len(D[0,:]),1)
    
    # Compensate for onset of response function
    temp = d.copy()

    d[:,:nskip] = 0;
    d[:,nskip:] = temp[:,:-nskip]
    
    # Make sure only one hit is counted per peak:
    Peaklist = []
    for i in np.arange(len(tracelist)):
        inds = peakfind(d[i,:],threshold, deadtime)
        for ind in inds:
            Peaklist.append([tracelist[i],ind])
    return Peaklist

def peakfind(s,t,deadt):
    """
    Hit finder function.
    s:  Signal
    t:  Threshold for hitfinding.
    """

    Hi = []; # initializes peaks index vector

    thresh = s.mean() + t*s.std() # Set Threshold based on raw data
    s[np.where(s<thresh)] = 0
    
    
    if s.sum() > 0:
        x = np.where(s>0)[0]
#         Take out peaks at the edge of the trace:
        x = x[np.where((x>2)&(x<len(s)-2))]
        
        inds = []
        for i in np.arange(len(x)):
            if i!=0 and x[i] != x[i-1] + 1: # Looks if index belongs to same peak
                newind = inds[s[inds].argmax()]
                if len(Hi)>0:               # Looks if index is within deadtime of earlier peak
                    if newind-Hi[-1]>deadt:
                        Hi.append(newind)
                else:
                    Hi.append(newind)
                inds = []
            inds.append(x[i])
        if len(inds)!=0:
            newind = inds[s[inds].argmax()]
            if len(Hi)>0:
                if newind-Hi[-1]>20:
                    Hi.append(newind)
            else:
                Hi.append(newind)
    return np.array(Hi)