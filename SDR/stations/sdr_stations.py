import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict
from FMstations import *

#This code ultimately produces:
#1. a dictionary of frequencies and their corresponding radio station(s)
#2. a plot depicting such
#Data is taken from a radio receiver hardware as a csv file


raw_data = np.genfromtxt("stationsdata.csv", delimiter = ",", dtype = None)

MINFREQ = 87900000 #minimum frequency of total bandwidth
MAXFREQ = 107900000 #maximum frequency of total bandwidth
SAMP_RATE = 2500000 #sample rate; bandwidth of a line
FREQ_BIN = raw_data[0][4] #frequency interval between each data points
INTERVAL = 10 #time it took for each scan in seconds
TOTAL_TIME = 900 #time it took for the scan session in seconds
SCAN = TOTAL_TIME / INTERVAL #number of scans

#First, we identify and implement the relevant frequency band for the plots

def total_freq(samp_rate, line):
    """total frequency band of the data

    samp_rate: sample rate used to collect the data
    freq_bin: difference in frequency between each data point
    line: number of iterations to create the band;
    each sampling may sample over only a part of the desired bandwidth
    """
    if samp_rate < (MAXFREQ - MINFREQ):
        allfreq = []  
        i = 0
        while i < line:
            freqband = np.arange(MINFREQ + (samp_rate * i), MINFREQ + (samp_rate * (i+1)), FREQ_BIN)
            allfreq = np.append(allfreq, freqband)
            i += 1
        return allfreq
    else:
        return np.arange(MINFREQ, MAXFREQ, FREQ_BIN)

TotalBand = total_freq(raw_data[0][3] - raw_data[0][2], 8)
    
#The data are power values of the signal
#The data should be processed into workable arrays or lists
#It's important to identify repeated scans and irrelevant data

def power_tot(data, line, trunc):
    """returns an array of arrays; all power values from the scans

    line: number of lines of data corresponding to a full scan over the bandwidth
    trunc: truncate number of, if any, of unneeded data, such as time, at the start of each line
    """
    P_tot = []
    j = 0
    while j < SCAN:
        P_band = []
        i = 0
        while i < line:
            P_band = np.append(P_band, data[j*line+i].tolist()[trunc:])
            i += 1
        P_tot.append(P_band)
        j += 1
    return P_tot

total_data = power_tot(raw_data, 8, 6)

def power_avg(data, xax=0, yax=1):
    """returns the time-averaged power values of each index-sharing set of numbers 
    
    data: an array of arrays 
    """
    indices_arr = np.swapaxes(data, xax, yax)
    y = []
    for index in indices_arr:
        y.append(np.average(index))
    return y

avg_data = power_avg(total_data)

#We have the values to make our plot, but we need to improve it to make it reasonably workable
#The plot needs to be smoothed out and the noisy peaks need to be removed

def Flatten(spec, flatter, n):
    """flattens the noise level of our original plot

    dataSpec: input spectrum to be flatted
    flatter: an array derived from the input spectrum used to divided by the spectrum
    n: half the number of a set of points that the median is taken over
    """
    MED = []
    i = 0
    for x in flatter:
        if i < n:
            MED.append(np.median(flatter[0 : i + n + 1])) 
            i += 1 
        elif i >= (len(spec) - (n + 1)):
            MED.append(np.median(flatter[i - n :]))
            i += 1
        else:
            MED.append(np.median(flatter[i - n : i + n + 1]))
            i += 1
    Spectrum = np.array(spec) / np.array(MED) * np.average(spec)
    return Spectrum

FlatterSpec = Flatten(avg_data, avg_data[24582:28679] * 8, 10)
    
#We have a flattened plot, but it's still rugged and noisy

def Reduce(spec, n):
    """takes the median of a set of points, removing noise-produced peaks and much noise
    
    n: half the number of a set of points that the median is taken over
    """
    R = []
    i = 0
    for x in spec:
        if i < n:
            R.append(np.median(spec[0 : i + n + 1]))
            i += 1
        elif i >= 32769:
            R.append(np.median(spec[i - n :]))
            i += 1
        else:
            R.append(np.median(spec[i - n : i + n + 1]))
            i += 1
    return R

ReducedSpec = Reduce(FlatterSpec, 10)

#The plot looks a less noisy, but a zooming inspection shows that the plot is very rugged
#That makes it difficult to systematically indentify some features, such as peaks
#A mathematical smoothing is required; either data fitting or convolution with a kernel

def smooth(spec, win_len, window, beta = 20):
    """smooths a signal with kernel window of win_len number of inputs
    
    spec: Input data spectrum to be smoothed
    win_len: the size of the kernel window used to smooth the input
    window: type of kernel; e.g. 'blackman'
    """
    if window == 'kaiser':
        w = eval('np.'+window+'(win_len, beta)')
    elif window == 'flat':
        w = np.ones(len(win_len, 'd'))
    else:
        w = eval('np.'+window+'(win_len)')
    s = np.r_[spec[win_len-1 : 0 : -1], spec, spec[-1 : -win_len : -1]]
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(win_len / 2 - 1) : int(-win_len / 2)]

SmoothSpec = smooth(ReducedSpec, 150, 'kaiser')

#Plotting the spectra so far to confirm that the code is working properly

def plot_specs():
    f, specs = plt.subplots(4, sharex = True)
    specs[0].plot(TotalBand, avg_data)
    specs[0].set_title('Time-Averaged Radio Spectra Between 87.9 and 107.9 MHz over 90 Scans')
    specs[1].plot(TotalBand, FlatterSpec)
    specs[1].set_title('Response-Flattened Noise Floor')
    specs[1].set_ylabel('Power (dBm)')
    specs[2].plot(TotalBand, ReducedSpec)
    specs[2].set_title('Noise-Reduced Medians')
    specs[3].plot(TotalBand, SmoothSpec)
    specs[3].set_title('Noise-Reduced, Smoothed Spectrum')
    specs[3].set_xlabel('Frequency (Hz)')
    plt.savefig('stations_specs.pdf')
    plt.show()
    
#We have our desired plot; now to use that result to obtain the feature of interest - peaks

def get_peaks(spec, threshold, xreducer, rounder=2):
    """identifies the peaks of a plot. Returns an array of 2 lists:

    1. the indices of the frequencies corresponding to the peaks;
    2. said frequencies, divided by xreducer for simpler units and rounded to rounder decimals
   
    spec: input data spectrum
    threshold: only data above which are taken into account to ignore the noise level
    """
    Peaks = []
    spec = spec.tolist()
    for i in np.arange(len(spec)):
        if spec[i] > threshold and spec[i] > spec[i-1] and spec[i] > spec[i+1]:
            Peaks.append(spec[i])
        else:
            continue
    Ordered_Indices = []
    while True:
        if np.array(Peaks).tolist() == []:
            Ordered_Freq = [(x * FREQ_BIN + MINFREQ) for x in Ordered_Indices]
            Reduced_Freq = np.around((np.array(Ordered_Freq) / xreducer), rounder)
            return [Ordered_Indices, Reduced_Freq.tolist()]
        elif len(Peaks) == 1:
            Ordered_Indices.append(spec.index(Peaks[0]))
            Peaks = np.delete(Peaks, 0)   
        else:
            Ordered_Indices.append(spec.index(np.amax(Peaks)))
            Peaks = np.delete(Peaks, np.array(Peaks).tolist().index(np.amax(Peaks)))

#Remaining to be filtered are extraneous digital peaks and any anomalies not corresponding to stations
#Finally, the stations are marked on the plot next to the corresponding peak

def mark_peaks(src_dict, spec, threshold, title, xreducer, error=.01, bound1='left', bound2='bottom', rot=90):
    """returns both a plot and a dictionary
    
    plot: shows the stations next to the marked peaks
    dictionary: matches the relevant peak frequencies with the corresponding station(s)
    
    src_dict: input dictionary of frequencies and stations from which the results are selected from
    spec: input spectrum data
    threshold: only data above which are taken in account to ignore the noise level
    title: title for the plot
    xreducer: the values of the x-axis divided by which to simpler units
    error: within which the obtained frequencies are acceptable as equivalent to that of a station
    remaining parameters: used the adjust the markings and labels of the plots
    """
    stations = []
    peakfreqs = []
    stations_i = []
    peaker = get_peaks(spec, threshold, xreducer)
    p0 = peaker[0]
    p1 = peaker[1]
    for i in np.arange(len(p1)):
        if p1[i] in src_dict.keys():
            stations.append(src_dict[p1[i]])
            peakfreqs.append(p1[i])
            stations_i.append(p0[i])
        else:
            for x in np.arange(p1[i]-error, p1[i]+error, error):
                if x in src_dict.keys():
                    stations.append(src_dict[x])
                    peakfreqs.append(p1[i])
                    stations_i.append(p0[i])
                else:
                    continue
    peaks = [spec[y] for y in stations_i]
    plt.title(title)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Reduced Power (dBm)')
    yoffset = (np.amax(spec) - np.amin(spec)) / 4
    plt.ylim(np.amin(spec) - yoffset, np.amax(spec) + yoffset)
    plt.plot(np.array(TotalBand) / 1000000, spec)
    plt.scatter(peakfreqs, peaks, marker = 'o', color = 'r', s = 40)
    text_bounds = {'ha':bound1, 'va':bound2}
    for i in np.arange(len(peakfreqs)):
        plt.text(peakfreqs[i], peaks[i] + (yoffset / 10), stations[i], text_bounds, rotation=rot)
    plt.savefig('stations_peaks.pdf')
    plt.show()
    stations_dict = OrderedDict()
    for i in np.arange(len(stations)):
        stations_dict[peakfreqs[i]] = stations[i]
    return stations_dict

#returns a dictionary of the relevant analog peaks and their corresponding stations + a plot showing that
#filters out the digital peaks

print(mark_peaks(BAFMRS, SmoothSpec, -54, 'Bay Area FM Radio Stations', 1000000)) 

#rather than plot the average spectrum, it can be useful to make a waterfall plot, showing time-evolution

def waterfall(data, line, flat1, flat2, n, win_len, window, title, axesrange, gridshape='auto'):
    """returns a waterfall grid consisting off all the spectra from the input 

    data: an array of arrays
    line: number of arrays that make up one full scan of the band
    flat1, flat2: boundaries of the data points from each array used for flattening the spectrum
    n: half the number used to take medians of the spectra for noise-reducing
    win_len: size of kernel window for smoothing
    window: type of kernel
    title: title of grid
    axesrange: boundaries of the values of the grid
    """
    wf = []
    i = 1
    while i <= len(data):
        flatter = data[-i][flat1:flat2]
        flatspec = Flatten(data[-i], np.array(flatter.tolist() * line), n)
        reduspec = Reduce(flatspec, n)
        smoospec = smooth(reduspec, win_len, window)
        wf.append(smoospec)
        i += 1
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Time (s)')
    plt.imshow(wf, extent=axesrange)
    ax.set_aspect(gridshape)
    plt.colorbar(orientation='vertical')
    plt.show()

waterfall(total_data, 8, 24582, 28679, 10, 150, 'kaiser', 'Radio Spectra Waterfall', [87.9, 107.9, 0, 900])

                      


