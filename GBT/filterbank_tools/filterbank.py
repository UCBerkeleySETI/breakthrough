#!/usr/bin/env python
"""
# filterbank.py

Python class and command line utility for reading and plotting filterbank files.

This provides a class, Filterbank(), which can be used to read a .fil file:

    ````
    fil = Filterbank('test_psr.fil')
    print fil.header
    print fil.data.shape
    print fil.freqs
    
    plt.figure()
    fil.plot_spectrum(t=0)
    plt.show()
    ````

TODO: check the file seek logic works correctly for multiple IFs

"""

import os
import sys
import struct
import numpy as np
import pylab as plt
from pprint import pprint

from astropy import units as u
from astropy.coordinates import Angle

MAX_PLT_POINTS      = 65536                  # Max number of points in matplotlib plot
MAX_IMSHOW_POINTS   = (8192, 4096)           # Max number of points in imshow plot
MAX_DATA_ARRAY_SIZE = 1024 * 1024 * 1024     # Max size of data array to load into memory

###
# useful helper functions
###


def db(x): 
    """ Convert linear to dB """
    return 10*np.log10(x)

def lin(x):
    """ Convert dB to linear """
    return 10.0**(x / 10.0)

def closest(xarr, val):
    """ Return the index of the closest in xarr to value val """
    idx_closest = np.argmin(np.abs(xarr - val))
    return idx_closest

def rebin(d, n_x, n_y=None):
    """ Rebin data by averaging bins together
    
    Args:
    d (np.array): data
    n_x (int): number of bins in x dir to rebin into one
    n_y (int): number of bins in y dir to rebin into one
    
    Returns: 
    d: rebinned data with shape (n_x, n_y)
    """

    if d.ndim == 2:
        d = d[:int(d.shape[0] / n_x) * n_x, :int(d.shape[1] / n_y) * n_y]
        d = d.reshape((d.shape[0] / n_x, n_x, d.shape[1] / n_y, n_y))
        d = d.mean(axis=3)
        d = d.mean(axis=1)
    elif d.ndim == 1:
        d = d[:int(d.shape[0] / n_x) * n_x]
        d = d.reshape((d.shape[0] / n_x, n_x))
        d = d.mean(axis=1)
    else:
        raise RuntimeError("Only NDIM <= 2 supported")
    return d

###
# Header parsing
###

# Dictionary of allowed keywords and their types
# Here are the keywordss that a filter bank file may
# contain.  Items marked with "[*]" are not yet # supported.  See docs for
# indivisuabl attribtues for more detailed info.
# 
#   * telescope_id (int): 0=fake data; 1=Arecibo; 2=Ooty... others to be added
#   * machine_id (int): 0=FAKE; 1=PSPM; 2=WAPP; 3=OOTY... others to be added
#   * data_type (int): 1=filterbank; 2=time series... others to be added
#   * rawdatafile (string): the name of the original data file
#   * source_name (string): the name of the source being observed by the telescope
#   * barycentric (int): equals 1 if data are barycentric or 0 otherwise
#   * pulsarcentric (int): equals 1 if data are pulsarcentric or 0 otherwise
#   * az_start (double): telescope azimuth at start of scan (degrees)
#   * za_start (double): telescope zenith angle at start of scan (degrees)
#   * src_raj (double): right ascension (J2000) of source (hours, converted from hhmmss.s)
#   * src_dej (double): declination (J2000) of source (degrees, converted from ddmmss.s)
#   * tstart (double): time stamp (MJD) of first sample
#   * tsamp (double): time interval between samples (s)
#   * nbits (int): number of bits per time sample
#   * nsamples (int): number of time samples in the data file (rarely used any more)
#   * fch1 (double): centre frequency (MHz) of first filterbank channel
#   * foff (double): filterbank channel bandwidth (MHz)
#   * FREQUENCY_START [*] (character): start of frequency table (see below for explanation)
#   * fchannel [*] (double): frequency channel value (MHz)
#   * FREQUENCY_END [*] (character): end of frequency table (see below for explanation)
#   * nchans (int): number of filterbank channels
#   * nifs (int): number of seperate IF channels
#   * refdm (double): reference dispersion measure (pc/cm**3)
#   * period (double): folding period (s)
#   * nbeams (int):total number of beams (?)
#   * ibeam (int): number of the beam in this file (?)

header_keyword_types = {
    'telescope_id' : '<l',
    'machine_id'   : '<l',
    'data_type'    : '<l',
    'barycentric'  : '<l',
    'pulsarcentric': '<l',
    'nbits'        : '<l',
    'nsamples'     : '<l',
    'nchans'       : '<l',
    'nifs'         : '<l',
    'nbeams'       : '<l',
    'ibeam'        : '<l',
    'rawdatafile'  : 'str',
    'source_name'  : 'str',
    'az_start'     : '<d',
    'za_start'     : '<d',
    'tstart'       : '<d',
    'tsamp'        : '<d',
    'fch1'         : '<d',
    'foff'         : '<d',
    'refdm'        : '<d',
    'period'       : '<d',
    'src_raj'      : 'angle',
    'src_dej'      : 'angle',
    }

def grab_header(filename):
    """ Extract the filterbank header from the file 
    
    Args:
        filename (str): name of file to open
    
    Returns:
        header_str (str): filterbank header as a binary string
    """
    f = open(filename, 'r')
    eoh_found = False
    
    header_str = ''
    header_sub_count = 0
    while not eoh_found:
        header_sub = f.read(512)
        header_sub_count += 1
        if 'HEADER_START' in header_sub:
            idx_start = header_sub.index('HEADER_START') + len('HEADER_START')
            header_sub = header_sub[idx_start:]
        
        if 'HEADER_END' in header_sub:
            eoh_found = True
            idx_end = header_sub.index('HEADER_END')
            header_sub = header_sub[:idx_end]
        header_str += header_sub
        
    f.close()
    return header_str

def len_header(filename):
    """ Return the length of the filterbank header, in bytes 
    
    Args:
        filename (str): name of file to open
    
    Returns:
        idx_end (int): length of header, in bytes    
    """
    f = open(filename, 'r')
    
    header_sub_count = 0
    eoh_found = False
    while not eoh_found:
        header_sub = f.read(512)
        header_sub_count += 1
        if 'HEADER_END' in header_sub:
            idx_end = header_sub.index('HEADER_END') + len('HEADER_END')
            eoh_found = True
            break
        
    idx_end = (header_sub_count -1) * 512 + idx_end 
    f.close()
    return idx_end

def parse_header(filename):
    """ Parse a header of a filterbank, looking for allowed keywords 
    
    Uses header_keyword_types dictionary as a lookup for data types. 
    
    Args:
        filename (str): name of file to open
        
    Returns:
        header_dict (dict): A dictioary of header key:value pairs
    """
    header = grab_header(filename)
    header_dict = {}
    
    #print header
    for keyword in header_keyword_types.keys():
        if keyword in header:
            dtype = header_keyword_types.get(keyword, 'str')
            idx = header.index(keyword) + len(keyword)
            dtype = header_keyword_types[keyword]
            if dtype == '<l':
                val = struct.unpack(dtype, header[idx:idx+4])[0]
                header_dict[keyword] = val
            if dtype == '<d':
                val = struct.unpack(dtype, header[idx:idx+8])[0]
                header_dict[keyword] = val
            if dtype == 'str':
                str_len = struct.unpack('<L', header[idx:idx+4])[0]
                str_val = header[idx+4:idx+4+str_len]
                header_dict[keyword] = str_val
            if dtype == 'angle':
                val = struct.unpack('<d', header[idx:idx+8])[0]
                val = fil_double_to_angle(val)
                
                if keyword == 'src_raj':
                    val = Angle(val, unit=u.hour)
                else:
                    val = Angle(val, unit=u.deg)
                header_dict[keyword] = val                
    
    return header_dict

def fil_double_to_angle(angle):
      """ Reads a little-endian double in ddmmss.s (or hhmmss.s) format and then
      converts to Float degrees (or hours).  This is primarily used to read
      src_raj and src_dej header values. """
      
      negative = (angle < 0.0)
      angle = np.abs(angle)

      dd = np.floor((angle / 10000))
      angle -= 10000 * dd
      mm = np.floor((angle / 100))
      ss = angle - 100 * mm
      dd += mm/60.0 + ss/3600.0 
      
      if negative:
          dd *= -1
      
      return dd


###
# Main filterbank class
###

class Filterbank(object):
    """ Class for loading and plotting filterbank data """
    
    def __repr__(self):
        return "Filterbank data: %s" % self.filename
    
    def __init__(self, filename, f_start=None, f_stop=None, 
                 t_start=None, t_stop=None, load_data=True):
        """ Class for loading and plotting filterbank data.
        
        This class parses the filterbank file and stores the header and data
        as objects:
            fb = Filterbank('filename_here.fil')
            fb.header        # filterbank header, as a dictionary
            fb.data          # filterbank data, as a numpy array
        
        Args:
            filename (str): filename of filterbank file.
            f_start (float): start frequency in MHz
            f_stop (float): stop frequency in MHz
            t_start (int): start integration ID
            t_stop (int): stop integration ID
            load_data (bool): load data. If set to False, only header will be read.
        """
        
        self.filename = filename
        self.header = parse_header(filename)
 
        ## Setup frequency axis
        f0 = self.header['fch1'] 
        f_delt = self.header['foff']
        self.freqs = np.arange(0, self.header['nchans'], 1, dtype='float64') * f_delt + f0
        
        # keep this seperate!
        file_freq_mapping =  np.arange(0, self.header['nchans'], 1, dtype='float64') * f_delt + f0
        
            
        # Slice out frequency start & stop if required
        chan_start_idx, chan_stop_idx = 0, self.header['nchans']
        if f_delt < 0:
            chan_start_idx, chan_stop_idx = self.header['nchans'], 0
            
            
        if f_start:
            chan_start_idx = closest(file_freq_mapping, f_start)
        if f_stop:
            chan_stop_idx = closest(file_freq_mapping, f_stop)
        
        if chan_start_idx > chan_stop_idx:
            self.freqs    = self.freqs[chan_stop_idx:chan_start_idx]
            self.freqs    = self.freqs[::-1]
        else:
            self.freqs    = self.freqs[chan_start_idx:chan_stop_idx]

        
        # Load binary data 
        self.idx_data = len_header(filename)
        f = open(filename, 'r')
        f.seek(self.idx_data)
        
        n_bytes  = self.header['nbits'] / 8
        n_chans = self.header['nchans']
        n_chans_selected = self.freqs.shape[0]
        n_ifs   = self.header['nifs']
        
        # only read first integration of large file (for now, later more flexible)
        filesize = os.path.getsize(self.filename)
        n_bytes_data = filesize - self.idx_data
        n_ints_in_file = n_bytes_data / (n_bytes * n_chans * n_ifs)
        
        # now check to see how many integrations requested
        ii_start, ii_stop = 0, n_ints_in_file
        if t_start:
            ii_start = t_start
        if t_stop:
            ii_stop = t_stop
        n_ints = ii_stop - ii_start
        
        
        # Set up indexes used in file read (taken out of loop for speed)
        i0 = np.min((chan_start_idx, chan_stop_idx))
        i1 = np.max((chan_start_idx, chan_stop_idx)) 
                
        if load_data:

            if n_ints * n_ifs * n_chans_selected > MAX_DATA_ARRAY_SIZE:
                print "Error: data array is too large to load. Either select fewer"
                print "points or manually increase MAX_DATA_ARRAY_SIZE."
                exit()
                
            self.data = np.zeros((n_ints, n_ifs, n_chans_selected), dtype='float32')
            
            for ii in range(n_ints):
                """d = f.read(n_bytes * n_chans * n_ifs)  
                """
                
                for jj in range(n_ifs):
       
                    f.seek(n_bytes * i0, 1) # 1 = from current location
                    d = f.read(n_bytes * n_chans_selected)          
                    f.seek(n_bytes * (n_chans - i1), 1)
                
                    if n_bytes == 4:
                        dd = np.fromstring(d, dtype='float32')
                    elif n_bytes == 2:
                        dd = np.fromstring(d, dtype='int16')
                    elif n_bytes == 1:
                        dd = np.fromstring(d, dtype='int8')
                    
                    # Reverse array if frequency axis is flipped
                    if f_delt < 0:
                        dd = dd[::-1]
                    
                    self.data[ii, jj] = dd        
        else:
            self.data = np.array([0])
            
        ## Setup time axis
        t0 = self.header['tstart']
        t_delt = self.header['tsamp']
        self.timestamps = np.arange(0, n_ints) * t_delt + t0
        
        # Finally add some other info to the class as objects
        self.n_ints_in_file  = n_ints_in_file
        self.file_size_bytes = filesize
        
    
    def info(self):
        """ Print header information """
        
        for key, val in self.header.items():
            if key == 'src_raj':
                val = val.to_string(unit=u.hour, sep=':')
            if key == 'src_dej':
                val = val.to_string(unit=u.deg, sep=':')
            print "%16s : %32s" % (key, val)
            
        print "\n%16s : %32s" % ("Num ints in file", self.n_ints_in_file)
        print "%16s : %32s" % ("Data shape", self.data.shape)
        print "%16s : %32s" % ("Start freq (MHz)", self.freqs[0])
        print "%16s : %32s" % ("Stop freq (MHz)", self.freqs[-1])

    def grab_data(self, f_start=None, f_stop=None, if_id=0):
        """ Extract a portion of data by frequency range.
        
        Args:
            f_start (float): start frequency in MHz
            f_stop (float): stop frequency in MHz
            if_id (int): IF input identification (req. when multiple IFs in file)
        
        Returns:
            (freqs, data) (np.arrays): frequency axis in MHz and data subset
        """
        i_start, i_stop = 0, None
        
        if f_start:
            i_start = closest(self.freqs, f_start)
        if f_stop:
            i_stop = closest(self.freqs, f_stop)

        plot_f    = self.freqs[i_start:i_stop]
        plot_data = self.data[:, if_id, i_start:i_stop]
        return plot_f, plot_data      

    def plot_spectrum(self, t=0, f_start=None, f_stop=None, logged=False, if_id=0, c=None, **kwargs):
        """ Plot frequency spectrum of a given file 
        
        Args:
            t (int): integration number to plot (0 -> len(data))
            logged (bool): Plot in linear (False) or dB units (True)
            if_id (int): IF identification (if multiple IF signals in file)
            c: color for line
            kwargs: keyword args to be passed to matplotlib plot()
        """
        ax = plt.gca()
        
        plot_f, plot_data = self.grab_data(f_start, f_stop, if_id)
        
        if isinstance(t, int):
            print "extracting integration %i..." % t
            plot_data = plot_data[t]
        elif t == 'all':
            print "averaging along time axis..."
            plot_data = plot_data.mean(axis=0)
        else:
            raise RuntimeError("Unknown integration %s" % t)

        # Rebin to max number of points
        dec_fac_x = 1
        if plot_data.shape[0] > MAX_PLT_POINTS:
            dec_fac_x = plot_data.shape[0] / MAX_PLT_POINTS
        
        plot_data = rebin(plot_data, dec_fac_x, 1)
        plot_f    = rebin(plot_f, dec_fac_x, 1)
        
        if not c:
            kwargs['c'] = '#333333'
        
        if logged:
            plt.plot(plot_f, db(plot_data), **kwargs)
            plt.ylabel("Power [dB]")
        else:
            
            plt.plot(plot_f, plot_data, **kwargs)
            plt.ylabel("Power [counts]")
        plt.xlabel("Frequency [MHz]")

        try:
            plt.title(self.header['source_name'])
        except KeyError:
            plt.title(self.filename)
        
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.xlim(plot_f[0], plot_f[-1])
    
    def plot_waterfall(self, f_start=None, f_stop=None, if_id=0, logged=True, **kwargs):
        """ Plot waterfall of data 
        
        Args:
            f_start (float): start frequency, in MHz
            f_stop (float): stop frequency, in MHz
            logged (bool): Plot in linear (False) or dB units (True),
            kwargs: keyword args to be passed to matplotlib imshow()
        """
        plot_f, plot_data = self.grab_data(f_start, f_stop, if_id)
        
        if logged:
            plot_data = db(plot_data)
        
        # Make sure waterfall plot is under 4k*4k
        dec_fac_x, dec_fac_y = 1, 1
        if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
            dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
            
        if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
            dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
        
        plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
        
        
        try:
            plt.title(self.header['source_name'])
        except KeyError:
            plt.title(self.filename)
        
        plt.imshow(plot_data, 
            aspect='auto',
            rasterized=True,
            interpolation='nearest',
            extent=(plot_f[0], plot_f[-1], self.timestamps[-1], self.timestamps[0]),
            cmap='viridis',
            **kwargs
        )
        plt.colorbar()
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [MJD]")
        

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Command line utility for reading and plotting filterbank files.")
    
    parser.add_argument('-w', action='store_true', default=False, dest='waterfall', 
                        help='Show waterfall (freq vs. time) plot')
    parser.add_argument('filename', type=str, 
                        help='Name of file to read')
    parser.add_argument('-b', action='store', default=None, dest='f_start', type=float,
                        help='Start frequency (begin), in MHz')
    parser.add_argument('-e', action='store', default=None, dest='f_stop', type=float,
                        help='Stop frequency (end), in MHz')    
    parser.add_argument('-B', action='store', default=None, dest='t_start', type=int,
                        help='Start integration (begin) ID')
    parser.add_argument('-E', action='store', default=None, dest='t_stop', type=int,
                        help='Stop integration (end) ID')    
    parser.add_argument('-i', action='store_true', default=False, dest='info_only',
                        help='Show info only')
    parser.add_argument('-a', action='store_true', default=False, dest='average',
                       help='average along time axis (plot spectrum only)')
    
    args = parser.parse_args()
    
    # Open filterbank data
    filename = args.filename
    load_data = not args.info_only
    
    # only load one integration if looking at spectrum
    if not args.waterfall:
        if args.t_start == None:
            t_start = 0
        else:
            t_start = args.t_start
        t_stop  = t_start + 1
        
        if args.average:
            t_start = None
            t_stop  = None
        
    else:
        t_start = args.t_start
        t_stop  = args.t_stop
                
    fil = Filterbank(filename, f_start=args.f_start, f_stop=args.f_stop, 
                     t_start=t_start, t_stop=t_stop, 
                     load_data=load_data)
    fil.info()
    
    # And if we want to plot data, then plot data.
    if not args.info_only:
        # check start & stop frequencies make sense
        #try:
        #    if args.f_start:
        #        print "Start freq: %2.2f" % args.f_start
        #        assert args.f_start >= fil.freqs[0] or np.isclose(args.f_start, fil.freqs[0])
        #    
        #    if args.f_stop:
        #        print "Stop freq: %2.2f" % args.f_stop
        #        assert args.f_stop <= fil.freqs[-1] or np.isclose(args.f_stop, fil.freqs[-1])
        #except AssertionError:
        #    print "Error: Start and stop frequencies must lie inside file's frequency range."
        #    print "i.e. between %2.2f-%2.2f MHz." % (fil.freqs[0], fil.freqs[-1])
        #    exit()
        
        if not args.waterfall:
            plt.figure("Spectrum", figsize=(8, 6))
        
            fil.plot_spectrum(logged=True, f_start=args.f_start, f_stop=args.f_stop, t='all')
        
        # don't bother doing imshow if it's only a few integrations
        if args.waterfall:
            plt.figure("waterfall", figsize=(8, 6))
            fil.plot_waterfall(f_start=args.f_start, f_stop=args.f_stop)
            #plt.clim(75, 85)
        
        plt.show()    