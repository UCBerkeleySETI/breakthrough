import numpy as np
import pylab as plt
from blimpy import Waterfall
import os, time, fnmatch
from scipy import ndimage, signal
import scipy.misc, fitsio, csv
from skimage import measure
from joblib import Parallel, delayed
from markov import markov
import argparse

parser = argparse.ArgumentParser(description='Process some params')
parser.add_argument('--band', type=str, default='C',
                    help='band of input')
parser.add_argument('--ind', type=int, default=0,
                    help='optional index')
parser.add_argument('--fil', dest='fil', type=str,
                    help='filterbank file')
parser.add_argument('--pref', dest='pref', type=str, default=None,
                    help='prefix to filenames')
parser.add_argument('--pdir', dest='pulse_dir', type=str, default='./pulses/',
                    help='directory to store pulses')
parser.add_argument('--tdir', dest='train_dir', type=str, default='./train/',
                    help='directory to store generated corpus')
parser.add_argument('--ndir', dest='npy_dir', type=str, default=None,
                    help='directory containing fin tune npys')
parser.add_argument('--fdir', dest='fil_dir', type=str, default=None,
                    help='directory containing filterbanks')
parser.add_argument('--mod', dest='mod', type=str, default=None,
                    help='modulation')
args = parser.parse_args()

# freq_range ordered [start freq, stop freq]
FREQ_BAND_TO_INFO = {'L' : {'freq_range' : [1000., 2000.],
                              'DM_range' : [200., 1000.],
                              'pulse_t0' : -320.,
                              'markov_t' : [0.8, 0.95]
                           },
                     'C' : {'freq_range' : [4000., 8000.],#[3562.31689453125, 8438.78173828125],
                              'DM_range' : [200., 1600.],
                              'pulse_t0' : -20.,
                              'markov_t' : [0.95, 0.99]
                           },
                     'M' : {'freq_range' : [819.921875, 851.171875],
                              'DM_range' : [100., 2600.],
                              'pulse_t0' : -400.,
                              'markov_t' : [0.2, 0.95]
                           },
                     'A' : {'freq_range' : [1129, 1465],
                              'DM_range' : [200., 2000.],
                              'pulse_t0' : -320.,
                              'markov_t' : [0.8, 0.95]
                           }
                    }

def find_files(directory, pattern='*.png', sortby="shuffle"):
    '''
    Recursively finds all files matching the pattern.
    '''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    if sortby == 'auto':
        files = np.sort(files)
    elif sortby == 'shuffle':
        np.random.shuffle(files)

    return files

def roll_dedisperse(d, nchan, fch1, foff, tsamp, dm):
    dd = d.copy()
    for c in xrange(nchan):
        f = fch1 + foff*c
        delayms = 4.15*dm*((fch1/1e3)**-2 - (f/1e3)**-2)
        delaysamp = -int(abs(np.round(delayms/1e3/tsamp)))
        dd[:, c] = np.roll(d[:, c], delaysamp)
    return dd

def get_ts(t_start, nus, DM):
    '''
    Calculate arrival times for all channels
    '''
    return 4.15*DM*(nus**(-2)-nus[-1]**(-2)) + t_start

def get_chans(fch1, foff, nchans, f1=8000., f2=4000.):
    '''
    Calculates and returns all frequency channels between f1 and f2
    that are represented in a filterbank file with specific fch1, foff, and nchans
    '''
    fend = fch1 + foff*nchans
    fs = np.linspace(fch1, fend, nchans)
    start = int((fch1 - f1)/abs(foff))
    end = int((fch1 - f2)/abs(foff))
    #start = min(start, nchans-1)
    start = max(start, 0) # Maybe add warning?
    end = min(end, nchans - 1)
    return start, end, fs[start:end]


def modulate(nus, bp=None, bs=1, t0=0, DM=500, mod=None):
    '''
    Modulate over frequencies, here's just an example, using gaussian envelop 
    try to mimick whatever variations/patterns seen in data
    nu in GHz
    '''
    length = nus.size
    df = nus[1] - nus[0]
    if mod is None:
        p = np.random.random()
        if p > 0.01:
            mod = 'markov'
        else:
            mod = 'sinc'
    if mod == 'markov':
        p1, p2 = FREQ_BAND_TO_INFO[args.band]['markov_t'] #CHANGEBACK
        tscale = np.random.uniform(p1, p2)
        mark = markov(length, tscale)
        #import IPython; IPython.embed()
        mark = ndimage.filters.gaussian_filter1d(mark, sigma=float(length)/100)#CHANGEBACK
    elif mod == 'sinc':
        scint_scale = np.random.uniform(0.005, 0.03)
        domain = np.linspace(-20, 20, int(length*2.2)) * np.abs(nus[-1]-nus[0])/scint_scale
        start = np.random.randint(0, int(length*1.1))
        mark = np.sqrt(np.sinc(domain)**2)[start: start+length]
        return mark
    else:
        mark = np.ones_like(nus)
    fpeak = np.random.uniform(nus[0], nus[-1], size=bs)
    fwidth = np.random.uniform(length/30, length/5, size=bs)*df #CHANGEBACK
    if bp is None:
        broad = np.random.random()
        if broad>0.5:
            return np.exp(-0.5 * (nus[np.newaxis,...] - fpeak[...,np.newaxis]) ** 4 / fwidth ** 4.)*mark
        else:
            return mark[np.newaxis,...]
    else:
        bpmod = np.where(bp>np.mean(bp), 1., bp/np.mean(bp))
        return np.exp(-0.5 * (nus - fpeak) ** 2 / fwidth ** 2.)*bpmod*mark
        
        
def post_process(image, clips=None, pool=(32,1)):
    '''
    Clips, fills NaNs, and performs pooling on image if specified
    '''
    #resize_factor = np.asarray([256./pulse.shape[0], 1])
    #image = ndimage.interpolation.zoom(data+pulse, resize_factor, mode='nearest')
    #orig = ndimage.interpolation.zoom(data, resize_factor, mode='nearest')
    if clips is not None:
        image = np.clip(image, clips[0], clips[1])
    image = np.nan_to_num(image)
    if pool is not None:
        image = measure.block_reduce(image, pool, np.mean)
    return image


def get_pulse(t, nu, ind, OUTDIR, batch_size=1, NT=256, band='L', plot=False):
    '''
    Generates batch of pulses, with each frame having shape len(nu) * NT
    '''
    if ind % 5000 == 0:
        print("{}".format(ind))
    tfs = {256:80., 512:165., 1024:4*80, 2048:8*80} # scaling for NT
    
    freq_band_info = FREQ_BAND_TO_INFO[band]
    #print freq_band_info
    t_0 = np.random.uniform(freq_band_info['pulse_t0'], tfs[NT], size=batch_size) #t_0 is start time in ms
    valid = False
    DM_range = freq_band_info['DM_range']
    DM = np.random.uniform(DM_range[0], DM_range[1], size=batch_size)
    tries = 0
    while not valid:
        seed = int(time.clock()*(ind+1)*1000)%(4294967295)  #2^32-1
        np.random.seed(seed)
        if tries > 5:
            #print "5 tries with DM {}, t0 {}, seed {}, ind {}".format(DM, t_0, seed, ind)
            seed = int(time.clock()*(ind+1)*1000)%(4294967295)  #2^32-1
            np.random.seed(seed)
            t_0 = np.random.uniform(freq_band_info['pulse_t0'], tfs[NT], size=batch_size)
            DM = np.random.uniform(DM_range[0], DM_range[1], size=batch_size)
            tries = 0
        modulation = modulate(nu, t0=t_0, DM=DM, mod=args.mod).squeeze()
        
        # ampwidth = np.random.uniform(0.2, 2) 
        # width0 = np.exp(np.random.uniform(np.log(0.02), np.log(2)))

        #for gbt c band CHANGEBACK
        ampwidth = np.random.uniform(0.02, 0.3) 
        width0 = width0 = np.exp(np.random.uniform(np.log(0.005), np.log(0.3)))
        amp0 = ampwidth / width0
        
        amp = amp0 * modulation[:, np.newaxis] 
        
        width = width0# * np.sqrt(modulation[:, np.newaxis])
        t0_all = get_ts(t_0, nu, DM)
        t_peak = t0_all[np.argmax(modulation)]
        tries += 1
        if t_peak > 0.01 and t_peak > t[0] and t_peak < t[-1]: #if t0_all[np.argmax(modulation)] > 0.01 and ...
            valid = True
            p = np.random.random((2))
            exp = 2.
            if p[0] > 0.5:
                exp = 4.
            outline = ((t - t0_all[:, np.newaxis])/ width)
            scatter_width = np.exp(np.random.uniform(np.log(width0), np.log(width0*5)))
            soutline = ((t - t0_all[:, np.newaxis])/ scatter_width)
            exp_outline = np.exp(-0.5 * outline ** exp )
            scatter_outline = np.where(outline>0, np.exp(-0.5 * soutline ), exp_outline )
            exp_outline = p[1]*scatter_outline + (1-p[1]) * exp_outline
            #zeroed_outline = np.where(np.abs(exp_outline - 1) < 0.001, 0, exp_outline)
            pulse = amp * exp_outline
            #print(np.sum(pulse))
            if np.sum(pulse) < 50:
                valid=False
    if True:
        pulse = pulse[::-1]
    #print pulse.shape

    if False: #!!! hack
        print(np.zeros(pulse.shape[1])[np.newaxis, :].shape, pulse.shape)
        pulse = np.vstack([np.zeros(pulse.shape[1])[np.newaxis, :], pulse])
    if plot:
        plot_img(pulse)
    h_dict={'DM': DM[0], 't0':t_0[0], 'amp':amp0, 'width':width0, 'tpeak':t_peak, 'energy':np.sum(pulse)}
    fitsio.write(OUTDIR+"pulse_"+str(ind)+".fits", pulse, header=h_dict)
    #pulse = fitsio.read(OUTDIR+"pulse_"+str(ind)+".fits")
    #print(pulse.shape)
    
    #pulse2, header = fitsio.read(OUTDIR+"pulse_"+str(ind)+".fits", header=True)
    #print('2222222    ', pulse2.shape)
    #scipy.misc.imsave(OUTDIR+"pulse_"+str(ind)+'.png', measure.block_reduce(pulse, (32,1), np.mean))

def get_pair(data, pulse, clips=[-1,1], batch_size=1, pool=(32,1), plot=False):
    '''
    Return pair of images, one with pulse, one without
    data is the single image
    '''
    #if len(data.shape)==2:
    #    data = data[np.newaxis, ...]
    #batch_size = data.shape[0]
    if data.shape[0]!=pulse.shape[0]:
        times = pulse.shape[0]/data.shape[0]+1
        pulse = measure.block_reduce(pulse, (times,1), np.mean)  #used for 4chan files
        data /= times
    if pool is not None: #for getting the mask
        ppulse = measure.block_reduce(pulse, pool, np.mean)
    else:
        ppulse = pulse
    pmask = ppulse > np.amax(ppulse) * 0.1
    #print('data shape, pulse shape')
    #print(data.shape, pulse.shape)
    mult = 1#4**np.random.random()/2 #CHANGEBACK
    #import IPython; IPython.embed()
    image = post_process(data+pulse*mult, clips=clips, pool=pool)
    orig = post_process(data, clips=clips, pool=pool)
    if plot:
        plot_img(pulse)
        plot_img(image)
#    plot_img(orig)
#    plot_img(post_process(data+(12*pulse), clips=clips, pool=pool))
#    import sys
#    sys.exit()
    
    return image, orig, pmask

def gen_pulse_corpus(FILE, N, OUTDIR, NT=256, band='L', fullband=True):
    '''
    Calls get_pulse in parallel to generate a corpus of N batches 
    of pulses in OUTDIR
    '''
    if band == 'C':
        fullband = False
    fbank = Waterfall(FILE, load_data=False)
    fheader = fbank.header
    print("HEADER INFO:")
    print(fheader)
    if fullband:
        fband_range = [fheader['fch1']+fheader['foff']*fheader['nchans'],fheader['fch1']]
        nu = np.arange(fband_range[0], fband_range[-1], np.abs(fheader['foff'])) * 1.e-3
    else:
        fband_range = FREQ_BAND_TO_INFO[band]['freq_range']
        fc1, fc2, nus = get_chans(fheader['fch1'], fheader['foff'], fheader['nchans'], 
                              f1=fband_range[1], f2=fband_range[0])
        nu = nus[::-1] * 1.e-3 # to GHz
    t = np.arange(0, NT, 1) * fbank.header['tsamp']*1.e3
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    _ = Parallel(n_jobs=8)(delayed(get_pulse)(t, nu, i, OUTDIR, 1, NT, band) for i in range(N))

def _get_sawtooth_f(t):

    rands = np.random.random(5)
    phs0 = rands[0]*2*np.pi
    nu = rands[1]*2*np.pi*50
    f = signal.sawtooth(nu*t+phs0, width=0.5)
    f1 = np.sin(nu*t+phs0)
    freq = (1-rands[2]**4)*f + rands[2]**4 * f1
    #amp = np.sin(2*(rands[3]nu*t+rands[4]*phs0))
    return freq
    
def _add_sawtooth(subdata, nu=None, t=None):
    """simulate sawtooth rfi"""
    nfreq, ntime = subdata.shape
    #print(nfreq, ntime)
    subdata = subdata.copy()
    if nu is None:
        nu = np.arange(nfreq)
    if t is None:
        t = np.arange(ntime)
    npulses = np.random.randint(0, 5)
    widths = nfreq/200 * np.random.random(npulses) + nfreq/500
    amps = 3 * np.random.random(npulses)
    f0s = np.random.choice(nu, size=npulses)
    f1s = nfreq/5 * np.random.random(npulses)
    for i in xrange(npulses):
        nu0_all = f0s[i] + f1s[i]*_get_sawtooth_f(t)
        pulse = amps[i] * np.exp(-0.5 * ( nu[:,np.newaxis] - nu0_all[np.newaxis, :]) ** 6 / widths[i] ** 6.)
    #print(pulse.shape, subdata.shape)
        subdata += pulse
    return subdata


def _process_train(t_0, step, t_stride, pulse_files, subdata, pref, pool, outdir, cleandir=None, fch1=None, savename=None, savemask=False, h=None, snr_cut=2):
    
    t_inst = t_stride*step + t_0
    
    if savename is None:
        savename = pref+'_'+str(t_inst)

    # if add_sawtooth:
    #     subdata = _add_sawtooth(subdata)
    try:
        pulse, header = fitsio.read(np.random.choice(pulse_files), header=True)
    except:
        pulse, header = fitsio.read(np.random.choice(pulse_files), header=True)
    signa, clean, pmask = get_pair(subdata, pulse, clips=None, pool=pool)
    header['AMP_EFF'] = np.amax(signa-clean)
    #import IPython; IPython.embed()
    if fch1 is None:
        fch1 = h['fch1']
    dd = roll_dedisperse((signa-clean).T, signa.shape[0],fch1, h['foff']*pool[0], h['tsamp']*pool[1], header['DM']).T
    noise_dd = roll_dedisperse(clean.T, signa.shape[0], fch1, h['foff']*pool[0], h['tsamp']*pool[1], header['DM']).T
    snr = np.amax(np.sum(dd, axis=0))/np.std(np.sum(noise_dd, axis=0))
    if snr_cut is not None and snr < snr_cut:
        factor = snr_cut/snr
        mult = np.random.uniform(factor, 8*factor)
        signa = clean + mult*(signa - clean)
        snr *= mult
        header['AMP_EFF'] *= mult
        header['ENERGY'] * mult
        header['AMP'] *= mult
    else:
        mult = 1
    header['SNR'] = snr
    #print("saving "+signa_dir+pref+'_'+str(t0+t_inst))
    if savemask:
        np.savez(outdir+savename, frame=signa.astype(np.float16), mask=pmask)
        if cleandir is not None:
            np.savez(cleandir+savename, frame=clean.astype(np.float16))
    else:
        np.save(outdir+savename, signa.astype(np.float16))
        #np.save(outdir+savename+'_dd.npy', (mult*dd+noise_dd).astype(np.float32))
        if cleandir is not None:
            np.save(cleandir+savename, clean.astype(np.float16))
    return savename, header


def gen_train(FILE, pref, pulse_dir, OUTDIR, pool=(32,1), verbose=False, avg_per_sample=True, add_sawtooth=False, dT=256, mini_batch=128, band='L', NT=None):
    '''
    Generate training set for parameter determination
    pool:  pooling sizes. Results will be reduced by (freq, time)
    '''
    pulse_files = find_files(pulse_dir, pattern='*.fits', sortby=None)
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    if pref is None:
        pref = get_pref(FILE)
    fbank = Waterfall(FILE, load_data=False)
    fheader = fbank.header
    freq_band_info = FREQ_BAND_TO_INFO[band]
    #print freq_band_info 
    fband_range = freq_band_info['freq_range']
    fc1, fc2, nus = get_chans(fheader['fch1'], fheader['foff'], fheader['nchans'], 
                              f1=fband_range[1], f2=fband_range[0])
    if NT is None:
        NT = fbank.n_ints_in_file#get_NT(FILE) #839680 #5149696
        print 'nintsinfile', NT
    nt = mini_batch * dT #3072 #20480
    while nt > NT:
        mini_batch /= 2
        nt = mini_batch * dT
    t0 = 0
    clean_dir = OUTDIR + "clean/"
    signa_dir = OUTDIR + "signa/" 
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
        os.makedirs(signa_dir)
    with open(OUTDIR+'/param_labels_orig.csv', 'ab') as labelfile:
        writer = csv.writer(labelfile, delimiter=',')
        writer.writerow(['fname', 'DM', 't0', 'amp', 'amp_eff', 'width', 'energy', 'snr'])
        while t0 + nt < NT:
            print("{}/{}".format(t0, NT))
            fbank.read_data(t_start=t0, t_stop=nt+t0)#, f_start=fband_range[0], f_stop=fband_range[1])#f_start=4000, f_stop=8000)
            data = fbank.data.squeeze()
            data = data[:,300:3031]
            #import IPython; IPython.embed()
            if fbank.header['nbits'] == 8:
                data = data.astype('uint8').astype(np.float32)
            if not avg_per_sample:
                data = normalize(data, 0) # normalize along nt
               
            data = data.T    
            data = data.reshape(data.shape[:-1] + (nt/dT, dT))
            
           
            if avg_per_sample:
                data = normalize(data, 1) # normalize along dT
             
            for step in xrange(nt/dT): #(t_0, step, t_stride, pulse_files, subdata, add_sawtooth, pref, pool, outdir, cleandir=None, savename=None)
                savename, header = _process_train(t0, step, dT,
                                    pulse_files, data[:,step,:], 
                                    pref, pool, signa_dir, cleandir=clean_dir, h=fheader, fch1=fband_range[1])
                writer.writerow([savename,
                    header['DM'], header['T0'], header['AMP'], header['AMP_EFF'], header['WIDTH'], header['ENERGY'], header['SNR']])
            t0 += nt


def gen_train_fromnpy(npy_dir, pref, pulse_dir, OUTDIR, pool=(32,1), verbose=False, avg_per_sample=True, add_sawtooth=False):
    """generate training set for parameter determination"""
    pulse_files = find_files(pulse_dir, pattern='*.fits', sortby=None)
    npy_files = find_files(npy_dir, pattern='*.npy', sortby='auto')
    print(npy_dir)
    
    clean_dir = OUTDIR + "clean/"
    signa_dir = OUTDIR + "signa/" 
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
        os.makedirs(signa_dir)
    with open(OUTDIR+'/param_labels_orig.csv', 'wb') as labelfile:
        writer = csv.writer(labelfile, delimiter=',')
        writer.writerow(['fname', 'DM', 't0', 'amp', 'width', 'energy'])
        for ind, npy_file in enumerate(npy_files):
            if ind % 1000 == 0:
                print("{}/{}".format(ind, len(npy_files)))
            data = np.load(npy_file).squeeze()
            #print(data.shape)
            pref = '.'.join(os.path.basename(npy_file).split('.')[:-1])
            if avg_per_sample:
                mean = np.mean(data,axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                data -= mean
                data /= std
            data = data.T
            step = int(os.path.basename(npy_file).split('.')[0].split('_')[-1])

            savename, header = _process_train(0, step, 1, 
                            pulse_files, data,
                            pref, pool, signa_dir, clean_dir)
            writer.writerow([savename, 
                             header['DM'], header['T0'], header['AMP'], header['WIDTH'], header['ENERGY']])


            
def normalize(data, a):
    ''' 
    Takes in an array data, and normalizes it along axis a
    '''
    mean = np.mean(data,axis = a, keepdims=True)
    std = np.std(data, axis = a, keepdims=True)
    data -= mean
    data /= std
    return data

def get_NT(FILE):
    '''
    Returns number of time integrations in fil FILE; loads one 
    frequency channel (across all time) of file to do this
    '''
    fbank = Waterfall(FILE, load_data=False)
    fch1 = fbank.header['fch1']
    f_off = fbank.header['foff']
    fbank.read_data(f_start = fch1 + f_off, f_stop = fch1) # Read one channel of file to determine n_ints
    n_ints = fbank.data.shape[0]
    return n_ints

# Currently only adjusting axes for L-Band, will fix soon
def plot_img(img):
    print(img.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, aspect='auto')
    plt.xlabel('Time (ms)')
    plt.yticks(np.arange(img.shape[0], 0, -img.shape[0]/10)) 
               #np.arange(2000, 1000, -100))
    #plt.ylabel('Frequency (MHz)')
    plt.ylabel('Frequency Channel #')
    plt.colorbar()
    plt.show()
    

def get_optimal_params(fil):
    wf = Waterfall(fil, load_data=false)
    h = wf.header
    return

def run(FILE, band, pulse_dir, train_dir, pref, genpulse=0, gentrain=True, npy_dir=None, dT=1024, pool=(2,4)):
    #FILE = "/bldata/FRBML/FRB20180301/spliced_blc0506_guppi_58177_24023_048367_G204.17-6.06_0001.0000.fil"
    #FILES = find_files("/data2/SETI/BreakThrough/FRB/clean/", pattern="*.8.4chan.fil", sortby='auto') # point to GBT_L later
    #if not os.path.exists(PULSE_DIR):
    #    os.makedirs(PULSE_DIR)
    if genpulse > 0:
        gen_pulse_corpus(FILE, genpulse, pulse_dir, NT=dT, band=band)
    
    #    pref = os.path.basename(FILE).split('.')[0]
    #    print(FILE)
    if gentrain: # pool=(2,4)
        gen_train(FILE, pref, pulse_dir, train_dir, dT=dT, pool=pool, verbose=False, 
            avg_per_sample=True, add_sawtooth=False,band=band, NT=None)

    if npy_dir is not None:
        gen_train_fromnpy(npy_dir, pref, pulse_dir, train_dir, pool=pool)
    
    #gen_test_all()
def get_pref(fname):
    #bname = os.path.basename(fname)
    bname = '_'.join(fname.split('/')[-2:])
    bname = bname.split('.')[0]
    return bname
    #pref = '_'.join(bnam.split('_')[])
if __name__ == "__main__":
    FILES = find_files(args.fil_dir, pattern='*4chan*.fil', sortby='auto')
    #inds = '_'+str(args.ind)+'/'
    ind = args.ind
    #run(FILES[0], args.band, args.pulse_dir, args.train_dir, args.pref, genpulse=50000, dT=256, pool=(8,1))
    for fil in FILES[3:]:
        run(fil, args.band, args.pulse_dir, args.train_dir, args.pref, genpulse=0, dT=256,pool=(8,1))
    #run(None, args.band, args.pulse_dir, args.train_dir, args.pref, genpulse=0, gentrain=False, npy_dir=args.npy_dir)
