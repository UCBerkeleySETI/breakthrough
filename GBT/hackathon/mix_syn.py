from __future__ import print_function
import sys
sys.path.append('breakthrough/GBT/filterbank_tools/')

import numpy as np
from filterbank import Filterbank as FB, db

extrema = lambda a: (a.min(),a.max())
scalef  = lambda a,ex: (a-ex[0])/(ex[1]-ex[0])
iscalef = lambda a,ex: ex[0]+(a*(ex[1]-ex[0]))

if __name__ == '__main__':   
    infilf = '../data/blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0002.fil'
    synf = '../data/test_data'
    outf = 'synandskysig.fits'
    mixcoef = 0.5
    fbin = FB(infilf)
    f, data = fbin.grab_data()
    data = db(data)
    imgshape = [32,1] #plot_data.shape
    datext = extrema(data)
    with open(synf,'rb') as fid:
        syndata = np.fromfile(fid, count=np.prod(imgshape), dtype='<f4')        
    # scale syn data into range of data, mix
    syndata = iscalef(scalef(syndata,extrema(syndata)),datext)
    data = (1.0-mixcoef)*data + mixcoef*syndata.reshape(imgshape)
    
    fits.writeto(outf,data)
