from __future__ import print_function
from os.path import split as pathsplit, join as pathjoin, splitext
import sys
sys.path.append('../../GBT/filterbank_tools/')

import numpy as np
import pylab as pl

from filterbank import Filterbank as FB, db

#from astropy.io import fits

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) not in (2,3):
        print("usage: %s input.fil [output.fits]")

    infilf = argv[1]
    outf = argv[2] if len(argv) == 3 else splitext(infilf)[0]+'.fits'

    fbin = FB(infilf)
    f, data = fbin.grab_data()
    data = db(data)

    # dump to fits for fun
    #fits.writeto(outf,data,clobber=True)
    #infits = fits.open(outf,memmap=True)
    #data = infits[0].data
    
    print(data.shape)
    print(f.shape)
    pl.imshow(data)
    f_step = int(f.shape[0]/25)
    t_step = int(data.shape[0]/3)
    xt =  (f[::f_step]*10).astype(int)/10.0
    pl.xticks(np.arange(len(xt)),xt,rotation=90)
    pl.yticks(np.arange(data.shape[0])[::t_step])
    
    pl.show()

