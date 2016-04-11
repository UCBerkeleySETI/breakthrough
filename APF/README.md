This directory contains sample FITS format files from the Automated Planet
Finder, APF. A brief introduction to the Automated Planet Finder is at
https://www.ucolick.org/public/telescopes/apf.html and for a technical
introduction, see http://arxiv.org/pdf/1402.6684v1.pdf

The FITS files contained in this directory consist of APF observations of KIC
8462852, also known as "Tabby’s Star". FITS files can be viewed with the free
software ds9, which can be downloaded from http://ds9.si.edu

If you are unfamiliar with astronomical optical spectroscopy, you may wish to
start by trying to display the raw spectrum of Tabby's Star, as read out from
the CCD detector at the telescope, ucb-amp193.fits. Adjust the contrast in ds9
until you see the light from the spectrograph, and can reproduce the image of
this spectrum as shown on the Breakthrough Listen webpage.

This FITS file is a 2-dimensional spectrum, where light of different wavelengths
(or equivalently, frequencies) falls on the CCD detector, creating a record of
the brightness of the star as a function of wavelength. A simple spectrum may
have only one order, but this is an example of an echelle spectrum. An echelle
spectrum is chopped up into dozens of spectral orders, each containing a
different range of wavelengths, so that different pixel positions in the image
correspond to different wavelengths of light. The wavelength decreases from left
to right and top to bottom, similar to the words on a page.

For a technical description of astronomical spectroscopy, read
http://arxiv.org/pdf/1010.5270v2.pdf

From the 2D spectrum, we can extract a 1D spectrum, which contains a measurement
of the brightness as a function of wavelength. For a single spectral order, this
extracted spectrum would be a one-dimensional array. Since APF splits up the
starlight into multiple orders, the extracted spectrum is a 2D array, where the
zeroth dimension increases in wavelength along an order, and the first dimension
steps through the spectral orders.

This directory contains the following extracted spectra:

apf_wav.fits - reduced 1D spectrum of Tabby's star, extracted from ucb-amp193.fits

ramp.193.fits - corresponding wavelength calibration solution

Hydrogen alpha is in the 53rd order. The magnesium B triplet is in the 34th
order (0-indexed). Try reading apf_wav.fits into Python using
http://docs.astropy.org/en/stable/io/fits/ and see if you can reproduce the plot
of the region around H-alpha shown on the Breakthrough Listen webpage.