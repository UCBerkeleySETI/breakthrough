import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import pyfits as pf 
from lmfit import minimize, Parameters, fit_report
from lmfit.models import GaussianModel

def gaussian_fit(x, y, title_name):
	mod = GaussianModel()
	pars = mod.guess(y, x=x)
	#pars = mod.make_params(amplitude = -2000, sigma = 1, center = 6562.801)
	out = mod.fit(y, pars, x=x)
	plt.figure()
	plt.plot(x, y)
	plt.plot(x, out.best_fit, 'r-')
	plt.title(title_name)
	print(out.fit_report(min_correl = 0.25))
	print('Center at ' + str(out.best_values['center']) + ' Angstrom')
	plt.show()

if __name__ == "__main__":

	wave = pf.open('apf_wave.fits copy')
	wave_im = wave[0].data

	apf_red = pf.open('ramp.194.fits copy')
	image_red = apf_red[0].data

	wave_h_alpha = wave_im[53,1942-500:1942+500]
	red_h_alpha = image_red[53,1942-500:1942+500]

	red_h_alpha_flip = red_h_alpha*(-1)
	image_red_flip = image_red*(-1)
	left_med = np.median(red_h_alpha[0:50])
	right_med = np.median(red_h_alpha[-50:])
	median = (right_med + left_med)/2

	red_h_alpha_flip = red_h_alpha_flip + median
	red_h_alpha = red_h_alpha - median



	gaussian_fit(wave_h_alpha,red_h_alpha, "Guassian")
