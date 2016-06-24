import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import pyfits as pf 
from lmfit import minimize, Parameters, fit_report
from lmfit.models import SkewedGaussianModel

def skewed_gaussian_fit(x, y, title_name):
	mod = SkewedGaussianModel()
	params = mod.make_params(amplitude=-2000, center=6562.801, sigma=1, gamma=0)
	result = mod.fit(y, params, x=x)
	print(result.fit_report())
	#out = mod.fit(y, pars, x=x)
	plt.figure()
	plt.plot(x, y)
	plt.plot(x, result.best_fit)
	plt.title(title_name)
	plt.show()

def gaussian(params, x):
	""" Gaussian model """
	A     = params['A'].value
	mu    = params['mu'].value
	sigma = params['sigma'].value
	return A*(np.exp(-(((x-mu)**2)/(2.*(sigma**2)))))

def residual(params, x, data):
    """ Residual function, returns model - data """
    model = gaussian(params, x)
    return data - model

def fit_gaussian(x_data, y_data):
	""" Fit a gaussian to data """
	# Setup fit
	# Start values need to be ballpark correct!
	params = Parameters()
	params.add('A', value=-2000)
	params.add('mu', value=6562.801)
	params.add('sigma', value=1)

	out = minimize(residual, params, args=(x_data, y_data))

	print(fit_report(out.params))

	plt.plot(x_data, y_data)
	plt.plot(x_data, gaussian(out.params, x_data))
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
	right_med = np.median(red_h_alpha[0:50])
	left_med = np.median(red_h_alpha[-50:])
	median = (right_med + left_med)/2

	red_h_alpha_flip = red_h_alpha_flip + median
	red_h_alpha = red_h_alpha - median

	plt.figure()
	plt.plot(image_red_flip[53])
	plt.figure()
	plt.plot(wave_h_alpha,red_h_alpha_flip)
	plt.figure()
	plt.plot(red_h_alpha_flip)
	plt.figure()

	fit_gaussian(wave_h_alpha,red_h_alpha)
	plt.figure()

	skewed_gaussian_fit(wave_h_alpha,red_h_alpha, "Guassian Flip")








