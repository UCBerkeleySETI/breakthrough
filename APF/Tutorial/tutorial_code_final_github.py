import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import pyfits as pf 
from lmfit import minimize, Parameters, fit_report
from lmfit.models import GaussianModel

############# Functions to perform #############

def plot_spec(data_file):
	plt.figure()
	plt.imshow(data_file, cmap = 'gray', vmin = np.median(data_file), vmax = np.median(data_file) *1.1, origin = 'lower')
	plt.title('APF Sample Spectrum')
	plt.show()

def plot_reduced_spec(data_file):
	plt.figure()
	plt.imshow(data_file, cmap = 'gray', vmin = np.median(data_file), vmax = np.median(data_file) *2, aspect = 'auto', origin = 'lower')
	plt.title('APF Reduced Spectrum')
	plt.show()

def plot_telluric(data_file, bias):
	patch = data_file[1683:1688, 2200:2800]
	telluric = np.sum(patch, axis = 0)
	telluric_adj = telluric - (5*bias)
	plt.figure()
	plt.plot(telluric_adj)
	plt.title('Raw Telluric Absorption lines - [1683:1688, 2200:2800]')
	plt.show()

def plot_telluric_with_2D(data_file, bias):
	patch = data_file[1683:1688, 2200:2800]
	telluric = np.sum(patch, axis = 0)
	telluric_adj = telluric - (5*bias)
	plt.figure()
	plt.subplot(2,1,1)
	plt.imshow(patch, cmap = 'gray', aspect = 'auto', vmin = np.median(data_file), vmax = np.median(data_file) *1.2, origin = 'lower')
	plt.title('Telluric features (2D)')
	plt.subplot(2,1,2)
	plt.plot(telluric_adj)
	plt.title('Telluric features (1D)')
	plt.show()

def plot_h_alpha(data_file, bias):
	plt.figure()
	h_alpha_patch = data_file[1491:1506,1500:2500]
	h_alpha_order = np.sum(h_alpha_patch, axis = 0)
	plt.plot(h_alpha_order - bias*15)
	plt.title('Raw H-alpha absoprtion line - [1491:1506,1500:2500]')
	plt.show()

def plot_reduced_h_alpha(data_file):
	plt.plot(data_file[53])
	plt.title('Reduced H-Alpha absoprtion line - [53]')
	plt.show()

def plot_Na_D(data_file, bias):
	Na_D_patch = data_file[1333:1348, 1200:2200]
	Na_D_patch_1D = np.sum(Na_D_patch, axis = 0) - bias*15
	plt.figure()
	plt.plot(Na_D_patch_1D)
	plt.title('Raw Sodium D absorption lines - [1333:1348, 1200:2200]')
	plt.show()

def cut_n_zoom(x1,x2,y1,y2):
	try:
		plt.figure(figsize=(10,10))
		plt.imshow(image[x1:x2, y1:y2], cmap = 'gray', aspect = 'auto', vmin = np.median(image), vmax = np.median(image) *1.2, origin = 'lower')
		plt.show()
	except:
		print('Bad coordinates Chief. Try again.')

def cosmic_ray_spot(patch, name, bias, num_of_orders):
	plt.figure()
	patch_1D = np.sum(patch, axis = 0) - bias*(num_of_orders)
	plt.plot(patch_1D, color = 'b')
	plt.title('Cosmic rays along Raw ' + name)
	for i in range(5, patch_1D.size - 5):
		if (patch_1D[i]>patch_1D[i-1]) and (patch_1D[i]>patch_1D[i+1]) and (patch_1D[i]>(bias*1.25)):
			half_max = ((patch_1D[i]) + (patch_1D[i+5] + patch_1D[i-5])/2)/2
			left_side = np.where(patch_1D[:i] <= half_max)
			left_mark = left_side[0][-1]
			right_side = np.where(patch_1D[i:] <= half_max)
			right_mark = right_side[0][0] + i
			peak_x = right_mark - ((right_mark - left_mark)/2)
			plt.axvline(x=peak_x, ymin = np.min(patch_1D) - 1000, color = 'r', linestyle = '-.')
	plt.show()

def gaussian_fit(x, y, title_name):
	mod = GaussianModel()
	pars = mod.guess(y, x=x)
	out = mod.fit(y, pars, x=x)
	plt.figure()
	plt.plot(x, y)
	plt.plot(x, out.best_fit, 'r-')
	plt.title(title_name)
	print(out.fit_report(min_correl = 0.25))
	print('Center at ' + str(out.best_values['center']) + ' Angstrom')
	plt.show()



#################    Runs if name is main    ################

if __name__ == "__main__":
	#	Load the data
	apf_file = pf.open('ucb-amp194.fits')
	image = np.fliplr(np.rot90(apf_file[0].data))
	# ^ flips over y-axis and rotates the original array by 90 degrees
	
	#	Load the reduced data
	apf_red = pf.open('ramp.194.fits')
	image_red = apf_red[0].data

	#	Calculate the bias
	bias = np.median(image[-30:])

	#	Show the plots
	plot_spec(image)
	plot_reduced_spec(image_red)
	plot_telluric(image, bias)
	plot_telluric_with_2D(image, bias)
	plot_h_alpha(image, bias)
	plot_reduced_h_alpha(image_red)
	plot_Na_D(image, bias)

	#	Patches of data with certain absoprtion lines
	h_alpha_patch = image[1491:1506,1500:2500]
	Na_D_patch = image[1333:1348, 1200:2200]
	telluric_patch = image[1683:1688, 2200:2800]

	#	Cosmic Ray Spotting Plots
	cosmic_ray_spot(h_alpha_patch, 'H-Alpha', bias, 15)
	cosmic_ray_spot(Na_D_patch, 'Sodium D', bias, 15)
	cosmic_ray_spot(telluric_patch, 'Telluric lines', bias, 5)

	#	Gaussian Model of H-alpha line
	##	Load wavelength solution
	wave = pf.open('apf_wave.fits')
	wave_im = wave[0].data

	x = wave_im[53,1500:2500]
	y = np.sum(h_alpha_patch, axis = 0) - bias*15
	x_reduced = wave_im[53,:4000]
	y_reduced = image_red[53,:4000]
	##	Plot H-alpha data vs. wavelength solution and fit to Gaussian model
	print("Reduced H-Alpha")
	gaussian_fit(x_reduced, y_reduced, "Reduced H-Alpha")
	print(" ")
	print("Raw H-Alpha")
	gaussian_fit(x, y, "Raw H-Alpha")


################################################################










