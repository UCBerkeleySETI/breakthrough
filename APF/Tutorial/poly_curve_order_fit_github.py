import matplotlib.pyplot as plt
import numpy as np 
import pyfits as pf 


def get_coeffs(file_name):
	#	open order coefficients and read values
	coeff_array = np.zeros((79,5))
	with open(file_name, "r") as text:
		lines = text.read().splitlines()
		for i in range(len(lines)):
			a0 = float(lines[i][6:13].strip())
			a1 = float(lines[i][17:26].strip())
			a2 = float(lines[i][27:39].strip())
			a3 = float(lines[i][40:52].strip())
			a4 = float(lines[i][54:].strip())
			coeffs = np.array([a0,a1,a2,a3,a4])
			coeff_array[i] += coeffs
	return coeff_array

def plot_with_polys(raw_array, coeff_array):
	plt.imshow(raw_array, cmap = "gray", origin = "lower", 
		aspect = "auto", vmin = np.median(raw_array), 
		vmax = np.median(raw_array) *1.1)
	plt.title("Raw image with polynomial functions overplotted")
	x = np.arange(0,4608)
	for i in range(coeff_array[:,0].size):
		a0 = coeff_array[i,0]
		a1 = coeff_array[i,1]
		a2 = coeff_array[i,2]
		a3 = coeff_array[i,3]
		a4 = coeff_array[i,4]
		plt.plot(x, a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

def reduce_raw_data(image, coeff_array, bias):
	x = np.arange(0, 4608)
	y_values = np.zeros((79,4608))
	reduced_image = np.zeros((79,4608))
	for i in range(coeff_array[:,0].size):
		a0 = coeff_array[i,0]
		a1 = coeff_array[i,1]
		a2 = coeff_array[i,2]
		a3 = coeff_array[i,3]
		a4 = coeff_array[i,4]
		for j in range(x.size):
			y = a0 + a1*x[j] + a2*x[j]**2 + a3*x[j]**3 + a4*x[j]**4
			y_values[i,j] = y
			y = int(round(y))
			reduced_image[i,j] = int(np.sum(image[y-1:y+2,j], 
				axis = 0)-3*bias)
	return 	reduced_image, y_values



if __name__ == "__main__":

	#	opening the APF Data and plotting the data as a 2D array
	apf_file = pf.open('ucb-amp194.fits')

	image = np.fliplr(np.rot90(apf_file[0].data))
	# ^ flips and rotates the original array by 90 degrees 
	#	to make it correct (as far as website photo goes)
	header = apf_file[0].header

	#	Calculate the bias
	bias = np.median(image[-30:])

	#	get reduced image
	apf_reduced = pf.open('ramp.194.fits')
	header_red = apf_reduced[0].header
	image_red = apf_reduced[0].data


	#	creates an array of polynomial coefficients for each order
	coeff_array = get_coeffs("order_coefficients.txt")
	
	#	plots reduced data array with overlaying polynomial functions
	plt.figure(figsize=(12,8))
	plot_with_polys(image, coeff_array)


	#	extracting each order and creating a reduced image to plot
	reduced_image, y_values = reduce_raw_data(image, coeff_array, bias)
	plt.figure(figsize=(12,8))
	plt.subplot(2, 1, 1)
	plt.imshow(reduced_image, cmap = "gray", origin = "lower", 
		aspect = "auto", vmin = np.median(reduced_image), 
		vmax = np.median(reduced_image) *1.1)
	plt.title("Reduced Image through Polyfit Technique")
	plt.subplot(2, 1, 2)
	plt.title("Reduced Image File")
	plt.imshow(image_red, cmap = "gray", origin = "lower", 
		aspect = "auto", vmin = np.median(image_red), 
		vmax = np.median(image_red) *1.1)
	plt.figure(figsize=(12,8))
	plt.subplot(2, 1, 1)
	plt.plot(reduced_image[53])
	plt.title("Reduced Image (Polyfit) H-alpha")
	plt.subplot(2, 1, 2)
	plt.plot(image_red[53])
	plt.title("Reduced Image File H-alpha")
	plt.show()






