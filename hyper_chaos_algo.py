# standard libraries
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import skimage.measure
from time import time
#user defined functions
import helper
import histogram
import correlation_coeff


def less_10e14neg(x):
	while(x < math.pow(10,-14)):
		x = x * 10
	return x


def logistic_map_sequences(u, x0, shape):
	w, h = shape
	n = w * h
	x = []
	
	x1 = x0
	
	x2 = 0
	# print(x1)
	N0 = 1000
	for i in range(n+N0):
		try:
			x2 = u * x1 * (1 - x1)
			x2 = abs(x2) % 10000000000000000000000000000000
			x2 = less_10e14neg(x2)
			x.append(x1)

			x1 = x2
		except Exception as e:
			print(e)
			break

	
	x = np.array(x[N0:])
	
	
	# print(x)

	return x


def sine_map(x0, y0, a, b, shape):
	x = []
	y = []
	x1 = x0
	y1 = y0
	x2 = 0
	y2 = 0
	N0 = 1000
	w, h = shape
	n = w * h

	for i in range(n + N0):
		try:
			x2 = math.sin((math.pi * a * math.pow(x1, 2)) - (b * y1))
			y2 = math.sin(1 - (math.pi * b * x1))
			x.append(x1)
			y.append(y1)
			y1 = y2
			x1 = x2
		except Exception as e:
			print(e)
			break
	return np.array(x[N0:]), np.array(y[N0:])

def construction_of_tables(shape, x0, u_x, y0, u_y):
	
	w,h = shape
	power = math.pow(10, 7)
	
	
	# print("inside construction_of_tables:")
	# print("x0", x0)
	# print("u_x", u_x)
	# print("y0", y0)
	# print("u_y", u_y)

	# x = logistic_map_sequences(u_x, x0, shape)
	# y = logistic_map_sequences(u_y,y0, shape)
	# print("logistic map")
	# print(x)
	# print(y)
	x, y = sine_map(x0, y0, u_x, u_y, shape)	
	# print("sine_map")
	# print(x)
	# print(y)
	x = (np.mod(np.floor(x*power),w)+1).reshape(shape)
	y = (np.mod(np.floor(y*power),h)+1).reshape(shape)

	# print(x)

	CT = np.zeros(shape)
	XT = np.zeros(shape)
	YT = np.zeros(shape)
	
	for i in range(w):
		for j in range(h):
			
			if abs(x[i,j]-i) < w/4 or abs(y[i,j]-i) < h/4:
				CT[i,j] = 1
			else:
				CT[i,j] = 0

			if abs(x[i,j]-i) < w/4:
				XT[i,j] = ((x[i,j] + w/4)%w) 

			else:
				XT[i,j] = x[i,j]

			if abs(y[i,j]-i) < h/4:
				YT[i,j] = ((y[i,j] + h/4)%h) 
			else:
				YT[i,j] =y[i,j]
	
	return CT.astype(int), XT.astype(int), YT.astype(int)


def scramble(mat, x0, u_x, y0, u_y):

	n, m = mat.shape
	I1, I2 = np.split(mat, 2, axis=0)

	# print(I1.shape)
	# print(u, y0)

	# print("inside scramble:")
	# print("x0", x0)
	# print("u_x", u_x)
	# print("y0", y0)
	# print("u_y", u_y)
	

	CT, XT, YT = construction_of_tables(I1.shape, x0, u_x, y0, u_y)

	for j in range(m):
		for i in range(int(n/2)):

			if CT[i,j] == 0:
				I1[i,j], I1[XT[i,j]-1,YT[i,j]-1] = helper.swap(I1[i,j], I1[XT[i,j]-1,YT[i,j]-1])
			else:
				I1[i,j], I2[XT[i,j]-1,YT[i,j]-1] = helper.swap(I1[i,j], I2[XT[i,j]-1,YT[i,j]-1])
				
	for i in range(int(n/2)):
		for j in range(m):

			if CT[i,j] == 0:
				I2[i,j], I2[XT[i,j]-1,YT[i,j]-1] = helper.swap(I2[i,j], I2[XT[i,j]-1,YT[i,j]-1])
			else:
				I2[i,j], I1[XT[i,j]-1,YT[i,j]-1] = helper.swap(I2[i,j], I1[XT[i,j]-1,YT[i,j]-1])
				
	# helper.display_image(np.concatenate((I1,I2), axis = 0))
	return np.concatenate((I1,I2), axis = 0), x0, u_x, y0, u_y					


def create_key_stream(x0, u_x, y0, u_y, shape, power):

	# print("inside create_key_stream:")
	# print("z0", z0)
	# print("u_z", u_z)

	k, unused = sine_map(x0, u_x, y0, u_y, shape)
	k = np.mod(np.floor(k * power), 256).astype(int)

	return k


def diffuse(mat, x0, u_x, y0, u_y):
	# print(mat)
	shape, p = helper.initial_processing(mat)
	w, h = shape
	power = math.pow(10, 28)
	n = w * h


	# print("inside diffuse:")
	# print("z0", z0)
	# print("u_z", u_z)
	
	k = create_key_stream(x0, u_x, y0, u_y, shape, power).reshape(p.shape)

	s0 = int((np.sum(p) - p[0,0])% 256)
	# print(k[ 0,0], p[0, 0], s0)

	p[0,0] = s0 ^ p[0, 0] ^ k[0, 0]
	print(k.shape)
	for i in range(2, n ):

		kt1 = math.floor(((p[ 0, i-2] + k[ 0, i-1]) % 256) / 256 * (i - 1)) + 1
		kt2 = math.floor(((p[ 0, i-2] + k[ 0, i-1]) % 256) / 255 * (n - i - 1)) + i + 1
		
		p[0,i - 1] = p[0, i - 1] ^ k[0, i - 1] ^ p[0, kt1 - 1] ^ p[0, kt2- 1]

	kt1 = math.floor(((p[0, n-2] + k[0, n -1]) % 256) / 256 * (n - 1)) + 1
	p[0, n - 1] = p[0, n - 1] ^ k[0, n - 1] ^ p[0, kt1 - 1]

	print("diffusion done")

	return p.reshape(shape), x0, u_x, y0, u_y
 

def encrypt(mat, x0, u_x, y0, u_y):
	
	scrambled_mat, x0, u_x, y0, u_y = scramble(mat, x0, u_x, y0, u_y)
	# helper.display_image(scrambled_mat)
	# histogram.plot_histogram(scrambled_mat)
	"""TEST"""
	# diffuse(scrambled_mat)
	# return scrambled_mat
	# helper.display_image(scrambled_mat)
	diffuse_mat, x0, u_x, y0, u_y = diffuse(scrambled_mat, x0, u_x, y0, u_y)

	
	# return scrambled_mat, u_z, z0, u_x, x0, u_y, y0
	return diffuse_mat, u_x, x0, u_y, y0

def decrypt(mat, u_x, x0, u_y, y0):
	

	shape,p = helper.initial_processing(mat)
	n ,m = shape
	power = math.pow(10,28)
	shape_x = (int(n/2), m)

	# print(shape_x)
	CT, XT, YT = construction_of_tables(shape_x, x0, u_x, y0, u_y)
	k = create_key_stream(x0, u_x, y0, u_y, shape, power).reshape(p.shape)

	L = n * m

	tn = 1
	while tn >= 1:

				
		kt1 = math.floor(((p[0, L-2] + k[0, L -1]) % 256) / 256 * (L - 1)) + 1
		p[0, L - 1] = p[0, L - 1] ^ k[0, L - 1] ^ p[0, kt1 - 1]

		for i in range(L-1, 1, -1 ):

			kt1 = math.floor(((p[ 0, i-2] + k[ 0, i-1]) % 256) / 256 * (i - 1)) + 1
			kt2 = math.floor(((p[ 0, i-2] + k[ 0, i-1]) % 256) / 255 * (L - i - 1)) + i + 1
			
			p[0,i - 1] = p[0, i - 1] ^ k[0, i - 1] ^ p[0, kt1 - 1] ^ p[0, kt2- 1]

		s0 = int((np.sum(p) - p[0,0])% 256)
		# print(k[ 0,0], p[0, 0], s0)

		p[0,0] = s0 ^ p[0, 0] ^ k[0, 0]

		img_mat = p.reshape(shape)
		# print(img_mat)
		I1, I2 = np.split(img_mat, 2, axis = 0)
					
		for i in range(int(n/2)-1, -1, -1):
			for j in range(m-1, -1, -1):

				if CT[i,j] == 0:
					I2[i,j], I2[XT[i,j]-1,YT[i,j]-1] = helper.swap(I2[i,j], I2[XT[i,j]-1,YT[i,j]-1])
				else:
					I2[i,j], I1[XT[i,j]-1,YT[i,j]-1] = helper.swap(I2[i,j], I1[XT[i,j]-1,YT[i,j]-1])

		for j in range(m-1, -1, -1):
			for i in range(int(n/2)-1, -1, -1):

				if CT[i,j] == 0:
					 I1[i,j], I1[XT[i,j]-1,YT[i,j]-1] = helper.swap(I1[i,j], I1[XT[i,j]-1,YT[i,j]-1])
				else:
					I1[i,j], I2[XT[i,j]-1,YT[i,j]-1] = helper.swap(I1[i,j], I2[XT[i,j]-1,YT[i,j]-1])

		p = np.concatenate((I1, I2), axis = 0)
		# print(tn)
		tn = tn -1 

	return p


"""TEST"""
# mat = np.arange(64).reshape((8,8))
# I_enc, u_z, z0, u_x, x0, u_y, y0 = encrypt(mat)
# print(I_enc)
# I_dec = decrypt(I_enc, u_z, z0, u_x, x0, u_y, y0)
# print(I_dec)



"""Main"""

""" paths to image files"""
# path = "./lena.png"
# path = './sample1.gif'
# path = './data/frame1.jpg'
# path = './barbara.png'
# path = './baboon.png'
# path = "./airplane.png"
path = "./peppers.png"


# Read plain image 
mat = helper.img_to_mat(path)
helper.display_image(mat)
# histogram.plot_histogram(mat)


# print(mat)
# print(type(mat[0,0]))


w,h = mat.shape
# rand_pix_vert = helper.Rand(0, ((w-2)*(h-1)), 3000)
# rand_pix_hor = helper.Rand(0, ((w-2)*(h-1)), 3000)
# rand_pix_diag = helper.Rand(0, ((w-2)*(h-2)), 3000)

# I_enc1, u_x, x0, u_y, y0 = encrypt(mat1, x0, u_x, y0, u_y)

"""correlation coefficients"""
# a,b = helper.vertical_pairings(mat, rand_pix_vert)
# mat_cor_ver = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.horizontal_pairings(mat, rand_pix_hor) 
# mat_cor_hor =  correlation_coeff.correlation_coeff(a, b)
# a, b = helper.diagonal_pairings(mat, rand_pix_diag)
# mat_cor_diag = correlation_coeff.correlation_coeff(a, b)

# a,b = helper.vertical_pairings(mat)
# mat_cor_ver = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.horizontal_pairings(mat) 
# mat_cor_hor =  correlation_coeff.correlation_coeff(a, b)
# a, b = helper.diagonal_pairings(mat)
# mat_cor_diag = correlation_coeff.correlation_coeff(a, b)

# a,b = helper.vertical_pairings1(mat, rand_pix_vert)
# fig = plt.figure(figsize=(10,7))
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(a, b, 'b.', linewidth = 1)
# ax1.set_ylim(0, 300)
# ax1.set_xlim(0, 300)
# ax1.legend(loc='best')
# plt.show()

# a, b = helper.horizontal_pairings1(mat, rand_pix_hor) 
# fig = plt.figure(figsize=(10,7))
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(a, b, 'b.', linewidth = 1)
# ax1.set_ylim(0, 300)
# ax1.set_xlim(0, 300)
# ax1.legend(loc='best')
# plt.show()

# a, b = helper.diagonal_pairings1(mat, rand_pix_diag)

# fig = plt.figure(figsize=(10,7))
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(a, b, 'b.', linewidth = 1)
# ax1.set_ylim(0, 300)
# ax1.set_xlim(0, 300)
# ax1.legend(loc='best')
# plt.show()


# mat1 = np.copy(mat)
x0, u_x, y0, u_y  = 0.1, 2.6, 0.48480196431, 0.95
# t0 = time()
I_enc, u_x, x0, u_y, y0 = encrypt(mat, x0, u_x, y0, u_y)
# t1 = time()
# print("enctime", (t1-t0)/(6.328571428571428))


# a, b = helper.vertical_pairings(I_enc, rand_pix_vert)
# enc_cor_ver = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.horizontal_pairings(I_enc, rand_pix_hor)
# enc_cor_hor = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.diagonal_pairings(I_enc, rand_pix_diag)
# enc_cor_diag = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.vertical_pairings1(I_enc, rand_pix_vert)
# fig = plt.figure(figsize=(10,7))
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(a, b, 'b.', linewidth = 1)
# ax1.set_ylim(0, 300)
# ax1.set_xlim(0, 300)
# ax1.legend(loc='best')
# plt.show()

# a, b = helper.horizontal_pairings1(I_enc, rand_pix_hor)
# fig = plt.figure(figsize=(10,7))
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(a, b, 'b.', linewidth = 1)
# ax1.set_ylim(0, 300)
# ax1.set_xlim(0, 300)
# ax1.legend(loc='best')
# plt.show()

# a, b = helper.diagonal_pairings1(I_enc, rand_pix_diag)
# fig = plt.figure(figsize=(10,7))
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(a, b, 'b.', linewidth = 1)
# ax1.set_ylim(0, 300)
# ax1.set_xlim(0, 300)
# ax1.legend(loc='best')
# plt.show()



# a, b = helper.vertical_pairings(I_enc)
# enc_cor_ver = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.horizontal_pairings(I_enc)
# enc_cor_hor = correlation_coeff.correlation_coeff(a, b)
# a, b = helper.diagonal_pairings(I_enc)
# enc_cor_diag = correlation_coeff.correlation_coeff(a, b)

# print("horizontal ", mat_cor_hor, enc_cor_hor)
# print("vertical ", mat_cor_ver, enc_cor_ver)
# print("diagonal ", mat_cor_diag, enc_cor_diag)



"""Resistance to differential attacks"""
# print(path)
# print('npcr', helper.npcr(I_enc, I_enc1))
# print('uaci', helper.uaci(I_enc, I_enc1))


"""Information entropy analysis"""
# entropy_img = skimage.measure.shannon_entropy(np.ravel(mat))
# entropy_enc = skimage.measure.shannon_entropy(np.ravel(I_enc))
# print("original: ", entropy_img)
# print("enc: ", entropy_enc)


# helper.display_image(I_enc)
# histogram.plot_histogram(I_enc)

# z01 = z0 + (10** (-9))

# I_enc1, u_z, z01, u_x, x0, u_y, y0 = encrypt(mat, x0, u_x, y0, u_y, z01, u_z)
helper.display_image(I_enc)
# helper.display_image(np.absolute(I_enc1-I_enc))
	
I_dec = decrypt(I_enc, u_x, x0, u_y, y0)
# t2 = time()
# print("dec_time", t2-t1)


"""Quality of encryption"""
# mse = helper.mse(mat, I_dec)
# print("mse:", mse)
# print("psnr", helper.psnr(mse))


"""key sensitivity analysis"""
helper.display_image(I_dec)
# I_dec_wrong = decrypt(I_enc, u_z, z01, u_x, x0, u_y, y0)
# helper.display_image(I_dec_wrong)
# I_dec1 = decrypt(I_enc1, u_z, z01, u_x, x0, u_y, y0)
# helper.display_image(I_dec1)
# I_dec1_wrong = decrypt(I_enc1, u_z, z0, u_x, x0, u_y, y0)
# helper.display_image(I_dec1_wrong)