import numpy as np 
from PIL import Image, ImageOps
import cv2
import secrets
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import os
import random
import statistics


def vertical_pairings1(mat, rand_pix):
	"""vertical pairings of elements with their adjacent elements"""
	w, h = mat.shape
	x = mat[:-1,:]
	y = mat[1:, :]

	x_cor_list = []
	y_cor_list = []
	for i in range(len(rand_pix)):
		pix = rand_pix[i]
		row = (pix // w)
		col = pix - (row*512)
		x_cor_list.append(x[row, col])
		y_cor_list.append(y[row, col])

	return x_cor_list, y_cor_list


def vertical_pairings(mat):
	"""vertical pairings of elements with their adjacent elements"""
	w, h = mat.shape
	x = mat[:-1,:]
	y = mat[1:, :]

	x_cor_list = []
	y_cor_list = []
	for i in range(w-1):
		for j in range(h):
			x_cor_list.append(x[i, j])
			y_cor_list.append(y[i, j])

	return x_cor_list, y_cor_list


def horizontal_pairings1(mat, rand_pix):
	""" horizontal pairings of elements with their adjacent elements"""
	w, h = mat.shape
	x = mat[:,:-1]
	y = mat[:, 1:]
	
	x_cor_list = []
	y_cor_list = []
	for i in range(len(rand_pix)):
		pix = rand_pix[i]
		row = (pix // (h-1))
		col = pix - (row*511)
		x_cor_list.append(x[row, col])
		y_cor_list.append(y[row, col])

	return x_cor_list, y_cor_list


def horizontal_pairings(mat):
	""" horizontal pairings of elements with their adjacent elements"""
	w, h = mat.shape
	x = mat[:,:-1]
	y = mat[:, 1:]
	
	x_cor_list = []
	y_cor_list = []
	for i in range(w):
		for j in range(h-1):
			x_cor_list.append(x[i, j])
			y_cor_list.append(y[i, j])

	return x_cor_list, y_cor_list


def diagonal_pairings1(mat, rand_pix):
	""" diagonal pairings of elements with their adjacent elements"""
	w, h = mat.shape
	x = mat[:-1,:-1]
	y = mat[1:, 1:]
	x_cor_list = []
	y_cor_list = []
	for i in range(len(rand_pix)):
		pix = rand_pix[i]
		row = (pix // (h-1))
		col = pix - (row*511)
		x_cor_list.append(x[row, col])
		y_cor_list.append(y[row, col])

	return x_cor_list, y_cor_list


def diagonal_pairings(mat):
	""" diagonal pairings of elements with their adjacent elements"""
	w, h = mat.shape
	x = mat[:-1,:-1]
	y = mat[1:, 1:]
	x_cor_list = []
	y_cor_list = []
	for i in range(w-1):
		for j in range(h-1):
			x_cor_list.append(x[i, j])
			y_cor_list.append(y[i, j])

	return x_cor_list, y_cor_list


def npcr(mat1, mat2):
	"""Calculate the npcr of two images"""
	npcr = 0
	w, h = mat1.shape
	if mat1.shape != mat2.shape:
		return -1
	for i in range(w):
		for j in range(h):
			if mat1[i,j] != mat2[i,j]:
				npcr += 1
	npcr /= (w*h)
	return npcr*100			


def mse(mat1, mat2):
	"""MSE of plain image and decrypted image"""
	mse = 0
	w, h = mat1.shape
	if mat1.shape != mat2.shape:
		return -1
	print("inside mse")
	print(mat1)
	print(mat2)
	for i in range(w):
		for j in range(h):
			mse += 	pow((int(mat1[i,j]) - int(mat2[i,j])), 2)
	return mse/ (w*h)


def psnr(mse):
	return 20 * math.log((255/math.sqrt(mse)), 10)

def uaci(mat1, mat2):
	"""calculate the uaci of two images"""
	uaci = 0
	w, h = mat1.shape
	if mat1.shape != mat2.shape:
		return -1

	for i in range(w):
		for j in range(h):
			sum_=0
			sum_ = abs(int(mat1[i,j]) - int(mat2[i,j]))
			sum_ /= 255
			sum_ *= 100
			uaci += sum_
	uaci /= (w*h)
	return uaci


def Rand(start, end, num):
	"""Generate num random numbers in range start-end"""
	res = []
	for j in range(num):
		res.append(random.randint(start, end))

	return res


def img_to_mat(path):
	""" function to read image and convert to a matrix""" 
	img = Image.open(path)
	img2 = ImageOps.grayscale(img)

	return np.array(img2)


def parameters_hca():
	x0 =  rand_gen(100000000000,100000000000)
	u_x = rand_gen(100000000000,100000000000)
	# print(u, x0)
	y0 =  rand_gen(100000000000,100000000000)
	u_y = rand_gen(100000000000,100000000000)

	z0 =  rand_gen(100000000000,100000000000)
	u_z = rand_gen(100000000000,100000000000)
	return x0, u_x, y0, u_y, z0, u_z


def image_info(img):
	""" function to return image info"""
	print(img.format)
	print(img.size)
	print(img.mode)


def display_image(mat):
	""" display image from matrix"""
	img = Image.fromarray(mat)
	img.show()


def swap(a,b):
	a, b = b, a
	return a,b


def ruled_randomizer(n, d):
	L = [0] * n

	max_var = n + 1
	q = [0] * n
	d = d.ravel() 

	for i in range(1, n+1):

		q[i-1] = (1 + (d[i-1]%n))

		# print(q[i-1])
		if(L[q[i-1]- 1] == 1):
			k = max_var - 1
			while(L[k-1] == 1):
				k = k - 1
			q[i-1] = k
			max_var = k
		else:
			L[q[i-1]-1] = 1

	return q


def display_image_float(mat):
	new_mat = np.matrix(mat.ravel())
	normalizer = Normalizer().fit(new_mat)
	normalized_mat = normalizer.transform(new_mat).reshape(mat.shape)
	# print(mat.shape)
	res = cv2.normalize(normalized_mat, None, 0, 255, norm_type=cv2.NORM_MINMAX)
	cv2.imshow("encrypted", res)
	cv2.waitKey()
	# plt.imshow(normalized_mat.reshape(mat.shape))
	# plt.show()


def rand_gen(below, baseline):
	""" random number generator within range"""
	return secrets.randbelow(below)/ baseline


def initial_processing(mat):
	""" function to return all matrix shape and convert to vector for processing"""
	flat_arr = mat.ravel()
	return mat.shape, np.matrix(flat_arr)

