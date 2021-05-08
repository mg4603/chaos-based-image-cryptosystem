""" pre-defined libraries"""
import math
import numpy as np
"""User-defined libraries"""
import helper
import phase_diagram
import histogram
path = "./sample.png"




mat = helper.img_to_mat(path)

histogram.plot_histogram(mat)

def encrypt_im(mat):
	
	shape, vector = helper.initial_processing(mat)
	w,h = shape
	
	# shape = mat.shape
	# w, h = shape
	# vector = np.matrix(mat.ravel())
	
	# x0 = 0.38 + helper.rand_gen(6200000000000000, 10000000000000000)
	# y0 = helper.rand_gen(10000000000000000, 10000000000000000)

	# alpha = 0.49
	# beta = 0.79


	alpha = 0.49
	beta = 0.79
	alpha_ = helper.rand_gen(5100000000000001, 10000000000000000)
	beta_ = helper.rand_gen(2100000000000001, 10000000000000000)
	alpha += alpha_
	beta += beta_

	# alpha = 0.85
	# beta = 0.90

	# x01 = 0.38 + helper.rand_gen(6200000000000000, 10000000000000000)
	# y01 = helper.rand_gen(10000000000000000, 10000000000000000)

	x0 = 0.5
	y0 = 0.6
	x01 = 0.5
	y01 = 0.6
	
	# alpha1 = 0.49
	# beta1 = 0.79
	alpha1 = 0.49
	beta1 = 0.79
	alpha_1 = helper.rand_gen(5100000000000001, 10000000000000000)
	beta_1 = helper.rand_gen(2100000000000001, 10000000000000000)
	alpha1 += alpha_1
	beta1 += beta_1
	n = w*h

	r1 = []
	r2 = []
	try:
		x1 = math.pow(math.sin(alpha / y0), 3/2)
		t1 = (x1 * (10**15)) % n
		r1.append(t1)
		y1 = math.cos(beta * math.acos(x0))
		
		
		x11 = math.pow(math.sin(alpha1 / y01), 3/2)
		y11 = math.cos(beta1 * math.acos(x01))
		t2 = (y11 * (10**15))%256
		r2.append(t2)

	except Exception as e:
		print(e)

	x2 = 0
	y2 = 0

	for i in range(n-1):
		try:
			x2 = math.pow(math.sin(alpha / y1), 3/2)
			t1 = (x2 * (10**15)) % n
			r1.append(t1)
			y2 = math.cos(beta * math.acos(x1))
					
			x1 = x2
			y1 = y2

			x21 = math.pow(math.sin(alpha1 / y11), 3/2)
			y21 = math.cos(beta1 * math.acos(x11))
			t2 = (y21 * (10**15))%256
			r2.append(t2)

			x11 = x21
			y11 = y21

		except Exception as e:
			print(x0, x01)
			print(y0, y01)
			print(e)
			break
	r1 = np.array(r1)
	r1 = np.round(r1, 0)
	r2 = np.array(r2)
	r2 = np.uint8(r2)
	print(x0, x01)
	print(y0, y01)
	# print("encrypt")
	# print(vector)
	# print(r1)
	# print(r2)
	# phase_diagram.draw_phase_diagram(x, y)
	

	# for j in range(1000):
	# 	for i in range(n):
	# 		temp = vector[0, i]
	# 		index = round(r1[i]) - 1

	# 		vector[0,i] = vector[0,index]
	# 		vector[0, index] = temp
	


	for i in range(n):
		temp = vector[0, i]
		index = round(r1[i]) - 1

		vector[0,i] = vector[0,index]
		vector[0, index] = temp

	ci_vec = np.bitwise_xor(vector[0,:] , r2[:])
	
	mat2 = np.asarray(ci_vec).reshape(shape)

	return mat2, x0, y0, alpha, beta, x01, y01, alpha1, beta1





def decrypt(mat, x0, y0, alpha, beta, x01, y01, alpha1, beta1):
	shape, vector = helper.initial_processing(mat)
	w,h = shape
	
	# shape = mat.shape
	# w, h = shape
	# vector = np.matrix(mat.ravel())

	
	n = w*h

	r1 = []
	r2 = []
	try:
		x1 = math.pow(math.sin(alpha / y0), 3/2)
		t1 = (x1 * (10**15)) % n
		r1.append(t1)
		y1 = math.cos(beta * math.acos(x0))
		
		
		x11 = math.pow(math.sin(alpha1 / y01), 3/2)
		y11 = math.cos(beta1 * math.acos(x01))
		t2 = (y11 * (10**15))%256
		r2.append(t2)

	except Exception as e:
		print(e)

	x2 = 0
	y2 = 0

	for i in range(n-1):
		try:
			x2 = math.pow(math.sin(alpha / y1), 3/2)
			t1 = (x2 * (10**15)) % n
			r1.append(t1)
			y2 = math.cos(beta * math.acos(x1))
					
			x1 = x2
			y1 = y2

			x21 = math.pow(math.sin(alpha1 / y11), 3/2)
			y21 = math.cos(beta1 * math.acos(x11))
			t2 = (y21 * (10**15))%256
			r2.append(t2)

			x11 = x21
			y11 = y21

		except Exception as e:
			print(e)


	r1 = np.array(r1)
	r1 = np.round(r1, 0)
	r2 = np.array(r2)
	r2 = np.uint8(r2)

	# print("decrypt")
	# print(vector)
	# print(r1)
	# print(r2)

	image_mat = np.bitwise_xor(vector[0,:], r2[:])
	
	# for j in range(100):
	# 	for i in range(n-1, 0, -1):
	# 		temp = image_mat[0, i]
	# 		index = round(r1[i]) - 1

	# 		image_mat[0,i] = image_mat[0,index]
	# 		image_mat[0, index] = temp
	
	for i in range(n-1, 0, -1):
			temp = image_mat[0, i]
			index = round(r1[i]) - 1

			image_mat[0,i] = image_mat[0,index]
			image_mat[0, index] = temp


	# print(image_mat)
	return image_mat.reshape(shape)




# mat = np.array([[1 , 2, 3],[4, 5, 6],[7, 8, 9]])
# mat = np.array([[1, 2],[3, 4]])
# mat2, x0, y0, alpha, beta = encrypt_im(mat)
# print(mat2)
# decrypt(mat2, x0, y0, alpha, beta)

enc_img_mat, x0, y0, alpha, beta,  x01, y01, alpha1, beta1 = encrypt_im(mat)
histogram.plot_histogram(enc_img_mat)
helper.display_image(enc_img_mat)


decrypt_img_mat = decrypt(enc_img_mat, x0, y0, alpha, beta,  x01, y01, alpha1, beta1)

helper.display_image(decrypt_img_mat)
