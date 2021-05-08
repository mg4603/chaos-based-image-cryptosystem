import numpy as np
import math
from sklearn.preprocessing 	import KBinsDiscretizer
import sys
#userdefined libs
import helper
import phase_diagram
import histogram
path = "./sample.png"
# path = "./sample1.gif"
# path = "./good_vals.png"

# Read plain image (I) of dimension (M × N = R).

mat = helper.img_to_mat(path)
print(mat.shape)
helper.display_image(mat)
#Divide I into two vertical halves—I 1 and I 2 each of size M × (N/2).


# print(I1.shape)

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


def diffusion_layer(mat):
	
	shape, vector = helper.initial_processing(mat)
	w,h = shape
	
	# shape = mat.shape
	# w, h = shape
	# vector = np.matrix(mat.ravel())
	I1, I2 = np.split(mat, 2, axis = 1)
	print(I1.shape)
	x = []
	y = []
	n = w*h


	####### HENON MAP
	# a = 1.4 
	# b = 0.3
	# x0 = 0.631354477
	# y0 = 0.189406343
	# x = []
	# y = []
	# N0 = 30000
	# x.append(x0)
	# y.append(y0)
	# try:
	# 	x1 = 1 - (a * math.pow(x0, 2)) + y0
	# 	y1 = b * x0
	# 	x.append(x1)
	# 	y.append(y1)

	# except Exception as e:
	# 	print(e)

	# for i in range(int((n/2))-2):
	# 	try:
	# 		x2 = 1 - (a * math.pow(x1, 2)) + y1
	# 		y2 = b * x1
	# 		x.append(x2)
	# 		y.append(y2)
	# 		x1 = x2
	# 		y1 = y2


	# 	except Exception as e:
	# 		print(e)
	# 		print(x1,y1,t1)
	# 		break
	
	############################################

	u = 0.6 + helper.rand_gen(100000000000,100000000)
	x0 = helper.rand_gen(1000000000,1000000000)
	y0 = helper.rand_gen(1000000000,1000000000)
	
	N0 = 30000
	# x0 = -40 + helper.rand_gen(50000000,100000000)
	# y0 = -30 + helper.rand_gen(20000000000,10000000000)
	# u = 1.05 + helper.rand_gen(100000,10000000)
	print(u, x0, y0)
	t0 = 0.4 - (6/(1+math.pow(x0, 2)+math.pow(y0, 2)))
	x1 = x0
	y1 = y0
	t1 = t0

	for i in range((int((n/2)))+30000):
		try:
			x2 = 1 + u * (x1*math.cos(t1) - y1*math.sin(t1))
			x2 = x2 % sys.maxsize
			y2 = u * (x1*math.sin(t1) - y1*math.cos(t1))
			y2 = y2 % sys.maxsize
			t1 = 0.4 - (6/(1+math.pow(x1, 2)+math.pow(y1, 2)))
			x.append(x1)
			y.append(y1)
			x1 = x2
			y1 = y2


		except Exception as e:
			print(e)
			print(x1,y1,t1)
			break

	discretizer = KBinsDiscretizer(n_bins = int(n/2)+ 30000, encode='ordinal', strategy = "uniform")
	x = np.array(x[30000:]).reshape(I1.shape)
	y = np.array(y[30000:]).reshape(I2.shape)
	

	# x = np.array(x).reshape((-1,1))
	# y = np.array(y).reshape((-1, 1))
	


	x = discretizer.fit_transform(x).astype(int)
	y = discretizer.fit_transform(y).astype(int)

	x = np.array(ruled_randomizer(int(n/2), x)).reshape(I1.shape)
	y = np.array(ruled_randomizer(int(n/2), y)).reshape(I2.shape)

	sorted_x = np.argsort(x).argsort()
	sorted_y = np.argsort(y).argsort()
	

	I11 = np.bitwise_xor( np.take_along_axis(x, sorted_y, axis = 0), x).reshape(I1.shape)
	I21 = np.bitwise_xor( np.take_along_axis(y, sorted_x, axis = 0), y).reshape(I2.shape)
	
	I_1 = np.concatenate((I11, I21), axis = 1)
	# print(I_1.shape)


	I3, I4 = np.split(I_1, 2, axis = 0)
	x = x.reshape(I3.shape)
	y = y.reshape(I4.shape)
	sorted_x = np.argsort(x, axis = 0)
	sorted_y = np.argsort(y, axis = 0)
	I31 = np.bitwise_xor( np.take_along_axis(x, sorted_y, axis=1), x).reshape(I3.shape)
	I41 = np.bitwise_xor( np.take_along_axis(y,sorted_x, axis = 1), y).reshape(I4.shape)
	
	I_2 = np.concatenate((I31, I41), axis = 0)
	# print(I_2)

	# print("encrypt")
	# print(vector)
	# print(r1)
	# print(r2)
	# phase_diagram.draw_phase_diagram(x, y)

	return I_2

def less_10e14neg(x):
	# print(x)
	while(x < math.pow(10,-14)):
		x = x * 10
	return x


def confusion_layer(mat, S):
	shape, vector = helper.initial_processing(mat)
	w,h = shape
	n = w * h
	while(S>1):
		S = S/ 10
	u =  helper.rand_gen(10000000000,10000000000)
	x0 = helper.rand_gen(10000000000,10000000000)
	print(x0)
	x0 = x0 + S
	x = []
	x1 = x0
	x2 = 0
	# print(x1)
	for i in range(n+30000):
		try:
			x2 = u * x1 * (1 - x1)
			x2 = x2 % math.pow(10,30)
			x2 = less_10e14neg(x2)
			x.append(x1)
			x1 = x2
		except Exception as e:
			print(e)

	discretizer = KBinsDiscretizer(n_bins = n + 30000, encode='ordinal', strategy = "quantile")
	
	x = np.array(x[30000:]).reshape(shape)
	x = discretizer.fit_transform(x).astype(int)
	x = np.array(ruled_randomizer(n, x))
	
	x = np.remainder(x, 8)
	print(x)
	
	encrypted_I = np.right_shift(vector, x).reshape(shape)
	return encrypted_I






def encrypt(mat):
	array_like = mat.ravel()
	S = 0
	for i in array_like:
		S = S + i
	diffuse_image_mat =	diffusion_layer(mat)
	
	return confusion_layer(diffuse_image_mat, S)



# print(mat)
# mat = np.array([[236 , 236, 236, 236],[236, 236, 236, 236],[236, 236, 236, 236],[236, 236, 236, 236]])
encrypt_img = encrypt(mat)
# encrypt_img = diffusion_layer(mat)
helper.display_image_float(encrypt_img)
# print(mat)

# helper.display_image(diffuse_image_mat)