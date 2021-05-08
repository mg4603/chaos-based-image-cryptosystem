""" pre-defined libraries"""
import math
import numpy as np
import secrets
"""User-defined libraries"""
import helper
import phase_diagram
path = "./sample.png"



"""step 1

1. Take source image of size m*n as input.
"""
mat = helper.img_to_mat(path)

""" values for two parameters for classic henon map"""
a = 1.4
b = 0.3

m, n = mat.shape

"""step 2

2. Generate random variables from Henon chaotic map by following steps:
(a) X, Y = Henon(m*n) in order to generate random variables using Henon
chaotic map of size equal to no. of pixels.
(b) X = abs(floor(X(1:m*n)*1000000));
(c) Y = abs(floor(Y(1:m*n)*1000000));
(d) X = reshape(X,m,n); Y = reshape(Y,m,n);
(e) D = X*Y;
(f) D = D*sum(key) sum of the digits of 128 bit key.


"""
x0 = helper.rand_gen(10000000000000000, 10000000000000000)
y0 = helper.rand_gen(10000000000000000, 10000000000000000)
try:
		x1 = 1 - (a*math.pow(x0, 2)) + y0
		y1 = b * x0
except Exception as e:
	print(e)

x2 = 0
y2 = 0
x = []
y = []
len_mat = m * n
for i in range(len_mat):
	try:
		x2 = 1 - (a*math.pow(x1, 2)) + y1
		y2 = b * x1
		x.append(x1)
		y.append(y1)
		x1 = x2
		y1 = y2
	except Exception as e:
		print(e)

x = np.array(x).reshape((m, n))
y = np.array(y).reshape((m, n))

x = np.absolute(np.floor(x) * 1000000)
y = np.absolute(np.floor(x) * 1000000)

def getSum(n):
	sum_ = 0
	for digit in str(n):
		sum_ += int(digit)
	return sum_



key = secrets.randbits(128)

D = np.matmul(x, y)
D = D * getSum(key)

print(D.shape)


"""step3

3. Generate permutation matrix P(m*n) by calculating position for each row i as
(a) pos = mod(D, n) + 1
(b) P(i, pos) = 1
(c) Other entries being zero for the row i

"""


P = np.zeros((m,n))

for i in range(m):
	pos = np.mod(D, n) +1
	P[i, pos] = 1

"""


4. Perform combined shuffling operation by first performing vertical shuffling and
then horizontal shuffling as shown below: For each i, j from 1 to n
(a) vI(1:n, j) = P*I(1:n, j)
(b) cI(j, 1:n) = vI(j, 1:n)*P
"""
vI = np.zeros((m,n))
cI = np.zeros((m,n))
for j in range(n):
	vI[1:n,j] = P * mat[1:n, j]
	cI[j, 1:n] = vI[j, 1:n] * P


"""

5. For i = 1:m
(a) j = mod(i,16) + 1
(b) o D(i,1:m) = bitxor(D(i,1:m),okey(1,j))

"""

for i in range(m):
	j = (i% 16) + 2



"""
6. D = mod(D,255) + 1. cipher image


"""

D = np.mod(D, 255) 

"""
7. Generate final encrypted image by applying XOR operation between the shuffled
image and the cipher image.
"""



enc_mat = np.bitwise_xor(mat, cipher_mat)

helper.display_image(enc_mat)
