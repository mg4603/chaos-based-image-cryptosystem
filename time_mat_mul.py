import numpy as np
from time import time
a = np.array([[1,2,3],[4, 5, 6],[7, 8, 9]])

t1 = time()
b = np.matmul(a, a)
b = np.matmul(a, a)
b = np.matmul(a, a)
b = np.matmul(a, a)
t2 = time()
print("numpy multiplication: ",t2-t1)

w, h = a.shape
mul = np.zeros((w,h))
t3 = time()
for i in range(w):
	for j in range(h):
		for k in range(w):
			mul[i, j] += a[i,k]*b[k,j]
for i in range(w):
	for j in range(h):
		for k in range(w):
			mul[i, j] += a[i,k]*b[k,j]
for i in range(w):
	for j in range(h):
		for k in range(w):
			mul[i, j] += a[i,k]*b[k,j]
for i in range(w):
	for j in range(h):
		for k in range(w):
			mul[i, j] += a[i,k]*b[k,j]

t4 = time()
print("for loop multiplication: ", t4-t3)

print("ratio", (t4-t3)/(t2-t1))