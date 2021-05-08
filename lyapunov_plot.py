"""standard libs"""
import os
import unittest
import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import *

from math import log
from multiprocessing import Pool
from statistics import mean

""" userdefined libs """
from lyapunov_exponent import lyapunov_exponent
from henon_map import HenonMap
from sine_map import SineMap



# HENON_MAP_REFERENCE_SOLUTION = array([-1.61, 0.41])
# TEST_TOLERANCE = 0.01


# class LyapunovExponentsTestCase(unittest.TestCase):
#     def henon_map_test(self):
#         l = lyapunov_exponent(HENON_MAP, initial_conditions=array((0.1, 0.1)))
#         self.assertEqual(norm(HENON_MAP_REFERENCE_SOLUTION - l, inf) < TEST_TOLERANCE, True)



# if __name__ == '__main__':
#     unittest.main()



amin = 1.75
amax = 3
# step_size_a = (amax - amin) /2000

bmin = 1.75
bmax = 3
# step_size_b = (bmax - bmin) / 2000

avalues = arange(amin, amax, 0.01)
bvalues = arange(bmin, bmax, 0.01)



# def fn(a):
# 	x = array((0.1, 0.1))
# 	b = 0.3
# 	HENON_MAP = HenonMap(a, b)
# 	return lyapunov_exponent(HENON_MAP, initial_conditions=x)


# def fn1(b):
# 	x = array((0.1, 0.1))
# 	a = 1.4
# 	HENON_MAP = HenonMap(a, b)
# 	return lyapunov_exponent(HENON_MAP, initial_conditions=x)

def fn(a):
	x = array((0.1, 0.1))
	b = 2.6
	SINE_MAP = SineMap(a, b)
	return lyapunov_exponent(SINE_MAP, initial_conditions=x)


def fn1(b):
	x = array((0.1, 0.1))
	a = 0.95
	SINE_MAP = SineMap(a, b)
	return lyapunov_exponent(SINE_MAP, initial_conditions=x)

fn_pool = Pool(os.cpu_count())

fn_pool1 = Pool(os.cpu_count())

alambdas = fn_pool.map(fn, avalues)

blambdas = fn_pool1.map(fn1, bvalues)

alambdas1 = []
blambdas1 = []
alambdas2 = []
blambdas2 = []

for lambdas in alambdas:
	a1, a2 = lambdas
	alambdas1.append(a1)
	alambdas2.append(a2) 


for lambdas in blambdas:
	b1, b2 = lambdas
	blambdas1.append(b1)
	blambdas2.append(b2)

"a"    
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(1,1,1)

ax1.set_ylabel('\u03BB')
ax1.plot(avalues, alambdas1, 'b-', linewidth = 1, label = '\u03BB'+'1')
ax1.plot(avalues, alambdas2, 'r-', linewidth = 1, label = '\u03BB'+'2')
ax1.grid('on')
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('a')
ax1.legend(loc='best')
ax1.set_title('Lyapunov exponent')

plt.show()




"b"
fig1 = plt.figure(figsize=(10,7))
ax2 = fig1.add_subplot(1,1,1)


ax2.set_xlabel('b')
ax2.plot(bvalues, blambdas1, 'b-', linewidth = 1, label = '\u03BB'+'1')
ax2.plot(bvalues, blambdas2, 'r-', linewidth = 1, label = '\u03BB'+'2')
ax2.grid('on')
ax2.set_ylim(-0.5, 1.5)
ax2.set_ylabel('\u03BB')
ax2.legend(loc='best')
ax2.set_title('Lyapunov exponent')
plt.show()