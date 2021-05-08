""" Standard libs"""
import statistics
import numpy as np
import math


def d(x):
	"""function to calculate D(x)"""
	e_x = np.mean(x)
	n = len(x)
	summation = 0
	for i in range(n):
		summation += pow((x[i]-e_x),2)
	return summation/n


def cov(x, y):
	"""function to calculate the covariance of two sequences"""
	n = len(x)
	e_x = np.mean(x)
	e_y = np.mean(y)
	summation = 0
	for i in range(n):
		summation += ((x[i]-e_x)*(y[i]-e_y))

	return summation/ n


def correlation_coeff(x, y):
	x = np.array(x)
	y = np.array(y)
	d_x = d(x)
	d_y = d(y)
	e_x = np.mean(x)
	e_y = np.mean(y)
	cov_xy = cov(x, y)
	correlation_coefficient = cov_xy/(math.sqrt(d_x)*math.sqrt(d_y))
	return correlation_coefficient
