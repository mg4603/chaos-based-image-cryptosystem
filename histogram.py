import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image, ImageOps
# import cv2

# img = cv2.imread('./sample.png',0)
# plt.hist(img.ravel(),256,[0,256]); plt.show()
# img1 = Image.open('./sample.png')
# img2 = ImageOps.grayscale(img1)


def plot_histogram(mat):
	plt.hist(np.ravel(mat), bins = 255)
	plt.margins(x=0)
	plt.show()
	return

# plot_histogram(np.asarray(img2))