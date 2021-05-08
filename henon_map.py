from numpy import *

class HenonMap:
	""" Class to hold the parameters of the Henon Map and evaluate it along its directional derivative.
        The default parameters are chosen as the canonical ones in the initialization.
        It instantiates a callable object f_df, in such a way that f_df(xy, w)
        returns two values f(xy) and df(xy, w), where
        f(xy) is the Henon map evaluated at the point xy and
        df(xy, w) is the differential of the Henon evaluated at xy in the direction of w.
    """

	def __init__(self, a=1.4, b=0.3):
		self.a, self.b = a, b

	def f(self, xy):
		x, y = xy
		return array([self.a - x ** 2 + y, self.b * x])

	def df(self, xy, w):
		x, y = xy
		j = array([[-2 * x, 1],
                   [self.b, 0]])
		return j @ w

	def __call__(self, xy, w):
		return self.f(xy), self.df(xy, w)