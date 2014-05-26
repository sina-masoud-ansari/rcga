import numpy as np
import numpy.testing as npt
import unittest
import sys

class Signal:

	def __init__(self):
		pass

	def transf(self, x):
		if len(np.nonzero(x)[0]) < len(x):
			# Raise error
			print "nonzero elements found"
			sys.exit(1)
			# TODO raise error
		return np.log(x)

	def untransf(self, x):
		return np.exp(x)
		
	def norm(self, x):
		# remove mean and divide by standard deviation
		self.mean = np.mean(x)
		self.std = np.std(x)
		return (x - self.mean) / self.std

	def denorm(self, x):
		return x * self.std + self.mean

	def prep(self, x):
		return self.norm(self.transf(x))

	def postp(self, x):
		return self.untransf(self.denorm(x))


class TestSignal(unittest.TestCase):

	def setUp(self):
		x = np.linspace(np.pi, 2*np.pi, 300)
		# Need to ensure all non-zero values otherwise nan will occur during transform
		self.y = np.sin(x) + np.cos(2*x) + 0.5*np.sin(4*x) + np.random.normal(scale=0.2, size=300) + np.ones(300)*10 

	def test_trans_untransf(self):
		s = Signal()
		npt.assert_almost_equal(s.untransf(s.transf(self.y)), self.y)

	def test_norm_denorm(self):
		s = Signal()
		npt.assert_almost_equal(s.denorm(s.norm(self.y)), self.y)

	def test_prep_post(self):
		s = Signal()
		npt.assert_almost_equal(s.postp(s.prep(self.y)), self.y)
		

if __name__ == '__main__':
	unittest.main()
