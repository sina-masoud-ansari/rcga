import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import unittest
import sys

class Reservoir:

	def __init__(self, size, density=0.25, spec_rad=0.9, bias=0.0):

		# params
		self.size = size
		self.density = density
		self.Wbias_scaling = bias
		self.spectral_radius = spec_rad

		#########
		# Model #
		#########

		# Output feedback weights (otherwise known as Win)
		self.Wofb = sp.sparse.rand(self.size, 1, self.density)
		#self.Wofb = np.random.rand(self.size, 1) # 100% density
		self.Wofb = self.Wofb * 1 # weight adjustment

		print "Wofb"
		print self.Wofb

		# Internal reservoir weights
		self.Wres = sp.sparse.rand(self.size, self.size, self.density, format='lil')
		#self.Wres = np.random.rand(self.size, self.size)
		self.Wres = self.Wres * 1  # weight adjustment

		print "Wres constructed"
		print self.Wres

		# remove self-self loops
		self.Wres.setdiag(np.zeros(self.size))
		# scale to support echo state property
		self.Wres = self.rescale(self.Wres)

		print "Wres rescaled"
		print self.Wres

		# Reservoir to readout weights
		self.Wout = np.random.uniform(-1, 1, self.size).reshape(self.size, 1)
		self.Wout = self.Wout * 1

		# Reservoir node bias
		self.Wbias = np.random.uniform(-1, 1, self.size).reshape(self.size, 1)
		self.Wbias = self.Wbias * self.Wbias_scaling

		print "Wbias"
		print self.Wbias
	
		# state and output
		#x0 = np.random.uniform(-1, 1, self.size).reshape(self.size, 1)
		x0 = np.zeros(self.size).reshape(self.size, 1)
		y0 = 0.0 
		self.init(x0, y0)

	def rescale(self, M):
		spr = self.spec_rad(M)
		return M * self.spectral_radius / spr

	def spec_rad(self, M):
		w,v = sp.sparse.linalg.eigs(M, k=self.size-2)
		#w,v = np.linalg.eig(M)
		return np.amax(np.absolute(w))

	def step(self):
		self.x = self.f(self.Wres * self.x + self.Wofb * self.y + self.Wbias)
		self.y = np.dot(self.Wout.T, self.x) # not sparse

	def f(self, x):
		return np.tanh(x)

	def collect(self):
		self.states = np.append(self.states, self.x.T, axis=0)
		print self.states
		self.output = np.append(self.output, self.y)
	
	def init(self, x, y):
		self.x = x
		self.y = y
		self.states = self.x.T
		self.output = np.array(self.y)

	def teach(self, target):
		#x0 = np.random.randn(self.size, 1)
		x0 = np.zeros(self.size).reshape(self.size, 1)
		y0 = target[0]
		self.init(x0, y0)

		for i in range(1, len(target)):
			self.step()
			self.y = target[i] # overwrite generated y value with teacher signal
			self.collect()

	def learn(self, washout=0):

		# for one step ahead sequence
		yf = self.output[1:]
		sf = self.states[:-1, :]

		# washout
		yf = yf[washout:]
		sf = sf[washout:, :]

		# solve
		print "Wout b4"
		print self.Wout
		self.Wout = np.linalg.lstsq(sf, yf)[0]
		self.Wout = self.Wout.reshape(len(self.Wout), 1)
		print "Wout after"
		print self.Wout

	# SMAPE after Ahmed et al
	def smape(self, x, y):
		m = len(x)
		sum = 0.0
		for i in range(0, m):
			err = sp.absolute(y[i] - x[i]) / ((sp.absolute(y[i]) + sp.absolute(x[i])) / 2.0)
			sum = sum + err
		sum = sum / m
		return sum


	def test(self, input, steps):
		#x0 = np.random.randn(self.size, 1)
		#x0 = np.zeros(self.size).reshape(self.size, 1)
		x0 = self.x
		y0 = input[0]
		self.init(x0, y0)

		# warm up
		for i in range(1, len(input[:-steps])):
			self.step()
			self.y = input[i] # overwrite generated y value with teacher signal
			self.collect()
		
		# generate
		for i in range(0, steps):
			self.step()
			self.collect()
	
		# compute error
		#error = self.smape(input, self.output)

		return self.output

class TestReservoir(unittest.TestCase):
	pass


if __name__ == '__main__':
	unittest.main()
