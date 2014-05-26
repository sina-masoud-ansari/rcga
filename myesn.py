import numpy as np
import matplotlib.pyplot as plt
import sys

class Data:

	def __init__(self, fname):
		self.data = np.loadtxt(fname, delimiter=",", skiprows=1, usecols=(1,2,3,4,5))
		self.dates = np.loadtxt(fname, delimiter=",", skiprows=1, usecols=(0,), dtype=str)

	def get_high(self):
		return self.data[:, 1]

class Util:
	@staticmethod
	def MSE(A, B):
		return ((A - B) ** 2).mean(axis=0)
	
	@staticmethod
	def RMSE(A , B):
		MSE = Util.MSE(A, B)
		return np.sqrt(MSE)

	@staticmethod
	def NRMSE(A, B):
		RMSE = Util.RMSE(A, B)
		r = np.amax(A) - np.amin(A)
		return RMSE / r

class Reservoir:

	def __init__(self, size, spec_rad=1.0, bias_out=True):
		self.r_size = size
		self.spec_rad = spec_rad
		self.bias_out = bias_out

		# create the reservoir weight matrix (fully connected)
		self.W = np.random.randn(self.r_size, self.r_size)

		# condition W matirix for echo state property
		self.rho_W = max(np.abs(np.linalg.eig(self.W)[0]))
		self.W *=  self.spec_rad / self.rho_W

		# create input weight matrix with NO bias
		self.W_in = np.random.randn(self.r_size, 1)

		# create state vector x
		self.x_t = np.zeros((self.r_size, 1))

		# create output feedback connections
		#self.W_ofb = np.random.randn(self.r_size, 1)

	def partition(self):
		# partition sizes
		self.N_u = len(self.u)
		self.N_ignore = int(self.ignore_ratio * self.N_u)
		self.N_test = int(self.test_ratio * self.N_u)
		self.N_train = int(self.train_ratio * self.N_u)
		
		# partition start and end indices
		self.S_ignore = 0
		self.E_ignore = self.S_ignore + self.N_ignore
		
		self.S_train = self.E_ignore
		self.E_train = self.S_train + self.N_train
		
		self.S_test = self.E_train
		self.E_test = self.S_test + self.N_test
		
		# create partitions of input
		self.u_ignore = self.u[self.S_ignore:self.E_ignore]
		self.u_train = self.u[self.S_train:self.E_train]
		self.u_test = self.u[self.S_test:self.E_test]
		
		# Report partitions
		print "Ignoring from {start} to {end}, length {length}".format(start=self.S_ignore, end=self.E_ignore, length=self.N_ignore)
		print "Train set from {start} to {end}, length {length}".format(start=self.S_train, end=self.E_train, length=len(self.u_train))
		print "Test set from {start} to {end}, length {length}".format(start=self.S_test, end=self.E_test, length=len(self.u_test))		
	
	def create_examples(self):
		# create training input and target pairs
		# this is done by right shifting the training input by 1 step
		self.t_in = self.u_train[:-1]
		self.t_out = self.u_train[1:]

	def step_x(self, u_t):
		return np.tanh(np.dot(self.W_in, u_t) + np.dot(self.W, self.x_t))

	def warmup(self, n):
		# run warm up steps
		for t in range(0, n):
			u_t = self.u[t]
			self.x_t = self.step_x(u_t)

	def __train(self):

		self.warmup(len(self.u_ignore))

		# declare state matrix X and target matrix Y
		X = None
		Y = None

		# run training steps and collect states
		for t in range(0, len(self.t_in)):
			u_t = self.t_in[t]
			y_t = self.t_out[t]
			self.x_t = self.step_x(u_t)
			if t == 0:
				X = self.x_t.T
				Y = y_t
			else:
				X = np.vstack((X, self.x_t.T))
				Y = np.vstack((Y, y_t))
	
		if self.bias_out:
			# model constant output for 1 from one node
			bias = np.ones((X.shape[0], 1))
			X = np.hstack((X, bias))

		self.W_out = np.linalg.lstsq(X, Y)[0] 

	def test(self, err_func, plot=False):
		self.warmup(self.N_ignore + self.N_train)

		Y = None
		
		# Begin prediction
		for t in range(0, len(self.u_test)):
			
			y_t = None
			if self.bias_out:
				x_t_b = np.vstack((self.x_t, np.ones((1,1))))
				y_t = np.dot(self.W_out.T, x_t_b)
			else:
				y_t = np.dot(self.W_out.T, self.x_t) # should test that this is the correct shape

			if t == 0:
				Y = y_t
			else:
				Y = np.vstack((Y, y_t))

			u_t = y_t
			self.x_t = self.step_x(u_t)

		# assign prediction output
		self.Y = Y.reshape(Y.shape[0],)
		
		# append to existing signal 
		self.prefix = self.u[self.S_ignore:self.E_train]
		self.output = np.concatenate((self.prefix, self.Y))
		self.real = np.concatenate((self.prefix, self.u_test))

		if plot:
			self.__plot()
		
		# test error
		return err_func(self.real, self.output)

	def __plot(self):
		showlen = 2 * len(self.Y)
		plt.plot(self.real[-showlen:], 'b-')
		plt.plot(self.output[-showlen:], 'g-')
		plt.axvline(len(self.Y)-1, color='r', linestyle='--')
		plt.show()

	def train(self, input, r_ignore=0.1, r_test=0.1):
		self.u = input
		self.ignore_ratio = r_ignore
		self.test_ratio = r_test
		self.train_ratio = 1.0 - self.ignore_ratio - self.test_ratio
		self.partition()
		self.create_examples()
		self.__train()

# MAIN

# input file CSV (Date, Open, High, Low, Close, Volume)
fname = sys.argv[1]

# reservoir constants
size = 40
spr = 1.0
#np.random.seed(47)

# input
data = Data(fname)
input = data.get_high()
# For the linear and sin test cases
#input = np.loadtxt(fname)

# steup reservoir and train
res = Reservoir(size=size, spec_rad=spr)
res.train(input)
print "NRMSE: ",res.test(Util.NRMSE, plot=True)

