import numpy as np
import matplotlib.pyplot as plt
import sys
import multiprocessing
from multiprocessing import Process, Queue

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

	@staticmethod
	# MAPE
	def MAPE(A, F):
		# F is forecast, A is actual
		# normalised input makes first values 0
		A = A + 0.0001
		F = F + 0.0001
		e = A - F
		pe = 100 * np.divide(e, A)
		ape = np.absolute(pe)
		return np.mean(ape)

	@staticmethod
	def sparse_matrix(M, p):
		"""
		M is a (m, n) matirx, p connection probability
		Effectively this will randomly remove some connections by setting
		values to zero, such that the exected connection density is p
		"""
		for i in range(0, M.shape[0]):
			for j in range(0, M.shape[1]):
				r = np.random.rand()
				if r > p:
					M[i][j] = 0
		return M

	@staticmethod
	def validation(res, samples, forecast_length, plot=False):
		e = 0.0
		for s in samples:
			train = s[:-forecast_length]
			res.train(train)
			e += res.test(s, len(train), plot=plot)
		return e / len(samples)

	@staticmethod
	def validationMP_work(res, samples, forecast_length, queue):
		error = 0.0
		for s in samples:
			train = s[:-forecast_length]
			res.train(train)
			error += res.test(s, len(train))
		queue.put(error)

	@staticmethod
	def validationMP(res, samples, forecast_length, plot=False):
		"""
		Validation with multiprocessing support
		"""
		nproc = multiprocessing.cpu_count()
		work_len = max(1, int(len(samples)/nproc))
		work_items = []
		for i in range(nproc):
			w_start = i*work_len
			w_end = min(i+1*work_len, len(samples))
			w_item = samples[w_start:w_end]
			if len(w_item) > 0:
				work_items.append(w_item)
				#print len(w_item)

		queue = Queue()
		errors = []
		procs = []

		for w_item in work_items:
			p = Process(target=Util.validationMP_work, args=(res, w_item, forecast_length, queue))
			p.start()
			procs.append(p)
		for p in procs:
			errors.append(queue.get())
		for p in procs:
			p.join()

		return np.mean(np.array(errors))

	
#	@staticmethod
#	def crossValidation(u, samples, forecast_length, n=1, k):
#		# check params
#		if len(samples) < k:
#			print "crossvalidation parameters: k > num samples"
#			sys.exit(1)
#		# shuffle samples
#		for i in range(0, n):
#			samples = np.random.shuffle(samples)
#			samples_per_bin = len(samples) / k
#			sample_bins = []
#			# partition samples into bins
#			for j in range(0, k):
#				start = j * samples_per_bin
#				end = start + samples_per_bin
#				sample_bins.append([samples[start:end])
#			# process each bin as test set
#			for j in range(0, k):
#				test_set = sample_bins[j]
#				train_set = sample_bins[:j] + sample_bins[j:]
#				# flatten samples in train and test set
#				test_set = [sample for bin in test_set for sample in bin]
#				train_set = [sample for bin in train_set for sample in bin]
#				#

			


	@staticmethod
	def createSamples(u, train_length, test_length):
		samples = []

		# divide input into n equal sized segments
		length = train_length + test_length
		n_segments = len(u) / length
		
		for i in range(0, n_segments):
			sample_start = i * length
			sample_end = sample_start + length
			samples.append(u[sample_start:sample_end])

		return samples

class ReservoirError(Exception):
	def __init__(self, msg):
		self.msg = msg
	def __str__(self):
		return self.msg
	

class Reservoir:

	def __init__(self, size, spectral_radius=0.9, density=0.5, washout=0.2, seed=None, \
					output_bias=True, W_rr=None, W_or=None, \
					W_rr_density=None, W_or_density=None):

		# check params
		if np.isclose(size, 0.0):
			raise ReservoirError("Size cannot be 0")
		if washout > 0.8:
			raise ReservoirError("Washout cannot be greater than 0.8")

		# set parameters
		self._size = size # internal representation for ga
		self.size = int(size)  # int rep. for math
		self.spectral_radius = spectral_radius
		self.output_bias = output_bias
		self.washout = washout
		if seed != None:
			np.random.seed(seed)

		self.density = density
		if W_rr_density == None:
			self.W_rr_density = self.density
		else:
			self.W_rr_density = W_rr_density
		if W_or_density == None:
			self.W_or_density = self.density
		else:
			self.W_or_density = W_or_density

		# setup matrices and vectors
		if W_rr == None:
			self.W_rr = np.random.rand(self.size, self.size) - 0.5
			self.W_rr = Util.sparse_matrix(self.W_rr, self.W_rr_density)
		else:
			self.W_rr = W_rr

		if W_or == None:
			self.W_or = np.random.rand(self.size, 1) - 0.5
			self.W_or = Util.sparse_matrix(self.W_or, self.W_or_density)
		else: 
			self.W_or = W_or
		# rescale W_rr
		try:
			self.rho_W_rr = np.max(np.abs(np.linalg.eig(self.W_rr)[0]))
		except ValueError:
			raise ReservoirError("Problem finding largest absolute eigenvalue")
		if np.isclose(self.rho_W_rr, 0.0):
			raise ReservoirError("Max eigenvalue must be greater than 0")
		self.W_rr *=  self.spectral_radius  / self.rho_W_rr

	def normalise(self, u):

		u_max = np.amax(u)
		u_min = np.amin(u)
		u_range = u_max - u_min

		u = (u - u_min) / u_range

		return u

	def check_valid_input(self, u, ignore):

		# check for valid length
		if ignore > len(u):
			return False
		if len(u) < 1:
			return False

		return True
		
	def warmup(self, u, start, end):

		# init state vector
		x_t = np.zeros((self.size, 1)) # node states

		# warm up (no recording)
		for t in range(start, end):
			y_t = 0.0
			if t == start:
				y_t = 0.0
			else:
				y_t = u[t-1] # teacher forcing output
			x_t = np.tanh(np.dot(self.W_rr, x_t) + np.dot(self.W_or, y_t))

		return x_t
	

	def train(self, u, ignore=None):

		if ignore == None:
			ignore = int(self.washout * len(u))


		# check valid values
		if not self.check_valid_input(u, ignore):
			print "Training parameters no good"
			sys.exit(1)
		
		# normalise input
		u = self.normalise(u)

		# setup start and end indices
		S_ignore = 0
		E_ignore = S_ignore + ignore
		
		S_train = E_ignore
		E_train = S_train + (len(u) - ignore)
		
		# setup state vector
		x_t = self.warmup(u, S_ignore, E_ignore)

		X = None # collected states matirx
		Y = None # collected outputs matrix
	
		# training (recording)
		for t in range(S_train, E_train):
			try:
				y_t = u[t-1] # teacher forcing output
			except IndexError:
				raise ReservoirError("Some problem with training partition")
			x_t = np.tanh(np.dot(self.W_rr, x_t) + np.dot(self.W_or, y_t))
			# collect states
			if t == S_train:
				X = x_t.T
				Y = u[t]
			else:
				X = np.vstack((X, x_t.T))
				Y = np.vstack((Y, u[t]))
		
		# learn W_ro
		
		# add constant bias if required
		if self.output_bias:
			X = np.hstack((X, np.ones((X.shape[0], 1))))
		
			self.W_ro = np.linalg.lstsq(X, Y)[0]
			#print "X: ", X.shape, "Y: ", Y.shape, "len_u: ", len(u), "len_ign: ", ignore
			#raise ReservoirError("Error learning readout weights")

	def test(self, u, ignore, plot=False):

		# check valid values
		if not self.check_valid_input(u, ignore):
			print "Test parameters unreasonable"
			sys.exit(1)

		# normalise input
		u = self.normalise(u)

		# setup start and end indices
		S_ignore = 0
		E_ignore = S_ignore + ignore

		# setup up partitions
		S_test = E_ignore
		E_test = S_test + (len(u) - ignore)

		# setup state vector
		x_t = self.warmup(u, S_ignore, E_ignore)

		Y = None # collect outputs
		
		# Test (it's already been warmed up)
		for t in range(S_test, E_test):
			y_t = 0.0
			if self.output_bias:
				y_t = np.dot(self.W_ro.T, np.vstack((x_t, np.ones(1))))
			else:
				y_t = np.dot(self.W_ro.T, x_t)
		
			x_t = np.tanh(np.dot(self.W_rr, x_t) + np.dot(self.W_or, y_t))
			# collect outputs
			if t == S_test:
				Y = y_t
			else:
				Y = np.vstack((Y, y_t))

		# calculate error
		Y = Y.reshape(len(Y),)
		u_pred = np.concatenate((u[S_ignore:E_ignore], Y))
		u_real = u
		#nrmse = Util.NRMSE(u_real, u_pred)
		#smape = Util.SMAPE(u_real, u_pred)
		mape = Util.MAPE(u_real, u_pred)
		error = mape

		if plot:
			showlen = 2 * len(Y)
			plt.plot(u_real[-showlen:], 'b-')
			plt.plot(u_pred[-showlen:], 'g-')
			plt.axvline(len(Y)-1, color='r', linestyle='--')
			plt.legend(['Real', 'Forecast'])
			plt.show()

		return error

def main1():
	fname = sys.argv[1] # input file
	u = np.loadtxt(fname)
	res = Reservoir(100)
	train_start = 250
	train_length = 400
	train_end = train_start + train_length
	train_ignore = 50
	test_ignore = 100
	test_start = train_end - test_ignore
	test_length = 10
	test_end = test_start + test_ignore + test_length
	u_train = u[train_start:train_end]
	res.train(u_train, train_ignore)
	u_test = u[test_start:test_end]
	error = res.test(u_test, test_ignore, plot=True)
	print "MAPE: {m}".format(m=error)

def main2():
	fname = sys.argv[1] # input file
	u = np.loadtxt(fname)
	reservoirs = []
	res_sizes = 100 + np.random.randn(10) * 100
	res_sizes = [s for s in res_sizes if s > 0]
	for size in res_sizes:
		reservoirs.append(Reservoir(size))
	sample_length = 100
	forecast_length = 10
	samples = Util.createSamples(u, sample_length, forecast_length)
	errors = []
	for res in reservoirs:
		errors.append(Util.validation(res, samples, forecast_length, plot=False))
	for e in errors:
		print "MAPE: {m}".format(m=e)

if __name__ == '__main__':
	main2()
