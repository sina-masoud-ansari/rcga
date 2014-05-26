import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Issues

* Seems like the first prediction is always off, check with test phase values = expeceted output 

"""

"""
Tweaks

* Could use randn for weights, but need to scale them or check they are between -1 and 1

"""

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
	# SMAPE after Ahmed et al
	def SMAPE(A, F):
		# F is forecast, A is actual
		# normalised input makes first values 0
		A = A[1:]
		F = F[1:]
		a = np.absolute(F - A)
		b = A + F
		err = np.divide(a, b)
		return np.mean(err)

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

def main():
	
	# Params
	np.random.seed(42)
	train_ratio = 0.9
	test_ratio = 0.03
	ignore_ratio = 1 - test_ratio - train_ratio
	r_size = 100
	conn_density = 0.5
	spec_rad = 1.3
	use_output_bias = True
	print "Size {s}, density {d}, spec_rad {spr}".format(s=r_size, d=conn_density, spr=spec_rad)
	
	# Load data
	fname = sys.argv[1]
	u = np.loadtxt(fname)
	
	# Normalise
	u_max = np.amax(u)
	u_min = np.amin(u)
	u_range = u_max - u_min
	# Normalisation
	u = (u - u_min) / u_range
	
	# Partition the input into training test and ignore sets
	N_u = len(u)
	N_ignore = int(ignore_ratio * N_u)
	N_test = int(test_ratio * N_u)
	N_train = N_u - N_ignore - N_test
	
	# partition start and end indices
	S_ignore = 0
	E_ignore = S_ignore + N_ignore
	
	S_train = E_ignore
	E_train = S_train + N_train
	
	S_test = E_train
	E_test = S_test + N_test
			
	# Report partitions
	print "Ignoring from {start} to {end}, length {length}".format(start=S_ignore, end=E_ignore, length=N_ignore)
	print "Train set from {start} to {end}, length {length}".format(start=S_train, end=E_train, length=E_train-S_train)
	print "Test set from {start} to {end}, length {length}".format(start=S_test, end=E_test, length=E_test-S_test)		
		
	# Setup model components
	# Could consider sigma * randn() + mu
	W_rr = np.random.rand(r_size, r_size) - 0.5 # res to res connections
	W_rr = Util.sparse_matrix(W_rr, conn_density)
	W_or = np.random.rand(r_size, 1) - 0.5 # output to res connections
	W_or = Util.sparse_matrix(W_or, conn_density)
	#W_ib = np.random.rand(r_size, 1) - 0.5  # bias input to nodes
	#W_ib = Util.sparse_matrix(W_ib, conn_density)
	x_t = np.zeros((r_size, 1)) # node states
	
	# Rescale W_rr
	rho_W = np.max(np.abs(np.linalg.eig(W_rr)[0]))
	W_rr *=  spec_rad / rho_W
	
	X = None # collected states matirx
	Y = None # collected outputs matrix
	
	# Warm up (no recording)
	for t in range(S_ignore, E_ignore):
		if t == S_ignore:
			y_t = 0
		else:
			y_t = u[t-1] # teacher forcing output
		x_t = np.tanh(np.dot(W_rr, x_t) + np.dot(W_or, y_t))
	
	# Training (recording)
	for t in range(S_train, E_train):
		y_t = u[t-1] # teacher forcing output
		x_t = np.tanh(np.dot(W_rr, x_t) + np.dot(W_or, y_t))
		# collect states
		if t == S_train:
			X = x_t.T
			Y = u[t]
		else:
			X = np.vstack((X, x_t.T))
			Y = np.vstack((Y, u[t]))
	
	# Learn W_ro
	
	# Add constant bias if required
	if use_output_bias:
		X = np.hstack((X, np.ones((X.shape[0], 1))))
	
	
	W_ro = np.linalg.lstsq(X, Y)[0]
	Y = None # collect outputs
	
	# Test (it's already been warmed up)
	for t in range(S_test, E_test):
		y_t = 0.0
		if use_output_bias:
			y_t = np.dot(W_ro.T, np.vstack((x_t, np.ones(1))))
		else:
			y_t = np.dot(W_ro.T, x_t)
	
		x_t = np.tanh(np.dot(W_rr, x_t) + np.dot(W_or, y_t))
		# collect outputs
		if t == S_test:
			Y = y_t
		else:
			Y = np.vstack((Y, y_t))
	
	# Construct input and output signal
	Y = Y.reshape(len(Y),)
	u_pred = np.concatenate((u[S_ignore:E_train], Y))
	u_real = u[:len(u_pred)]
	showlen = 2 * len(Y)
	print "Plotting {n} timesteps".format(n=len(u_real[-showlen:]))
	
	plt.plot(u_real[-showlen:], 'b-')
	plt.plot(u_pred[-showlen:], 'g-')
	plt.axvline(len(Y)-1, color='r', linestyle='--')
	plt.show()
	
	# error
	print "NRMSE: ",Util.NRMSE(u_real, u_pred)

if __name__ == '__main__':
	main()
