import bitstring as bs
import numpy as np
import os
import sys

class GA:
	MINIMISE = 0
	MAXIMISE = 1

	@staticmethod
	def create(I, N):
		"""
		Create population of N individuals of class I
		Calls I's __init__ function which should create a random individual
		Return list of individuals
		"""
		pop = []
		for n in range(0, N):
			pop.append(I())

		return pop

	@staticmethod
	def crossover(a, b):
		"""
		Bitstring crossover using random mask. Objects mult implement the
		bitstring method which returns a bitstring.Bits like object
		Bitstrings are made the same length if different by 0-padding on the right
		Bitstrings are left aligned for crossover.
		"""
		sa = a.bitstring()
		sb = b.bitstring()

		# make the bitstrings the same length by righs padding the shorter one
		if len(sa) < len(sb):
			d = len(sb) - len(sa)
			s = [0] * d
			sa.append(s)
		elif len(sb) < len(sa):
			d = len(sa) - len(sb)
			s = [0] * d
			sb.append(s)
		
		# generate mask
		mask_len = len(sa) 
		# mask_len in bytes
		mask_bytes = mask_len / 8
		mmod = mask_len % 8
		if mmod != 0:
			mask_bytes += 1
		mask = bs.BitArray(bytes=os.urandom(mask_bytes), length = 8 * mask_bytes)
		mask = mask[:mask_len]
		
		# generate child
		sc = (sa & mask) | (sb & ~mask)
		return sc

	@staticmethod
	def mutate(s, p):
		"""
		Randomly alter bitstring elements according to fixed probability p
		"""
		sm = bs.BitStream(s)
		for i in range(0, len(s)):
			r = np.random.rand()
			if r < p:
				sm.invert(i)
		return sm
		
	@staticmethod
	def evaluate(pop, fit_func, targets, err_fit):
		"""
		Determine population fitness across multiple targets
		Return vector of length pop size
		"""
		fitness = []
		for i in pop:
			f = fit_func(i, targets, err_fit)
			if len(fitness) == 0:
				fitness = f
			else:
				fitness = np.vstack((fitness, f))
			#print i, f, np.mean(f)
		
		print fitness
		f_mean = np.mean(fitness, axis=1).reshape(len(fitness), 1)

		#for i in range(0, len(pop)):
		#	print pop[i], f_mean[i]
		return f_mean
		 

	@staticmethod
	def rank(pop, fit_func, SEARCH_TYPE, f_limit, max_pop, targets):

		fitness = GA.evaluate(pop, fit_func, targets, err_fit=f_limit+1)

		pop_fit = zip(pop, fitness)

		# Set the search type
		if SEARCH_TYPE == GA.MINIMISE:
			# prune
			if f_limit != None:
				pop_fit = [p for p in pop_fit if p[1] < f_limit]
			# sort by fitness in ascending order
			pop_fit.sort(key=lambda x: x[1])
		elif SEARCH_TYPE == GA.MAXIMISE:
			# prune
			if f_limit != None:
				pop_fit = [p for p in pop_fit if p[1] > f_limit]
			# sort by fitness in descending order
			pop_fit.sort(key=lambda x: x[1], reverse=True)
		else:
			print "Unkown search type: ", SEARCH_TYPE
			sys.exit(1)

		pop_fit = pop_fit[:max_pop]
		
		if len(pop_fit) < 2:
			print "Population size is less than 2!"
			sys.exit(1)
	
		pop, fitness = zip(*pop_fit)
		return pop, fitness

	@staticmethod
	def breed(pop, fit_func, SEARCH_TYPE, c_type, max_off, max_pop, targets, crossover=True, mu=0.001, f_limit=None):
		"""
		Create next generation using replacement
		Method is top-n
		"""
		
		# determine fitness and rank of pop members
		pop, fitness = GA.rank(pop, fit_func, SEARCH_TYPE, f_limit, max_pop, targets)

		# calculate rank
		#rank = np.arange(0, len(pop), dtype=float)
		#rank = 1 - rank / (len(pop) - 1)

		maxf = np.amax(fitness)
		minf = np.amin(fitness)
		rangef = maxf-minf
		rank = 1 -(fitness - minf)/rangef

		# rezip
		pop_fit = zip(pop, rank)
		
		npop = [] # new population

		# Assortive mating
		for i, pi in enumerate(pop_fit):
			Ii = pi[0] # Individual i
			Ri = pi[1] # Rank of i
			print Ii, fitness[i], Ri

			Ri_lim = 0.3 * Ri # Assortive mating lower limit for i
			Im = np.random.rand() # chance of meeting mate Ij
			Im *= Ri # Higher ranked membeurs have higher chance to meet mates

			for j, pj in enumerate(pop_fit):
				Rj = pj[1] # Rank of j
				Rj_lim = 0.3 * Rj # Assortive mating lower limit for j
				r = np.random.rand() # roll for chance to meet mate
				if j > i and r < Im and Rj > Ri_lim and Ri > Rj_lim: 
					# meet a mate successfully and they aren't you and they are in 
					# your league
					Ij = pj[0] # Individual j
					
					# try to have C children
					nc = 0 # number of children
					nf = 0 # number of failures
					while nc < 10 and nf < 3:
						cs = GA.crossover(Ii, Ij)
						cs = GA.mutate(cs, mu)
						Ic = c_type(cs)
						if Ic.failure:
							# avoid adding failures to population
							nf += 1
						else:
							npop.append(Ic)	
							nc += 1

		npop, fitness = GA.rank(npop, fit_func, SEARCH_TYPE, f_limit, max_pop, targets)
		return npop

class Individual:
	def __init__(self, bitstring=None):
		self.failure = False
		if bitstring == None:
			self.dims = (1, 3)
			self.M = np.random.randn(*self.dims)
		else:
			try:
				self.parse(bitstring)
			except:
				self.failure = False

	def __str__(self):
			return str(self.M)

	def parse(self, s):
			# construct dims
			self.dims = tuple(s.readlist('2*uint:4'))
			# check for valid dims
			if not self.check_dims(): 
				self.failure = True
				return

			# use parents cell data
			self.M = np.ndarray(shape=self.dims, dtype=float)
			for i in range(0, self.dims[0]):
				for j in range(0, self.dims[1]):
					self.M[i][j] = s.read('float:32')

			# check for nans
			if not self.check_values():
				self.failure = True
				return

	def bitstring(self):
		s = bs.BitStream()
		# dimensions of vector
		s.append(bs.pack('2*uint:4', *self.dims))
		# vector elements
		for i in range(0, self.dims[0]):
			for j in range(0, self.dims[1]):
				s.append(bs.pack('float:32', self.M[i][j]))
		return s

	def check_dims(self):
		if self.dims[0] == 1:
			return True
		else:
			return False

	def check_values(self):
		return not np.isnan(self.M).any() and not np.isinf(self.M).any()

def fit_func(i, targets, err_fit):
	"""
	Fitness function (length of error vector)
	Return row vector of error between individual and each target
	"""
	fitness = []
	for t in targets:
		if len(i.M[0]) != len(t):
			fitness.append(err_fit)
		else:
			fitness.append(np.linalg.norm(t - i.M[0]))
	return np.array(fitness)

# attributes
N = 100 	# population size
NC = 7		# number of children
mu = 0.5	# mutation rate
G = 10 		# number of generations
f_limit = 200 # max allowable fitness

# create a population of random vectors
pop = GA.create(Individual, N)	# population
indices = np.arange(0, N)
record = None
targets = [[16,16,16]]
for i in range(0, G):
	# evaluate fitness of population with respect to function f and target t
	fitness = GA.evaluate(pop, fit_func, targets=targets, err_fit=f_limit+1)
	#fitness = np.mean(fitness, axis=0)


	# stats
	f_med = np.median(fitness)
	f_mean = np.mean(fitness)
	f_max = np.amax(fitness)
	f_min = np.amin(fitness)


	# generate next population

	print "Gen: {i}, size: {s}, median: {f_med}, best: {best}, worst: {worst}".format(i=i, s=len(pop), f_med=f_med, best=f_min, worst=f_max)
	pop = GA.breed(pop, fit_func, GA.MINIMISE, Individual, NC, N, targets, mu=mu, f_limit=f_limit)
	
	# cull 
for i in pop:
	print i
