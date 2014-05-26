import bitstring as bs
import numpy as np
import os
import sys

class GA:
	"""
	Genetic Algorithm
	"""

	MINIMISE = 0
	MAXIMISE = 1

	def __init__(self, pop, fitfunc, weights=None):
		self.pop = pop	# The population of bitstrings
		self.size = len(self.pop)
		self.maxsize = len(self.pop)
		self.fitfunc = fitfunc # The fitness [[function, searchtype]]
		self.nfunc = len(self.fitfunc)
		if weights == None:
			self.weights = [1.0/self.nfunc] * self.nfunc
		self.best = []

	def padOrTrim(self, a, b):

		len_a = len(a)
		len_b = len(b)

		# a is ref
		if len_a > len_b:
			# b is shorter so pad it
			diff = len_a - len_b
			values = '0b' + '0' * diff 
			b.append(values)
		else:
			# a is shorter so trim b
			b = b[0:len_a]

		return a, b
	
	def align(self, a, b):
		# aligns bitstrings a and b if they are of different length
		if len(a) != len(b):
			len_a = len(a)
			len_b = len(b)
			# Randomly choose one as the reference
			if np.random.rand() < 0.5:
				a, b = self.padOrTrim(a, b)
			else:
				b, a = self.padOrTrim(b, a)

		return a, b

	def crossover(self, sa, sb):
		#cp = np.random.randint(len(sa)) # used for single point crossover

		# generate mask for uniform crossover
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

	def mutate(self, s, p):
		"""
		Randomly alter bitstring elements according to fixed probability p
		"""
		sm = bs.BitStream(s)
		r = np.random.rand()
		if r < p:
			i = np.random.randint(len(sm))
			sm.invert(i)
		#for i in range(0, len(s)):
		#	r = np.random.rand()
		#	if r < p:
		#		sm.invert(i)
		return sm
	


	def mate(self):
		npop = []
		mrate = 0.01
		if self.stdev_fitness < 1:
			mrate = mrate * 10
		while len(npop) < self.maxsize:
			#roulette wheel selection
			i = np.random.randint(self.size) # random individual i
			rank_i = self.poprank[i]
			#if np.amax(self.poprank) < 0.1:
			#	rank_i = 1.0
			#	mrate = 0.2
				#print "Poor ranks, so everyone is 1.0"
			pi = np.random.rand() # fitness proportionate selection
			if pi < rank_i:
				j = np.random.randint(self.size) # random individual j
				#print "Selected individual {i} with rank {r} to mate with {j}".format(i=i, r=rank_i, j=j)
				#while (i == j):
					# make sure i is not j
				#	j = np.random.randint(self.size) # random individual j
				# end while
				#rank_j = self.poprank[j]
				#pj = np.random.rand() # fitness proportionate selection
				#if pj < rank_j:
				a, b = self.align(self.pop[i], self.pop[j])
				c = self.crossover(a, b)
				c = self.mutate(c, mrate)
				npop.append(c)	
		# end while
		self.pop = npop
		self.size = len(self.pop)

	def rank(self):
		"""
		Create individual, fitness tuple sorted by fitness ascending
		Set sorted individual ranks
		"""
		# find the min and range in each column
		min_ = np.zeros(self.nfunc)	
		max_ = np.zeros(self.nfunc)	
		range_ = np.zeros(self.nfunc)	
		for i in range(0, self.nfunc):
			u = self.fitness[:,i]
			min_[i] = np.amin(u)
			max_[i] = np.amax(u)
			range_[i] = np.amax(u) - min_[i] + 1e-7

		# compute normalised weighted fitness for each individual
		self.poprank = np.zeros(self.size)
		for i in range(0, self.size):
			u = self.fitness[i, :]
			v = np.zeros(self.nfunc)
			for j in range(0, self.nfunc):
				searchtype = self.fitfunc[j][2]
				#ratio = np.log( ((u[j] - min_[j]) / range_[j]) + 1e-7 )
				#ratio = (u[j] - min_[j]) / range_[j]
				ratio = u[j] / max_[j]
				if searchtype == GA.MAXIMISE:
					v[j] = ratio 
				elif searchtype == GA.MINIMISE:
					v[j] = 1 - ratio 
			self.poprank[i] = np.dot(v, self.weights) / self.nfunc

		#for i in range(0, self.size):
		#	print self.fitness[i], self.poprank[i]

		# normalise rank vector
		self.poprank =  (self.poprank - np.min(self.poprank)) / (np.amax(self.poprank - np.amin(self.poprank)) + 1e-7)

		#for i in range(0, self.size):
		#	print self.fitness[i], self.poprank[i]

		#tmp = []
		#for b in np.where(self.poprank > 0.99):
		#	tmp.append(self.fitness[b])

		best = np.where(self.poprank > 0.99)
		if len(best[0]) > 0:
			self.best.append(self.pop[best[0][0]])


	def run(self, N, verbose):
		self.generation = 0
		self.checkFitness()
		if verbose:
			print self.summary()
		for i in range(0, N):
				#self.filter()
				if self.size == 0:
					print "Pop crashed!"
					sys.exit(0)
				self.rank()
				self.mate()
				self.generation += 1
				self.checkFitness()
				if verbose:
					print self.summary()
			
	def filter(self):
		"""
		Remove individuals with fitness >= FAIL
		"""
		todel = []
		for i in range(0, self.nfunc):
			col = self.fitness[:,i]
			fail = self.fitfunc[i][1]
			mask = np.where( col >= fail)
			for m in mask:
				todel += m.tolist()
		todel = set(todel)
		tosave = set(range(0, self.size)) - todel

		fn = []
		pn = []
		for i in tosave:
			fn.append(self.fitness[i])
			pn.append(self.pop[i])
		fn = np.array(fn)
		self.fitness = fn
		self.pop = pn
		self.size = len(pn)

		#print self.fitness
		
	def checkFitness(self):
		"""
		!!! Fitness functions are assumed to return a value between [0, +inf)
		"""
		self.fitness = np.zeros((self.size, self.nfunc))

		# Row order access
		for i in range(0, self.size):
			for j in range(0, self.nfunc):
				func = self.fitfunc[j][0]
				fail = self.fitfunc[j][1]
				self.fitness[i][j] = func(self.pop[i], fail)
	
	def checkStats(self):
		self.max_fitness = np.max(self.fitness, axis=0)
		self.min_fitness = np.min(self.fitness, axis=0)
		self.mean_fitness = np.mean(self.fitness, axis=0)
		self.stdev_fitness = np.std(self.fitness, axis=0)
		self.median_fitness = np.median(self.fitness, axis=0)

	def summary(self):
		"""
		Print some summary info about the GA
		"""
		self.checkStats()
		s = """
---------------------------------------------------------
Generation: {g}
Population size: {s}
Weights: {w}
Max Fitness: {maxf}
Min Fitness: {minf}
Mean Fitness: {meanf}
Standard Deviation: {stdevf}
Median Fitness: {medianf}
---------------------------------------------------------
""".format(
			s=self.size,
			g=self.generation,
			#best=self.fitness[0],
			#worst=self.fitness[-1],
			maxf=self.max_fitness,
			minf=self.min_fitness,
			meanf=self.mean_fitness,
			stdevf=self.stdev_fitness,
			medianf=self.median_fitness,
			w=self.weights
			)

		return s
	
	@staticmethod
	def randomBitString(length):
		"""
		Return random bitstring with length bits
		"""
		bytes = length / 8
		remainder = length % 8
		if remainder != 0:
			bytes += 1
		s = bs.BitArray(bytes=os.urandom(bytes), length=8*bytes)
		s = s[:length]
		return s
	

	@staticmethod
	def randomPopulation(size, length):
		"""
		Create random population of bitstrings
		"""
		pop = []
		for i in range(0, size):
			pop.append(GA.randomBitString(length))
		return pop

	
def fitness_2(s, FAIL):
	#require positive integer
	#FAIL = np.iinfo(np.int32).min
	DEFN =  '2 * float:32, int:32'
	values = np.array(s.unpack(DEFN))
	if np.isfinite(values).all():
		value = values[-1]
		if value < 0:
			return FAIL
		else:
			return value
	else:
		return FAIL
	

def fitness_1(s, FAIL):
	"""
	Parse bitsting and check for fitness of individual
	Parsing requires a definition:
		see: https://pythonhosted.org/bitstring/reading.html
	"""
	#targets = np.array([[7, 1000]])
	targets = np.array([[5,5]])
	#DEFN =  '2 * float:32, int:32'
	DEFN =  '2 * float:32'
	values = np.array(s.unpack(DEFN))
	if np.isfinite(values).all():
		#values = values[:-1]
		values = np.array(values)
		fitness = []
		for t in targets:
			if len(values) != len(t):
				fitness.append(FAIL)
			else:
				fitness.append(np.linalg.norm(values - t))
				#print values, fitness
		fitness = np.mean(np.array(fitness))
	else:
		fitness = FAIL
	return fitness



def parse(s, DEFN):
	values = np.array(s.unpack(DEFN))
	return values


def test():
	size = 10
	length = 2 * 32
	DEFN = '2 * float:32'
	pop = GA.randomPopulation(size, length)
	for p in pop:
		print p, parse(p ,DEFN)


def main():
	print "GA setup"
	gen = 1000
	size = 300
	#length = 3 * 32
	length = 2 * 32
	DEFN = '2 * float:32'
	#F = [[fitness_1, np.finfo(np.float32).max, GA.MINIMISE], [fitness_2, 0, GA.MAXIMISE]]
	F = [[fitness_1, 1e2, GA.MINIMISE]]
	pop = GA.randomPopulation(size, length)
	ga = GA(pop, F)
	ga.run(gen, verbose=True)
	#for b in ga.best:
	#	print parse(b, DEFN)
	res = []
	for p in ga.pop:
		v = parse(p, DEFN)
		err = v - np.array([3,3])
		dist = np.linalg.norm(err)
		res.append((v.tolist(), err.tolist(), dist))
	res.sort(key=lambda x: x[2], reverse=True)
	for v,e,d in res:
		print v,d


if __name__=='__main__':
	main()
	#test()
