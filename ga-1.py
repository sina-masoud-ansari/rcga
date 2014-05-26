import bitstring as bs
import numpy as np
import os
import sys

class Reservoir:
	def __init__(self, N=None, bitstring=None):
		if bitstring == None:
			if N == None:
				sys.stderr.out("Expected size in call to Reservoir constructor\n")
				sys.exit(1)
			else:
				self.size = N
				self.M = np.random.randn(N, N)
				self.v = np.random.randn(N) * np.random.randint(1, 10, 1)
		else:
			self.from_bitstring(bitstring)

		self.result = None

	def test(self):
		self.result = np.dot(self.M, self.v)

	def __str__(self):
			return "M:"+str(self.M)+"\nv:"+str(self.v)+"\nresult:"+str(self.result)
	
	def packMatrix(self, M):
		s = bs.BitStream()
		# dimensions of matirx
		if len(M.shape) == 1:
			M = M.reshape((len(M), 1))
		s.append(bs.pack('2*uint:4', M.shape[0], M.shape[1]))
		# matrix elements
		for i in range(0, M.shape[0]):
			for j in range(0, M.shape[1]):
				s.append(bs.pack('float:32', self.M[i][j]))
		return s

	def to_bitstring(self):
		s = bs.BitStream()
		s.append(bs.pack('uint:4', self.size))
		s.append(self.packMatrix(self.M))
		s.append(self.packMatrix(self.v))
		return s

	def unpackMatrix(self, s):
		m = s.read('unint:4')
		n = s.read('unint:4')
		M = np.ndarray(shape=(m, n), dtype=float)
		for i in range(0, m):
			for j in range(0, n):
				M[i][j] = s.read('float:32')
		return M

	def from_bitstring(s):
		try: 
			self.size = s.read('unint:4');
			self.M = self.unpackMatrix(s)	
			self.v = self.unpackMatrix(s)
		except:
			# mark as invalid
			self.size = None 
	
	def checkValues(self, x):
		return not np.isnan(x).any() and not np.isinf(x).any()

	def isvalid(self):
		if self.size == None:
			return False
		else:
			# check for NaN, -inf, +inf
			return 	checkValues(self.size) and checkValues(self.M) and checkValues(self.v)

class GA:

	# Fitness search strategy
	MINIMISE = 0
	MAXIMISE = 1
	SEARCH_TYPES = [MINIMISE, MAXIMISE]

	def __init__(self, P):
		self.pop = zip(P, [None] * len(P)) # individuals with no fitness
		self.npop = None # Will be initialised in call to breed
		self.G = None # Will be initialised during call to run
		self.f = None # Will be initialised during call to run
		self.fmean = None # Will be initialised during call to run
		self.fstdev = None # Will be initialised during call to run
		self.targets = None # Will be initialised during setTargets
	
	def __str__(self):
		return self.vb1()

	def printga(self, verbosity):
		if verbosity == 1:
			print self.vb1()	
		elif verbosity == 2:
			print self.vb2()

	def vb1(self):
		s = "Generation: "+str(self.G)+"\n"
		s += "Population size: "+str(len(self.pop))+"\n"
		s += "Fitness: m: "+str(self.fmean)+", stdev: "+str(self.fstdev)+"\n"
		return s

	def vb2(self):
		s = self.vb1()
		for i, f in self.pop:
			s += str(i)+"\nf:"+str(f)+"\n\n"
		return s

	def eval(self, ffunc):
		# evalute each individual according to ffunc and targets
		if self.targets == None:
			sys.stderr.out("No targets set\n")
			exit(1)
		self.pop = [(p[0], ffunc(p[0], self.targets, self.flimit)) for p in self.pop]
		# update stats
		pop, fitness = zip(*self.pop)
		self.fmean = np.mean(fitness)
		self.fstdev = np.std(fitness)

	def padOrTrim(self, a, b):

		len_a = len(a)
		len_b = len(b)

		# a is ref
		if len_a > len_b:
			# b is shorter so pad it
			diff = len_a - len_b
			values = '0b' + '0' * diff 
			#defn = str(diff) + '* byte'
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
		for i in range(0, len(s)):
			r = np.random.rand()
			if r < p:
				sm.invert(i)
		return sm
		
	def mate(self, a, b):
		sa = a.to_bitstring()	
		sb = b.to_bitstring()	
		sa, sb = self.align(sa, sb)
		# TODO crossover and mutation
		sc = self.crossover(sa, sb)
		sc = self.mutate(sc, 0.005)
		return sc

	def breed(self, rank):
		self.npop = []
		len_rank = len(rank)
		for i in range(0, len_rank):
			for j in range(0, len_rank):
				if i != j:
					rnd = np.random.rand()
					if rnd < rank[i]:
						m = self.pop[i][0]
						f = self.pop[j][0]
						c = self.mate(m, f)
						self.npop.append(c)

	def update(self):
		# sort by fitness, ascending order
		self.pop.sort(key=lambda x: x[1])
		worst = None
		if self.type == GA.MINIMISE:
			worst = self.pop[-1]
		elif self.type == GA.MAXIMISE:
			worst = self.pop[0]
		rank = []
		for i, f in self.pop:
			# may need to consider log proportionate rank as large values will bias
			if self.type == GA.MINIMISE:
				r = 1 - f / worst[1]
			elif self.type == GA.MAXIMISE:
				r = 1 - worst[1] / f
			rank.append(r)
		self.breed(rank)

	def run(self, G, ffunc, verbosity=1):
		self.G = 0
		for g in range(0, G):
			# run the test of the individual
			for p, f in self.pop:
				p.test()
			# evaluate individuals (need to flag those that are failures)
			self.eval(ffunc)
			self.update()
			# print status
			if verbosity != 0:
				self.printga(verbosity)
			self.G += 1
	
	def setTargets(self, T):
		self.targets = T
	
	def setSearchType(self, type, flimit):
		if type not in GA.SEARCH_TYPES:
			sys.stderr.out("Unkown search type '{t}'\n".format(t=type))
			sys.exit(1)
		else:
			self.type = type
		self.flimit = flimit

def create_pop(N, n):
	# where N is pop size, n is reservoir size
	pop = []
	for i in range(0, N):
		pop.append(Reservoir(n))
	return pop

def ffunc(i, targets, flimit):
	# valuate fitness of individual i
	fitness = []
	r = i.result
	for t in targets:
		if len(r) != len(t):
			fitness.append(flimit)
		else:
			fitness.append(np.linalg.norm(t - r))

	return np.mean(fitness)


def main():
	print 'Starting GA program'
	G = 5
	N = 10
	print 'Generations: {G}, size: {N}'.format(G=G, N=N)

	print 'Creating initial populaiton'
	# create population
	n = 2
	#targets = [[15, 15, 15, 15]]
	targets = [[15, 15]]
	# TODO test with multiple targets
	pop = create_pop(N, n)
	pop = [Reservoir(2), Reservoir(3)] #TODO: create random sized pop
	ga = GA(pop)
	ga.setTargets(targets)
	ga.setSearchType(GA.MINIMISE, 100)
	ga.run(G, ffunc, verbosity=1)
	#ga.run(G, ffunc, GA.MAXIMISE, verbosity=1)

def testCrossover():
	pass

if __name__=='__main__':
	main()
