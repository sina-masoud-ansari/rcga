import numpy as np
import sys

class GA:

	def __init__(self, pop, mpr, cr, rr):
		"""
		pop is a list of numpy float vectors
		mpr is the mating pool ratio
		cr is the offspring ratio
		rr is the replacement ratio
		"""
		# Population and control
		self.pop = pop
		self.matingPoolRatio = mpr
		self.offspringRatio = cr
		self.replacementRatio = rr

		if self.pop == None:
			print "No population provided"
			sys.exit(0)
		
		# GA strategies
		self.fitnessFunction = None
		self.selectionMethod = None
		self.crossoverMethod = None
		self.mutationMethod = None
		self.replacementMethod = None

	# Control
	def run(self, n):

		if self.fitnessFunction == None \
								or self.selectionMethod == None \
		  	 					or self.crossoverMethod == None \
		   						or self.mutationMethod == None \
		   						or self.replacementMethod == None:
			print "Need to set up GA methods"
			sys.exit(0)
		
		print "Initial population:"
		for p in self.pop:
			print p

		# Time independent selection and replacement groups
		nparents = int(self.matingPoolRatio * len(self.pop)) # size of mating pool
		nchildren = int(self.offspringRatio * len(self.pop)) # number of offspring
		nreplace = int(self.replacementRatio * len(self.pop)) # number of original pop to replace

		# Start loop
		for i in range(0, n):
			#fitness = self.evaluateFitness()
			parents = self.selectParents(nparents)
			children = self.createChildren(parents, nchildren)
			children = self.mutate(children)
			self.pop = self.replace(self.pop, parents, children, nreplace)

			#print "Generation: ", i+1
			#for p in self.pop:
			#	print p
		print "End"
		for p in self.pop:
			print p

	# Fitness
	def setFitnessFunction(self, f):
		self.fitnessFunction = f

	def evaluateFitness(self):
		fitness = []
		for p in self.pop:
			fitness.append((p, self.fitnessFunction(p)))
		return fitness

	@staticmethod
	def RandomPopulation(size, length):
		# TODO: make this take a mean and stdev
		pop = []
		for i in range(0, size):
			pop.append(np.random.normal(size=length))
		return pop

	# Selection Methods
	@staticmethod
	def BinaryTournament(p, f, n, *args, **kwargs):	
		return GA.QTournament(p, f, n, q=2, **kwargs)

	@staticmethod
	def QTournament(p, f, n, q=0, **kwargs):
		"""
		Local q-tournament selection
		0 <= q <= (n - 1)
		0 = random (no selection pressure)
		q typically between 6 and 10
		q >= 11 considered hard selection
		f is fitness function
		n is size of intermediate population
		"""

		# Check q
		if q < 0 or q > (len(p) -1):
			print "For q-tournament selection, 0 <= q <= (n-1)"
			sys.exit(0)

		# Options
		replacement = kwargs.pop('replacement', True)
		stochastic = kwargs.pop('stochastic', False)
		if stochastic == True:
			print "Stochastic replacement not supported yet"
			sys.exit(0)

		# Begin tournament
		selected = []
		indices = range(0, len(p))
		for t in range(0, n):
			choices = np.random.choice(indices, q, replace=replacement)
			competitors = [p[x] for x in choices]
			fitness = np.array([f(x) for x in competitors])
			best = competitors[np.argmax(fitness)]
			selected.append(best)

		return selected

	def setSelectionMethod(self, s, *args, **kwargs):
		self.selectionMethod = s
		self.selectionMethod_args = args
		self.selectionMethod_kwargs = kwargs

	def selectParents(self, n):
		args = self.selectionMethod_args
		kwargs = self.selectionMethod_kwargs
		return self.selectionMethod(self.pop, self.fitnessFunction, n, *args, **kwargs)

	# Crossover Methods
	@staticmethod
	def DiscreteCrossover(P, n, p=0.5, **kwargs):
		"""
		Discrete crossover with two parents
		P is the population of parents
		n is the number of offspring to produce
		p is probability of gene selection from each parent
		"""
		children = []
		indices = range(0, len(P))
		while len(children) < n:
			choices = np.random.choice(indices, 2, replace=True)
			parents = [P[x] for x in choices]
			parent_a = parents[0]
			parent_b = parents[1]
			# TODO: assuming parents of same length
			length = len(parent_a)
			child_a = np.zeros(length)
			child_b = np.zeros(length)
			for i in range(0, length):
				r = np.random.rand()
				if r < p:
					child_a[i] = parent_b[i]
					child_b[i] = parent_a[i]
				else:
					child_a[i] = parent_a[i]
					child_b[i] = parent_b[i]

			#print "parent a: ",parent_a
			#print "parent b: ",parent_b
			#print "child a: ",child_a
			#print "child b: ",child_b

			r = np.random.rand()
			if r < 0.5:
				children.append(child_a)
				if len(children) < n:
					children.append(child_b)
			else:
				children.append(child_b)
				if len(children) < n:
					children.append(child_a)

		return children
							
	def setCrossoverMethod(self, m, *args, **kwargs):
		self.crossoverMethod = m
		self.crossoverMethod_args = args
		self.crossoverMethod_kwargs = kwargs

	def createChildren(self, parents, n):
		args = self.crossoverMethod_args
		kwargs = self.crossoverMethod_kwargs
		return self.crossoverMethod(parents, n, *args, **kwargs)
		
	# Mutation methods

	@staticmethod
	def NormalUniformMutation(P, p=0.01, a=1.0, **kwargs):
		for member in P:
			for i in range(0, len(member)):
				r = np.random.rand()
				if r < p:
					member[i] += a * np.random.normal()
		return P

	def setMutationMethod(self, m, *args, **kwargs):
		self.mutationMethod = m
		self.mutationMethod_args = args
		self.mutationMethod_kwargs = kwargs

	def mutate(self, pop):
		args = self.mutationMethod_args
		kwargs = self.mutationMethod_kwargs
		return self.mutationMethod(pop, *args, **kwargs)

	# Replacement Methods

	@staticmethod
	def RandomReplacement(P, parents, children, n):
		"""
		Select n individuals randomly from children to replace n random
		members from the original population
		"""
		# Select n  children 
		indices = range(0, len(children))
		choices = np.random.choice(indices, n, replace=False)
		individuals = [children[x] for x in choices]

		# Select original members
		indices = range(0, len(P))
		choices = np.random.choice(indices, n, replace=False)
		
		# Replace original members with new individuals
		for i in range(0, n):
			index = choices[i]
			#print "Replacing ",P[index]," with ",individuals[i]
			P[index] = individuals[i]

		return P

	def setReplacementMethod(self, m, *args, **kwargs):
		self.replacementMethod = m
		self.replacementMethod_args = args
		self.replacementMethod_kwargs = kwargs

	def replace(self, pop, parents, children, n):
		args = self.replacementMethod_args
		kwargs = self.replacementMethod_kwargs
		return self.replacementMethod(pop, parents, children, n, *args, **kwargs)


def f1(x):
	y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
	#y = np.array([0.1])
	dist = np.linalg.norm(y - x)
	return np.exp(-dist)


def main():
	print "GA with real valued encoding"

	# Population setup
	size = 200
	length = 5
	pop = GA.RandomPopulation(size, length)
	mpr = 0.3 # mating pool ratio
	cr = 0.3 # offspring ratio
	rr = 0.3 # replacement ratio

	# GA setup
	ga = GA(pop, mpr, cr, rr)
	ga.setFitnessFunction(f1)
	ga.setSelectionMethod(GA.BinaryTournament)
	ga.setCrossoverMethod(GA.DiscreteCrossover)
	ga.setMutationMethod(GA.NormalUniformMutation, 0.05, 0.01)
	ga.setReplacementMethod(GA.RandomReplacement)

	# Start GA
	N =200
	ga.run(N)


if __name__ == "__main__":
	main()
