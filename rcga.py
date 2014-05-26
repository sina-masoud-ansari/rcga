import numpy as np
import matplotlib.pyplot as plt
import sys

from rc import *
from ga import *

def testDescriptionTransform():
	r1 = Reservoir(3)
	desc = toDescription(r1)
	r2 = fromDescription(desc)
	print "size: ", r1.size == r2.size
	print "spec_rad: ", r1.spectral_radius == r2.spectral_radius 
	print "density: ", r1.density == r2.density 
	print "washout: ", r1.washout == r2.washout
	print "W_rr: ", np.allclose(r1.W_rr, r2.W_rr) 
	print "W_or: ", np.allclose(r1.W_or, r2.W_or)
	print "W_rr_density: ", r1.W_rr_density == r2.W_rr_density
	print "W_or_density: ", r1.W_or_density == r2.W_or_density 
					

def fromDescription(d):
	# Create a reservoir from a vector of reals
	d = d.tolist()
	_size = d.pop(0) # real size	
	size = int(_size) # integer size
	spectral_radius = d.pop(0)	
	density = d.pop(0)	
	washout = d.pop(0)
	# need to determine integer size from number of remaining elements
	# number of remaining elements is size*size + size + 2
	coeff = [1, 1, (2-len(d))]
	roots = np.roots(coeff)
	old_size = int(roots[np.where(roots > 0)][0])
	W_rr = None
	W_or = None
	if size <= old_size:
		# new size is equal to or smaller, skip the excess
		W_rr = np.array(d[:size*size]).reshape(size, size)
		d = d[old_size*old_size:]
		W_or = np.array(d[:size]).reshape(size, 1)
		d = d[old_size:]
	else:
		# new size is larger, add some random elements
		diff1D = size - old_size
		diff2D = size*size - old_size*old_size
		arr = d[:old_size*old_size] + np.random.rand(diff2D).tolist()
		W_rr = np.array(arr).reshape(size, size)
		d = d[old_size*old_size:]
		arr = d[:old_size] + np.random.rand(diff1D).tolist()
		W_or = np.array(arr).reshape(size, 1)
		d = d[old_size:]

	W_rr_density = d.pop(0)
	W_or_density = d.pop(0)

	res = Reservoir(_size, spectral_radius=spectral_radius, density=density, \
						washout=washout, W_rr=W_rr, W_or=W_or, \
						W_rr_density=W_rr_density, W_or_density=W_or_density)
	return res

def popFromDescription(descv):
	return [fromDescription(d) for d in descv]

def toDescription(res):
	# Create a vector of reals representation of a resevoir
	desc = []
	desc.append(res._size)	
	desc.append(res.spectral_radius)	
	desc.append(res.density)	
	desc.append(res.washout)	
	desc += [elem for row in res.W_rr for elem in row]
	desc += [elem for elem in res.W_or]
	desc.append(res.W_rr_density)
	desc.append(res.W_or_density)

	return np.array(desc)

def popToDescription(popv):
	return [toDescription(p) for p in popv]

def createReservoirs(n):
	print "Creating initial population ... ",
	reservoirs = []
	while len(reservoirs) < n:
		try:
			size = int(np.absolute(100 + np.random.randn() * 100))
			spectral_radius = np.random.rand()
			density = np.random.rand()
			washout = np.random.rand()
			W_rr_density = np.random.rand()
			W_or_density = np.random.rand()
			res = Reservoir(size, spectral_radius=spectral_radius, density=density, \
							washout=washout, W_rr_density=W_rr_density, \
							W_or_density=W_or_density)
		except ReservoirError:
			continue
		reservoirs.append(res)

	print "done"
	return reservoirs

def reservoirFitness(desc, samples, forecast_length):
	res = None
	try:
		res = fromDescription(desc)
	except ReservoirError:
		return 0.0
	mape = Util.validation(res, samples, forecast_length, plot=False)
	# values close to zero have fitness near 1
	#print mape, np.exp(-mape)
	return np.exp(-mape)

def loadSamples(file, length, forecast):
	# sample controls
	sample_length = 200
	forecast_length = 5
	samples = []

	# load input data
	with open(file, "r") as f:
		for filename in f:
			u = np.loadtxt(filename.strip())
			samples += Util.createSamples(u, length, forecast)
	f.close()
	return samples



def analyse():
	# load input data
	fname = sys.argv[1] # input file
	u = np.loadtxt(fname)

	# sample controls
	sample_length = 200
	forecast_length = 5
	samples = Util.createSamples(u, sample_length, forecast_length)
	
	# population and properties
	pop = GA.load()
	pop = popFromDescription(pop)
	size = [p.size for p in pop]
	spec_rad = [p.spectral_radius for p in pop]

	# find best
	psort = [(res, Util.validation(res, samples, forecast_length, plot=False)) for res in pop]
	psort = sorted(psort, key=lambda x: x[1]) # ascending order
	worst = psort[len(psort)-1]
	best = psort[0]
	print "Best: ", best[1]
	Util.validation(best[0], samples, forecast_length, plot=True)
	print "Worst: ", worst[1]
	Util.validation(worst[0], samples, forecast_length, plot=True)


def resume():
	# load input data
	fname = sys.argv[1] # input file
	u = np.loadtxt(fname)

	# sample controls
	sample_length = 100
	forecast_length = 5
	samples = Util.createSamples(u, sample_length, forecast_length)

	# resume
	pop = GA.load()

	# spores
	#spore_ratio = 0.25
	#spores = createReservoirs(spore_ratio * len(pop))
	#spores = popToDescription(spores)

	#GA stuff
	mpr = 0.3 # mating pool ratio
	cr = 0.3 # offspring ratio
	rr = 0.3 # replacement ratio

	# GA setup
	ga = GA(pop, mpr, cr, rr)
	ga.setFitnessFunction(reservoirFitness, samples, forecast_length)
	#ga.addSpores(spores)
	ga.setSelectionMethod(GA.BinaryTournament)
	ga.setCrossoverMethod(GA.DiscreteCrossover)
	ga.setMutationMethod(GA.NormalUniformMutation, 0.05, 0.01) #0.05, 0.01
	ga.setReplacementMethod(GA.RandomReplacement)

	# Start GA
	N = 300
	ga.run(N, checkpoint=True, checkperiod=5)


def main():
	# sample controls
	sample_length = 200
	forecast_length = 5
	samples = loadSamples(sys.argv[1], sample_length, forecast_length)
	print "Total samples: ",len(samples)
	# create validation partition
	n_test = int(0.3 * len(samples))
	np.random.shuffle(samples)
	test = samples[:n_test]
	train = samples[n_test:]

	# create initial population
	pop_size = 100
	pop = createReservoirs(pop_size)
	pop = popToDescription(pop)

	# spores
	#spore_ratio = 0.25
	#spores = createReservoirs(spore_ratio * len(pop))
	#spores = popToDescription(spores)

	#GA stuff
	mpr = 0.3 # mating pool ratio
	cr = 0.3 # offspring ratio
	rr = 0.3 # replacement ratio

	# GA setup
	ga = GA(pop, mpr, cr, rr)
	ga.setFitnessFunction(reservoirFitness, train, forecast_length)
	#ga.addSpores(spores)
	ga.setSelectionMethod(GA.BinaryTournament)
	ga.setCrossoverMethod(GA.DiscreteCrossover)
	ga.setMutationMethod(GA.NormalUniformMutation, 0.05, 0.01) #0.05, 0.01
	ga.setReplacementMethod(GA.RandomReplacement)

	# Start GA
	N = 1
	n = 50
	for i in range(0, N):
		ga.run(n, checkpoint=True, checkperiod=5)
		res = fromDescription(ga.best)
		e_train = Util.validation(res, train, forecast_length, plot=False)
		e_test = Util.validation(res, test, forecast_length, plot=False)
		print "Train error: {tr}, Test error: {tst}".format(tr=e_train, tst=e_test)



if __name__ == '__main__':
	main()
	#resume()
	#analyse()
