import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import ConfigParser
import os.path

from rc import *
from ga import *

class Options():
	def __init__(self, options_file):
		if os.path.isfile(options_file):
			self.cfg = ConfigParser.ConfigParser()
			self.cfg.read(options_file)
			self.process_sections()
		else:
			print "Configuration file '%s' not found" % options_file
			sys.exit(1)

	def process_sections(self):
		self.process_population()
		self.process_training()
		self.process_run()

	def process_population(self):
		# Population section
		section = 'Population'
		if section not in self.cfg.sections():
			print "Configuraiton file needs to include a '%s' section" % section
			sys.exit(1)
		else:
			self.population_size = int(self.cfg.get(section, 'population_size'))
			self.mating_pool_ratio = float(self.cfg.get(section, 'mating_pool_ratio')) 
			self.offspring_ratio = float(self.cfg.get(section, 'offspring_ratio')) 
			self.replacement_ratio = float(self.cfg.get(section, 'replacement_ratio')) 
			self.mutation_rate =  float(self.cfg.get(section, 'mutation_rate')) 
			self.mutation_scale = float(self.cfg.get(section, 'mutation_scale')) 

	def process_training(self):
		# Training section
		section = 'Training'
		if section not in self.cfg.sections():
			print "Configuraiton file needs to include a '%s' section" % section
			sys.exit(1)
		else:
			self.sample_length = int(self.cfg.get(section, 'sample_length'))
			self.forecast_length= int(self.cfg.get(section, 'forecast_length')) 
			self.num_samples = int(self.cfg.get(section, 'num_samples')) 
			self.test_ratio = float(self.cfg.get(section, 'test_ratio')) 

	def process_run(self):
		# Training section
		section = 'Run'
		if section not in self.cfg.sections():
			print "Configuraiton file needs to include a '%s' section" % section
			sys.exit(1)
		else:
			self.set_seed = self.cfg.get(section, 'set_seed') == 'True'
			self.seed = int(self.cfg.get(section, 'seed')) 
			self.checkpoint = self.cfg.get(section, 'checkpoint') == 'True'
			self.checkpoint_period = int(self.cfg.get(section, 'checkpoint_period')) 
			self.validate = self.cfg.get(section, 'validate') == 'True'
			self.validation_period = int(self.cfg.get(section, 'validation_period')) 
			self.num_generations = int(self.cfg.get(section, 'num_generations')) 
			self.plot = self.cfg.get(section, 'plot') == 'True' 

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
	print "Creating initial population of size {n} ...".format(n=n),
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
	mape = Util.validationMP(res, samples, forecast_length)
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
	np.random.shuffle(samples)
	return samples

def loadRandomSamples(file, length, forecast, nsamples):
	# sample controls
	sample_length = 200
	forecast_length = 5
	samples = []

	# load input data
	with open(file, "r") as f:
		for filename in f:
			u = np.loadtxt(filename.strip())
			samples += Util.createRandomSamples(u, length, forecast, nsamples)
	f.close()
	print "Total samples: ",len(samples)
	return samples

def partitionSamples(samlpes, test_ratio):
	n_test = int(test_ratio * len(samples))
	test = samples[:n_test]
	train = samples[n_test:]
	print "Num train: {ntr}, Num test: {ntst}".format(ntr=len(train), ntst=len(test))
	return train, test

def init_ga(pop, train, options):
	# GA setup
	ga = GA(pop, options.mating_pool_ratio, options.offspring_ratio, options.replacement_ratio)
	ga.setFitnessFunction(reservoirFitness, train, options.forecast_length)
	ga.setSelectionMethod(GA.BinaryTournament)
	ga.setCrossoverMethod(GA.DiscreteCrossover)
	ga.setMutationMethod(GA.NormalUniformMutation, options.mutation_rate, options.mutation_scale)
	ga.setReplacementMethod(GA.RandomReplacement)
	return ga

def validate(ga, train, test, options):
	res = fromDescription(ga.best)
	e_train = Util.validationMP(res, train, options.forecast_length)
	e_test = Util.validationMP(res, test, options.forecast_length)
	print "Best MAPE: Train error: {tr}, Test error: {tst}".format(tr=e_train, tst=e_test)

def save(prefix, ga, train, test):
	# Save output files
	pop_file = '%s.p' % prefix
	train_file = '%s.train' % prefix
	test_file = '%s.test' % prefix
	ga.save(pop_file)
	pickle.dump(train, open(train_file, "wb"))
	pickle.dump(test, open(test_file, "wb"))

def analyse(pop, samples, options):
	# find best
	res = fromDescription(ga.best)
	Util.validation(res, samples, options.forecast_length, plot=True)

def parse_args():
	parser = argparse.ArgumentParser(description="Reservoir Computing with Genetic Algorithm")
	parser.add_argument('-i', dest='input', action='store', default='rcga.input', metavar='INPUT_FILE_LIST', help='File containing newline separated list of files to be used as input. Input files are single value, newline separated.')
	parser.add_argument('-c', dest='config', action='store', default='rcga.ini', metavar='CONFIGURATION_FILE', help=' Holds all initialisation options for rcga in INI format')
	parser.add_argument('-a', dest='analyse', action='store_true', default=False, help='Analyse the results of training on test set after GA has finished')
	parser.add_argument('-o', dest='output', action='store', default='output', metavar='OUTPUT_PREFIX', help='Will save OUTPUT_PREFIX.p, OUTPUT_PREFIX.train, OUTPUT_PREFIX.test and OUTPUT_PREFIX.ini. Default value for OUTPUT_PREFIX is \'output\'')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-r', dest='resume', action='store', default='', metavar='OUTPUT_PREFIX', help='Option to resume from a previous run, loading OUTPUT_PREFIX.p, OUTPUT_PREFIX.train and OUTPUT_PREFIX.test with options specified in OUTPUT_PREFIX.ini')
	group.add_argument('--analyse-output', dest='analyse_output', action='store', default='', metavar='OUTPUT_PREFIX', help='Analyse output for a specific run given by OUTPUT_PREFIX. This will load OUTPUT_PREFIX.p, OUTPUT_PREFIX.train, OUTPUT_PREFIX.test and OUTPUT_PREFIX.ini')
	return parser.parse_args()

if __name__ == '__main__':

	# Parse command line arguments
	args = parse_args()
	start_new = args.resume == '' and args.analyse_output == ''

	# Load configuration, population and samples
	if start_new:
		# Initialise new 
		prefix = 'output'
		options = Options(args.config)
		if options.set_seed:
			np.random.seed(options.seed)
		pop = createReservoirs(options.population_size)
		pop = popToDescription(pop)
		# Load training and test data
		samples = loadRandomSamples(args.input, options.sample_length, options.forecast_length, options.num_samples)
		train, test = partitionSamples(samples, options.test_ratio)
	else:
		# Determine prefix
		if args.resume != '': 
			prefix = args.resume
		else:
			prefix = args.analyse_output
		# Resume from previous
		config_file = '%s.ini' % prefix
		pop_file = '%s.p' % prefix
		train_file = '%s.train' % prefix
		test_file = '%s.test' % prefix
		options = Options(config_file)
		pop = GA.load(pop_file)
		train = pickle.load( open( train_file, "rb" ) )
		test = pickle.load( open( test_file, "rb" ) )

	# If Analyse only (--analyse-output)
	if args.analyse_output != '':
		analyse(pop, test, options)	
		sys.exit(0)

	# Initialise GA
	checkfile = '%s.p' % prefix
	ga = init_ga(pop, train, options)
	# Run GA
	for i in range(options.validation_period):
		ga.run(options.num_generations, checkpoint=options.checkpoint, checkperiod=options.checkpoint_period, checkfile=checkfile)
		validate(ga, train, test, options)

	# Save output files
	prefix = args.output
	save(prefix, ga, train, test)

	# Analyse if requried (-a)
	if args.analyse:
		analyse(pop, test, options)	
