import sys
import pickle
import numpy as np
import pylab as pl
from DataReader import *
from Signal import *
from Reservoir import *


def plotit(expected, result, states):
	# plotting
	f, axarr = pl.subplots(3, sharex=True)
	axarr[0].plot(expected)
	axarr[1].plot(expected)
	axarr[1].plot(result)
	axarr[2].plot(states)
	pl.show()
	
def create_samples(signals, n, length):
	samples = []
	for y in signals:
		starts = np.random.randint(0, len(y)-length, n)
		ends = starts + 100
		points = zip(starts, ends)
		for start, end in points:
			f = y[start:end]
			sig = Signal()
			#input = sig.prep(f)
			input = f
			output = f
			samples.append((sig, input, output))

	print "Created "+str(len(samples))+" samples"

	return samples

def create_signals(symbols, type):
	signals = []
	for s in symbols:
		y = s.retrieve(type)
		signals.append(y)

	print "Loaded "+str(len(signals))+" symbols"
		
	return signals

def create_pop(n):

	pop = [] # population
	for i in range(0, n):
		res = Reservoir(3, density=0.75, spec_rad=0.9, bias=0.0)
		pop.append(res)

	print "Created "+str(len(pop))+" reservoirs"
	return pop


def main():
	#sb = Symbol(sys.argv[1])
	#print "Reading %d records" % len(sb.ohlc)
	#high = sb.ohlc[:, 1]
	#s = Signal()
	#s_high = s.prep(high)[:100]
	#y = s_high
	
	# signals
	#pl.plot(s_high)
	#pl.show()

	# Create reservoir population
	pop = create_pop(1)

	# Read in a set of symbols
	#sym_list = ['ready/ANZ.AX.csv', 'ready/BNZ.AX.csv']
	#sym_list = ['ready/ANZ.AX.csv']
	#sym_list = ['sin.csv']
	sym_list = ['linear.csv']
	dr = DataReader(sym_list)
	
	# Convert these into signals
	signals = create_signals(dr.symbols, 'high')
	samples = create_samples(signals, 1, 100)
	
	# Train and test reservoirs on samples
	steps = 20
	washout = 30
	for res in pop:
		avg_error = 0.0
		for sig, input, output in samples:
			res.teach(input[:-steps])
			res.learn(washout=washout)
			result = res.test(input, steps)
			#result = sig.postp(result)
			error = res.smape(output, result)
			avg_error = avg_error + error
			plotit(output, result, res.states)
		avg_error = avg_error / len(samples)
		print "avg_error: "+str(avg_error)
 

	#Reservoir
	#steps = 20
	#res = Reservoir(40, density=0.5, spec_rad=0.9, bias=2.0)
	#res.teach(y[:-steps])
	#res.learn(washout=10)
	#output, error = res.test(y, steps)
	##### NEED TO POST PROCESS OUTPUT!!
	#print "Error: "+str(error)

	# plotting
	#f, axarr = pl.subplots(2, sharex=True)
	#axarr[0].plot(y)
	#axarr[0].plot(output)
	#axarr[1].plot(res.states)
	#pl.show()
	#pickle.dump(res, open('save.pkl', 'wb'), -1)

	#print "Saved"


if __name__ == "__main__":
	main()
