import numpy as np


class DataReader:

	def __init__(self, input):
		self.symbols = []
		if isinstance(input, list):
			print "Input is list"
			self.load_from_list(input)
		elif isinstance(input, str):
			print "Input is string, assuming filename"

	def load_from_list(self, input):
		for s in input:
			self.symbols.append(Symbol(s))

	def load_from_file(self, fname):
		pass

# Read in a csv
class Symbol:
	
	def __init__(self, filename):
		self.dates, self.ohlc = self.read_data(filename)
	
	def read_data(self, filename):
		arr = []
		dates = []
		with open(filename) as f:
			lines = f.readlines()
			for i in range(1, len(lines)):
				line = lines[i].strip()
				d,o,h,l,c,v = line.strip().split(",")
				tmp = [float(o), float(h), float(l), float(c), float(v)]
				arr.append(tmp)
				dates.append(d)
		return np.array(dates), np.array(arr)

	def retrieve(self, type):
		t = type.lower()

		if t == "open":
			return self.ohlc[:, 0]
		elif t == "high":
			return self.ohlc[:, 1]
		elif t == "low":
			return self.ohlc[:, 2]
		elif t == "close":
			return self.ohlc[:, 3]
		else:
			print "Unsupported Symbol retrieval type: "+type
