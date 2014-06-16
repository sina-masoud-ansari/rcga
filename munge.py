import sys
import os
from datetime import datetime
from collections import OrderedDict

symbols = {}
for r,d,f in os.walk(sys.argv[1]):
    for files in f:
        if files.endswith(".csv"):
			fname = os.path.join(r,files)
			#print fname
			h = open(fname)
			nlines = 0
			for line in h:
				if nlines == 0:
					pass
				else:
					#print line.strip()
					count = 0
					sym = ""
					date = ""
					list = []
					dict = {}
					for x in line.strip().split(","):
						#print x
						if count == 0:
							sym = x
						elif count == 1:
							#date = x
							#print "'"+date+"'"
							date = datetime.strptime(x, '%d-%b-%Y')
						else:
							list.append(x)	
						count = count + 1
					dict.update({date:list})
					if sym in symbols:
						symbols[sym].update({date:list})
					else:
						symbols.update({sym:dict})
					#print dict
				nlines = nlines + 1
			h.close()

#print symbols.keys()
for k in symbols.keys():
	#print k
	f = open("ready/"+ k + ".csv", 'w')
	f.write("Date,Open,High,Low,Close,Volume\n")
	dates = symbols[k]
	dk = sorted(dates.keys())
	for d in dk:
		#print str(d.date()) + " " + str(dates[d])
		f.write(str(d.date()) + "," + str(dates[d][0]) + "," + str(dates[d][1]) + "," + str(dates[d][2]) + "," + str(dates[d][3]) + "," + str(dates[d][4])+"\n")
	f.close()

