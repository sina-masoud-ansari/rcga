import numpy as np

x = np.random.rand(4,4)
todel = []
for i in range(0, 4):
	col = x[:,i]
	mask = np.where( col > 0.9)
	for m in mask:
		todel += m.tolist()
		#for j in m:
			#print j,
			#todel.append(j)
todel = set(todel)
print todel
tosave = set(range(0, len(x))) - todel
print tosave
print x
xn = []
for i in tosave:
	xn.append(x[i])
xn = np.array(xn)
print xn

