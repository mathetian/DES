from math import sqrt, log, ceil
import matplotlib.pyplot as plt

for p0 in range(5):
	p     = 0.5 + p0/10.0
	array = []	
	for c0 in range(20):
		c  = c0/10.0 + 0.1
		gc = 1 - (2/(2+c))**2
		l  = ceil(log(1 - p, 1 - gc))
		array.append(round((l**3)*(c**2), 2))
	print array
	plt.plot([c0/10.0 + 0.1 for c0 in range(20)], array)
	 
