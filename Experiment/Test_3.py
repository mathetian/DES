from math import exp

def getP(N, m, i, p):
	return (1 - 1.0/(N/m + (i + 1)/2))*(1 - exp(-p))
	#return 1 - exp(-p)

def compute(N, m, c):
	t  = int(c*N/m); t0 = t/2
	p  = 1.0/(N/m + (t0-1)/2.0)
	r  = (4.0/(4 + c))**2
	for i in range(t0):
		r = r*(1 - p)
		p = getP(N, m, i + 1, p)
	return 1 - r

N = 2**51; m = 2**34; c = 1
print compute(N, m, c)
