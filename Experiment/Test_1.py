from math import sqrt, exp
N = 2**50

def compute(N, c, r):
	t = int(sqrt(c*N)/r)
	m = float(c*N/t); p = 1.0
	for i in range(t):
		p = p*(1 - m/N)
		m = N*(1 - exp(-m/N))
	return (1 - p)*100

for log_c in range(6):
	c = 2**(log_c - 3)
	p = 1 - (2/(2.0 + c))**2
	p_array = []
	for log_r in range(6):
		p1 = compute(N, c, 2**(log_r))
		p_array.append(round(p1, 4))
	print c, p, p_array
