from math import sqrt, exp
N = 2**30

def compute(N, c, r):
	m = ((N*c)**(2/3.0))*(r**(1/3.0))
	t = int((c*N)/m);p = 1.0
	for i in range(t):
		p = p*(1 - m/N)
		m = N*(1 - exp(-m/N))
	return (1 - p)*100

ends = []
for log_c in range(7):
	c  = 1
	if log_c > 3:
		c = 2**(2**(log_c - 3))
	elif log_c < 3:
		c = 2**(-2**(3 - log_c))
	p = 1 - (2/(2.0 + c))**2
	p_array = []
	for log_r in range(7):
		r  = 1
        	if log_r > 3:
                	r = 2**(2**(log_r - 3))
        	elif log_r < 3:
                	r = 2**(-(2**(3 - log_r)))
		p1 = compute(N, c, r)
		p_array.append(round(p1, 3))
	# print c, p, p_array
	ends.append(p_array)

result = [[i for i in range(7)] for j in range(7)]
for i in range(7):
	for j in range(7):
		result[i][j] = ends[j][i]
	print result[i]	
