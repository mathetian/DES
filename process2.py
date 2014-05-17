from math import exp

# N = 9474296896.0

# m=7680000.0
# t=4000
# l=1

#print m*t-N

N = 1000000.0
t = 10000
m = 10000.0
l = 1

def cal(N,m,t):
	rs = 1
	mt=m
	for i in range(t):
		rs = rs*(1-mt/N)
		mt=N*(1.0-exp(-mt/N))

	return (1.0-rs)


print (1-(1-cal(N,m,t))**l)*100

