from math import sqrt, log, ceil

array = []

for c0 in range(20):
	c  = c0/10.0 + 0.1
	gc = 1 - (2/(2+c))**2
	l  = ceil(log(1 - 0.7, 1 - gc))
	array.append([round(c, 2), l, round(gc, 2), round((1 - (1 -gc)**l)*100, 2), round((l**3)*(c**2), 2)])
print array
	 
