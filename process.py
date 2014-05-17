N = 1000000.0
m = 10000
t = 1000

rs = 0.0
for i in range(m):
	for j in range(t):
		rs += (((N-(i+1)*t)/N)**(j+1))

rs = rs/N*100

print rs