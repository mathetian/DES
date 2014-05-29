from math import log
from math import sqrt

# experiment 1, decrease trend

# fixed m 2**22, t change from 2**0 -> 2**20

def Test1_Inner(N, m , t0):
	rs = 1
	mt = m
	t = 2**t0
	for i in range(t):
		rs = rs*(1-mt/N)
		mt = N*(1.0-(1.0-1/N)**mt)

	#print '%2.0f'%t0, '%.4f'%(mt/m)

	return [t0, '%.4f'%(mt/m)]

def DoTest1():
	N = 8589934592.0
	m = 2194304.0 #2**22
	rs = []
	for i in range(20):
		rs.append(Test1_Inner(N, m, i))
	print rs
	x = []
	y = []
	for i in rs:
		x.append(i[0])
		y.append(float(i[1]))
	print x,y

# DoTest1()

#  m = 2**23
#  0 0.9995
#  1 0.9990
#  2 0.9981
#  3 0.9962
#  4 0.9924
#  5 0.9850
#  6 0.9704
#  7 0.9425
#  8 0.8912
#  9 0.8037
# 10 0.6718
# 11 0.5058
# 12 0.3385
# 13 0.2038
# 14 0.1134
# 15 0.0601
# 16 0.0310
# 17 0.0157
# 18 0.0079
# 19 0.0040

#  m = 2**22
#  0 0.9998
#  1 0.9995
#  2 0.9990
#  3 0.9981
#  4 0.9961
#  5 0.9922
#  6 0.9846
#  7 0.9697
#  8 0.9412
#  9 0.8889
# 10 0.8000
# 11 0.6667
# 12 0.5000
# 13 0.3333
# 14 0.2000
# 15 0.1111
# 16 0.0588
# 17 0.0303
# 18 0.0154
# 19 0.0078

#  m = 2**21
#  0 0.9999
#  1 0.9997
#  2 0.9995
#  3 0.9990
#  4 0.9980
#  5 0.9959
#  6 0.9919
#  7 0.9839
#  8 0.9683
#  9 0.9386
# 10 0.8843
# 11 0.7926
# 12 0.6565
# 13 0.4887
# 14 0.3233
# 15 0.1929
# 16 0.1067
# 17 0.0564
# 18 0.0290
# 19 0.0147

# experiment 2, multiple table

def Test2_Inner(p, l):
	p1 = 1-(1-p)**l
	if p1 > 0.99:
		print '%.1f'%p, l, '%.4f'%p1
		return [1, p, l]

	return [0, 0, 0, 0]

def DoTest2():
	p0 = 1
	flag = 0
	rs = []
	for i in range(9):
		for j in range(50):
			ret = Test2_Inner((i+1)/10.0, j)
			if ret[0] == 1:
				rs.append(ret[1:])
				break
	rs0 = []
	rs1 = []

	for i in rs:
		rs0.append(i[0])
		rs1.append(i[1])
	print rs0, rs1

# DoTest2()

# 0.1 22 0.9015
# 0.2 11 0.9141
# 0.3 7 0.9176
# 0.4 5 0.9222
# 0.5 4 0.9375
# 0.6 3 0.9360
# 0.7 2 0.9100
# 0.8 2 0.9600
# 0.9 2 0.9900

# experiment 3, fixed m*t

def Test3_Inner(N, m, t):
	rs = 1
	mt = m
	for i in range(t):
		rs = rs*(1-mt/N)
		mt = N*(1.0-(1.0-1/N)**mt)
	ps = 1 - rs
	print m ,',', t, ',' , '%.4f'%ps

def DoTest3():
	N = 8589934592.0
	m = 4194304.0 #2**22
	t = 2048
	for i in range(10):
		Test3_Inner(N, m/(2**i), t*(2**i))

# Test3_Inner(8589934592.0, 5284820.0, int(2581))
# DoTest3()

# 4194304.0 2048 0.5557
# 2097152.0 4096 0.5556
# 1048576.0 8192 0.5556
# 524288.0 16384 0.5556
# 262144.0 32768 0.5556
# 131072.0 65536 0.5556
# 65536.0 131072 0.5556
# 32768.0 262144 0.5556
# 16384.0 524288 0.5556
# 8192.0 1048576 0.5556

# experiment 4, Final Test

def Test4_Inner(N, m, t, k, rss):
	k0 = sqrt(k)

	m=m*k0
	t=(int)(t*k0)

	rs = 1
	mt = m
	for i in range(t):
		rs = rs*(1-mt/N)
		mt = N*(1.0-(1.0-1/N)**mt)

	flag = 1
	for l in range(20):
		prob = ((1-(rs**(l+1)))*100)
		if prob >= 99.0 and flag == 1:
			if prob >= 99.1 : break
 			print '%.2f'%k, '%.2f'%(k*(l+1)), l+1, '%.4f'%(((l+1)**3)*(k**2)), '%.3f'%(prob)
 			return ['%.2f'%k, '%.2f'%(k*(l+1)), l+1, '%.2f'%(((l+1)**3)*(k**2)), '%.2f'%(prob)]
			flag = 0
		rss.append(float('%.3f'%prob))
	
	# print rss
	# rss1 = []
	# rss1.append({'k^2':'%.2f'%k**2})
	# rss1.append({'p':'%.4f'%((1.0-rs)*100)})
	# rss1.append({'l':arr})
	# rss.append(rss1)

def DoTest4():
	# N = 8589934592.0 #2**33

	# m = 4194304.0 #2**22
	# t = 2048 #2**11

	N = 137438953472.0
	m = 16777216.0
	t = 8192

	orig = 0.1

	x = []
	y1 = []
	y2 = []
	y3 = [] 
	rss2 = []
	for i in range(200):
		rss = []
		rs2 = Test4_Inner(N, m, t, orig+i*0.01, rss)
		if rs2 != None:
			rss2.append(rs2)
		# rss2.append(rss)
	#print rss2
	x = [[] for i in range(len(rss2))]

	for i in range(5):
		for j in range(len(rss2)):
			x[j].append(float(rss2[j][i]))

	for i in x:
		print i
	# for i in rss2:
	# 	x.append(float(i[0]))
	# 	y1.append(float(i[3]))
	#print x
	#print y1
	# for i in range(20):
	# 	y1 = []
	# 	for j in range(20):
	# 		y1.append(rss2[j][i])
	# 	# print "y%d ="%i,y1

	# rss = []
	# Test4_Inner(N, m, t, sqrt(0.00001), rss)
	# print rss

DoTest4()

# k=1, 5 + 45% = {(l**3)(k**2) : , l*k : 5.7}
# k=1.56, 4  = { , l*k : 6.24}

def Test5_Inner(N, m, t, k0):
	k = sqrt(k0)
	m=m*k
	t=(int)(t*k)

	rs = 1
	mt = m
	for i in range(t):
		rs = rs*(1-mt/N)
		mt = N*(1.0-(1.0-1/N)**mt)

	flag = 1
	prob = 1-rs
	return ['%.2f'%(k0), '%.4f'%prob]

def DoTest5():
	N = 137438953472.0 #2**33

	m = 16777216.0 #2**22
	t = 8192 #2**11

	orig = 0.1

	rs = []
	for i in range(20):
		rs.append(Test5_Inner(N, m, t, orig+i*0.1))
	x = [] 
	y = []
	for i in range(20):
		x.append(float(rs[i][0]))
		y.append(float(rs[i][1]))

	print x
	print y

# DoTest5()

def Test6_Inner(N, m, t, k):
	m=m*k
	t=(int)(t/k)
	#print m, t, m*t, N
	rs = 1
	mt = m
	for i in range(t):
		rs = rs*(1-mt/N)
		mt = N*(1.0-(1.0-1/N)**mt)

	return "%.1f"%((1 - rs)*100)

def DoTest6():
	N = 137438953472.0 #2**33

	rs = []
	for i in range(10):
		N1 = N * (2**(i-5))
		t = 2**((int)(10+i/2))
		m = float(N1/t)
		rss = []
		for j in range(10):
			rss.append(Test6_Inner(N, m, t, 2**j))
		rs.append(rss)

	for i in range(10):
		rss = []
		for j in range(10):
			rss.append(float(rs[j][i]))
		print rss
# DoTest6()