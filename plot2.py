import matplotlib.pyplot as plt

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
y = [44, 21, 13, 10, 7, 6, 4, 3, 3]

plt.plot(x, y)
plt.axis([0, 1, 0, 50])
plt.ylabel('number of tables')
plt.xlabel('success rate for single table')
plt.show()

