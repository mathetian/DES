import matplotlib.pyplot as plt

x = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
y = [44, 21, 13, 10, 7, 6, 4, 3, 3]

plt.plot(x, y)
plt.axis([0, 100, 0, 50])
plt.ylabel('number of tables')
plt.xlabel('success rate for single table(%)')
plt.show()

