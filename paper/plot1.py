import matplotlib.pyplot as plt

x  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y0 = [0.9999, 0.9997, 0.9995, 0.999, 0.998, 0.9959, 0.9919, 0.9839, 0.9683, 0.9386, 0.8843, 0.7926, 0.6565, 0.4887, 0.3233, 0.1929, 0.1067, 0.0564, 0.029, 0.0147]
y1 = [0.9998, 0.9995, 0.999, 0.9981, 0.9961, 0.9922, 0.9846, 0.9697, 0.9412, 0.8889, 0.8, 0.6667, 0.5, 0.3333, 0.2, 0.1111, 0.0588, 0.0303, 0.0154, 0.0078]
y2 = [0.9995, 0.999, 0.9981, 0.9962, 0.9924, 0.985, 0.9704, 0.9425, 0.8912, 0.8037, 0.6718, 0.5058, 0.3385, 0.2038, 0.1134, 0.0601, 0.031, 0.0157, 0.0079, 0.004]
plt.plot(x, y0, 'r--',x, y1, 'g--', x, y2, 'b--', )
plt.axis([0, 20, 0, 1.1])
plt.ylabel('ratio')
plt.xlabel('log_2(t)')
plt.show()
