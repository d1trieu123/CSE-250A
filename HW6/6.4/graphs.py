import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(np.cosh(x))

def Q(x, y):
    return f(y) + (np.tanh(y)*(x-y)) + (0.5*(x-y)**2)

x = np.linspace(-10, 10, 100)
y1 = f(x)
y2 = Q(x, -2)
y3 = Q(x, 3)

plt.plot(x, y1, label='f(x)')
plt.plot(x, y2, label='Q(x,-2)')
plt.plot(x, y3, label='Q(x,3)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

