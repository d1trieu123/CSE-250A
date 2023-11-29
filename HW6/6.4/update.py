import numpy as np
import matplotlib.pyplot as plt

def f_prime(x):
    return np.tanh(x)

def update_rule(x_n):
    return x_n - f_prime(x_n)


x0_1 = -2
x0_2 = 3


num_iterations = 100


x_values_1 = [x0_1]
x_values_2 = [x0_2]

for _ in range(num_iterations):
    x0_1 = update_rule(x0_1)
    x0_2 = update_rule(x0_2)
    x_values_1.append(x0_1)
    x_values_2.append(x0_2)

# Plotting
plt.plot(range(num_iterations+1), x_values_1, label='x0 = -2')
plt.plot(range(num_iterations+1), x_values_2, label='x0 = 3')

plt.legend()
plt.show()
