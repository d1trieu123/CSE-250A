

import numpy as np
import matplotlib.pyplot as plt


# Load data
train3 = np.loadtxt("HW5/train3.txt", dtype = int)
train5 = np.loadtxt("HW5/train5.txt", dtype = int)
test3 = np.loadtxt("HW5/test3.txt", dtype = int)
test5 = np.loadtxt("HW5/test5.txt", dtype = int)


training = np.append(train3, train5, axis = 0)
test = np.append(test3, test5, axis = 0)
trainer = [0] * train3.shape[0] + [1] * train5.shape[0]
tester = [0] * test3.shape[0] + [1] * test5.shape[0]

iterations = 5000

def sigmoid(weight, x):
    return 1 / (1 + np.exp(-np.dot(x, weight)))

def log_likelihood(weight, x, y):
    return np.sum(y * np.log(sigmoid(weight, x)) + (1 - y) * np.log(1 - sigmoid(weight, x)))

def gradient(weight, x, y):
    return x * (y - sigmoid(weight, x))


def learn(x, y):
    t = x.shape[0]
    learning_rate = 0.0001
    w = np.random.randint(2, size = x.shape[1])
    list_lw = []
    list_pe = []
    for i in range(iterations):
        lw = 0
        correct = 0
        sums = [0] * x.shape[1]
        for j in range(t):
            lw += log_likelihood(w, x[j], y[j])
            sums += gradient(w, x[j], y[j])
            if (sigmoid(w, x[j]) >= 0.5 and y[j] == 1) or (sigmoid(w, x[j]) < 0.5 and y[j] == 0):
                correct += 1
        list_lw.append(lw)
        w = w + learning_rate * sums
        err = (t-correct)/float(t)
        list_pe.append(err)
        if i % 100 == 0:
            print("Iteration: %d, Error: %.4f" % (i, err))
    return list_lw, w, list_pe

def predict(w, x, y):
    correct = 0
    for i in range(x.shape[0]):
        if (sigmoid(w, x[i]) >= 0.5 and y[i] == 1) or (sigmoid(w, x[i]) < 0.5 and y[i] == 0):
            correct += 1
    return (x.shape[0] - correct)/float(x.shape[0])

trainedLw, trainedW, trainedPe = learn(training, trainer)



plt.plot(trainedLw)
plt.xlabel("Iterations")
plt.ylabel("Log likelihood")
plt.title("Log likelihood vs Iterations")
plt.show()

plt.plot(trainedPe)
plt.xlabel("Iterations")
plt.ylabel("Prediction Error")
plt.title("Prediction Error vs Iterations")
plt.show()

def print_weight_matrix(trainedW):
    weight_matrix = trainedW.reshape((8, 8))
    for row in weight_matrix:
        print(' '.join(f'{val:.2f}' for val in row))

print_weight_matrix(trainedW)

