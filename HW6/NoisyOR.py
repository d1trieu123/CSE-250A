import numpy as np
import math

def load_data():
    spectX = np.loadtxt('HW6\X.txt', dtype=int)
    spectY = np.loadtxt('HW6\Y.txt', dtype=int)
    return spectX, spectY

def initial_parameters():
    return np.ones((23, 1)) * 0.05

def compute_loss(p, X, Y):
    temp = X.dot(np.log(1 - p))
    return -1 / X.shape[0] * np.sum(Y[:, np.newaxis] * np.log(1 - np.exp(temp)) + (1 - Y[:, np.newaxis]) * temp)

def update_parameters(p, X, Y):
    temp0 = X.dot(np.log(1 - p))
    temp1 = (X.T * p).T * Y[:, np.newaxis] / (1 - np.exp(temp0))
    p_new = (np.sum(temp1, axis=0) / np.sum(X, axis=0))[:, np.newaxis]
    return p_new

def predict(p, X, Y):
    temp = X.dot(np.log(1 - p))
    prob = 1 - np.exp(temp)
    pred = np.where(prob >= 0.5, 1, 0)
    return np.sum(np.abs(pred - Y[:, np.newaxis]))

def train_model(iterations=256):
    spectX, spectY = load_data()
    p = initial_parameters()

    initial_loss = compute_loss(p, spectX, spectY)
    initial_error = predict(p, spectX, spectY)
    list_loss= [initial_loss]
    list_error= [initial_error]

    print("iteration 0\tnumber of mistakes %d\tL %.5f" % (initial_error, -initial_loss))

    for i in range(1, iterations + 1):
        p = update_parameters(p, spectX, spectY)
        error = predict(p, spectX, spectY)
        loss = compute_loss(p, spectX, spectY)
        list_error.append(error)
        list_loss.append(loss)

        if math.log(i, 2).is_integer():
            print("iteration %d\tnumber of mistakes %d\tL %.5f" % (i, error, -loss))

    return list_loss, list_error


train_model()
