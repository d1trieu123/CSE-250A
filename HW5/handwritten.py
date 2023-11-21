import numpy as np
import matplotlib.pyplot as plt

# Load data
test3_data = np.loadtxt('HW5/test3.txt')
test5_data = np.loadtxt('HW5/test5.txt')
train3_data = np.loadtxt('HW5/train3.txt')
train5_data = np.loadtxt('HW5/train5.txt')

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic Regression fit function using gradient ascent
def fit_logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    log_likelihood_vals = []

    for _ in range(iterations):
        linear_pred = np.dot(X, weights)
        predictions = sigmoid(linear_pred)
        log_likelihood = np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        log_likelihood_vals.append(log_likelihood)

        dw = (1 / n_samples) * np.dot(X.T, (y - predictions))  
        weights += learning_rate * dw  

    return weights, log_likelihood_vals

# Logistic Regression predict function
def predict_logistic_regression(X, weights):
    linear_pred = np.dot(X, weights)
    y_pred = sigmoid(linear_pred)
    class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
    return class_pred

def calculate_error_rate(predictions, true_labels):
    incorrect_predictions = np.sum(predictions != true_labels)
    error_rate = incorrect_predictions / len(true_labels)
    return error_rate

# Concatenate data for training and testing
X_test = np.concatenate((test3_data, test5_data))
y_test = np.concatenate((np.zeros(len(test3_data)), np.ones(len(test5_data))))

X_train = np.concatenate((train3_data, train5_data))
y_train = np.concatenate((np.zeros(len(train3_data)), np.ones(len(train5_data))))

# Train the model using gradient ascent
weights, log_likelihood_vals = fit_logistic_regression(X_train, y_train, learning_rate=0.01, iterations=1000)

# Make predictions
test_predictions = predict_logistic_regression(X_test, weights)
train_predictions = predict_logistic_regression(X_train, weights)

# Accuracy calculation
def accuracy(predictions, true_labels):
    return np.sum(predictions == true_labels) / len(true_labels)

test_accuracy = accuracy(test_predictions, y_test)
train_accuracy = accuracy(train_predictions, y_train)

# Print accuracies
print("Test accuracy: ", test_accuracy)
print("Train accuracy: ", train_accuracy)

# Plot log likelihood
plt.plot(range(1000), log_likelihood_vals)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.show()

# print weights matrix
def print_weights_matrix(weights):
    weights_matrix = np.array(weights).reshape(8, 8)
    print("Weights Matrix:")
    for row in weights_matrix:
        print(row)


print_weights_matrix(weights)


test_error_rate = calculate_error_rate(test_predictions, y_test)

# Print the error rate
print("Test error rate: {:.2%}".format(test_error_rate))