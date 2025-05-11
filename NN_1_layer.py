"""1 layer Neural network"""
import numpy as np
from scipy.stats import norm  # Import the normal distribution from scipy.stats
from matplotlib import pyplot as plt


N = 60000  # Number of datapoints
M = 55000   # Number of SGD steps
mean = 0
std_dev = 1
variance = std_dev ** 2
X = np.random.normal(loc=0, scale=np.sqrt(std_dev), size=N)  # Input data Gaussian
Y = norm.pdf(X, loc=mean, scale=std_dev)  # True values

K = 3  # Number of nodes in the hidden layer
delta_t = 0.05  # Stepsize in SGD

# Initialize the weights with small random values
Theta_1 = np.random.normal(loc=0, scale=np.sqrt(2 / K), size=K)
Theta_2 = np.random.normal(loc=0, scale=np.sqrt(2 / K), size=K)
Theta_3 = np.random.normal(loc=0, scale=np.sqrt(2 / K), size=K)

# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    ans = np.zeros_like(x)
    i = 0
    for elem in x:
        if elem > 0:
            ans[i] = 1
            i += 1
        else:
            ans[i] = 0
            i+= 1
    return ans

# Choose the activation function ('sigmoid' or 'relu')
activation_functions = {
    "sigmoid": (sigmoid, dsigmoid),
    "relu": (relu, drelu)
}

# Set the desired activation function
activation_name = "sigmoid" 
activation, dactivation = activation_functions[activation_name]

# Neural network function
def alpha(Theta_1, Theta_2, Theta_3, x):
    return np.sum(Theta_1 * activation(Theta_2 * x + Theta_3))

# Stochastic Gradient Descent (SGD)
for i in range(M):
    x = np.random.choice(X)  # Randomly select a data point
    y = norm.pdf(x, loc=mean, scale=std_dev)  # True value for the selected point

    # Compute the network output and error
    y_pred = np.sum(Theta_1 * activation(Theta_2 * x + Theta_3))
    error = y_pred - y

    # Update weights using SGD
    Theta_1 -= 2 * delta_t * error * activation(Theta_2 * x + Theta_3)
    Theta_2 -= 2 * delta_t * error * Theta_1 * dactivation(Theta_2 * x + Theta_3) * x
    Theta_3 -= 2 * delta_t * error * Theta_1 * dactivation(Theta_2 * x + Theta_3)

# Predict values using the trained network
y_predicted = []
x_storage = []
Test_error = 0
Train_error = 0
X_test = np.random.normal(loc=0, scale=np.sqrt(std_dev), size=1000)  # Input data Gaussian

for i in range(1000):
    x_test = np.random.choice(X_test)
    x_train = np.random.choice(X)
    y_train = norm.pdf(x_train, loc=mean, scale=std_dev)  # True value for the selected point
    y_pred_train = alpha(Theta_1, Theta_2, Theta_3, x_train)
    y_pred = alpha(Theta_1, Theta_2, Theta_3, x_test)
    y_true = norm.pdf(x_test, loc=mean, scale=std_dev)  # True value for the selected point
    y_predicted.append(y_pred)
    x_storage.append(x_test)
    Test_error += (y_pred - y_true) ** 2
    Train_error += (y_pred_train - y_train) ** 2

Test_error /= 1000  # Average error over the test set
Train_error /= 1000  # Average error over the training set
print("Train error:", Train_error)
print("Test error:", Test_error)

# Plot the results
plt.plot(x_storage, y_predicted, 'ro', label='Predicted')
plt.legend()
plt.show()