import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Generate a range of values
x = np.linspace(-10, 10, 1000)

# Plot each activation function
plt.figure(figsize=(10, 8))

plt.subplot(321)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(322)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.subplot(323)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(324)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(325)
plt.plot(x, softmax(x))
plt.title('Softmax')

plt.tight_layout()
plt.show()