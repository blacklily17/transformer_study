import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

inputs = np.array([0.2,0.5,0.7])
weights = np.array([0.3,0.8,0.2])
bias = 0.5

z = np.dot(inputs, weights) + bias
output = sigmoid(z)

print(f"输出值为: {output}")









