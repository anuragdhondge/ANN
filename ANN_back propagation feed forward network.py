#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Input and expected output
X = np.array([[1, 0]])
y = np.array([[1]])

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2)  # 2 inputs -> 2 hidden
weights_hidden_output = np.random.rand(2, 1)  # 2 hidden -> 1 output
bias_hidden = np.zeros((1, 2))
bias_output = np.zeros((1, 1))

# === Forward Propagation ===
# Input to hidden
hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
print("\n=== Forward Propagation ===\n")
print("Hidden input:\n", hidden_input, "\n")
print("Hidden output (after activation):\n", hidden_output, "\n")

# Hidden to output
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(final_input)
print("Final input:\n", final_input, "\n")
print("Predicted output:\n", predicted_output, "\n")

# === Error Calculation ===
error = y - predicted_output
print("=== Error Calculation ===\n")
print("Expected output:\n", y)
print("Predicted output:\n", predicted_output)
print("Error (expected - predicted):\n", error, "\n")

# === Backpropagation ===
print("=== Backpropagation ===\n")

# Output layer delta
d_predicted_output = error * sigmoid_derivative(predicted_output)
print("Delta at output layer:\n", d_predicted_output, "\n")

# Hidden layer delta
error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)
print("Delta at hidden layer:\n", d_hidden_layer, "\n")

# === Weight Updates ===
print("=== Weight and Bias Updates ===\n")
learning_rate = 0.1
weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
bias_output += d_predicted_output * learning_rate
bias_hidden += d_hidden_layer * learning_rate

print("Updated weights (hidden to output):\n", weights_hidden_output, "\n")
print("Updated weights (input to hidden):\n", weights_input_hidden, "\n")
print("Updated bias (output):\n", bias_output, "\n")
print("Updated bias (hidden):\n", bias_hidden, "\n")


# In[ ]:




