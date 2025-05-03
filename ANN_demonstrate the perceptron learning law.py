#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Set the visual theme for seaborn plots
sns.set_theme()

# Load the Iris dataset from a local file
df = pd.read_csv(r"C:\Users\91989\Downloads\iris.csv")  

# For binary classification: select only first 100 samples (Setosa & Versicolor)
y = df.iloc[:100].species.values  # Get the species column
y = np.where(y == 'setosa', -1, 1)  # Encode labels: 'setosa' as -1, 'versicolor' as 1
X = df[["sepal_length", "sepal_width"]].iloc[:100].values  # Use only two features for 2D plotting

# Define the Perceptron class
class Perceptron(object):
    def __init__(self, eta=0.5, epochs=50):
        self.eta = eta              # Learning rate
        self.epochs = epochs        # Number of passes over the training dataset

    def train(self, X, y):
        self.w_ = np.random.rand(1 + X.shape[1])  # Initialize weights (including bias)
        self.errors_ = []                         # To store number of misclassifications per epoch

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):  # Loop through each training sample
                update = self.eta * (target - self.predict(xi))  # Update rule
                self.w_[:-1] += update * xi      # Update weights
                self.w_[-1] += update            # Update bias
                errors += int(update != 0)       # Count if there was an update (i.e., misclassified)

            if errors == 0:
                return self                      # Stop early if perfectly classified
            else:
                self.errors_.append(errors)      # Log errors for plotting

        return self

    def net_input(self, X):
        # Calculate the linear combination of inputs and weights
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        # Return class label based on sign of net input
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Create and train the Perceptron classifier
clf = Perceptron(epochs=1000)
clf.train(X, y)

# Predict class labels for training set
y_hat = clf.predict(X)

# Print accuracy of the model
print("Accuracy:", np.mean(y == y_hat))

# Plot decision boundary
plt.figure(figsize=(10, 8))
plot_decision_regions(X, y, clf=clf)
plt.title("Perceptron Decision Region")
plt.xlabel("sepal length [cm]", fontsize=15)
plt.ylabel("sepal width [cm]", fontsize=15)
plt.show()

# Plot number of misclassifications for each epoch (learning curve)
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(clf.errors_) + 1), clf.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Misclassifications")
plt.title("Training Error per Epoch")
plt.show()


# In[ ]:




