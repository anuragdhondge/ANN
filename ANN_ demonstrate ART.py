#!/usr/bin/env python
# coding: utf-8

import numpy as np

class FuzzyART:
    def __init__(self, rho=0.75, alpha=0.01, beta=1.0):
        # Initialize parameters
        self.rho = rho      # Vigilance parameter: controls similarity threshold for learning
        self.alpha = alpha  # Choice parameter: prevents division by zero
        self.beta = beta    # Learning rate: determines how fast weights are updated
        self.weights = []   # Stores prototype vectors (categories)

    def _complement_coding(self, input_vector):
        # Complement coding: appends (1 - input) to input to handle both presence and absence of features
        cc_vector = np.concatenate([input_vector, 1 - input_vector])
        print(f"Complement Coded Input: {cc_vector}")
        return cc_vector

    def _choice_function(self, input_vector, weights):
        # Choice function Tj: evaluates similarity with category weights
        return np.sum(np.minimum(input_vector, weights)) / (self.alpha + np.sum(weights))

    def _match_function(self, input_vector, weights):
        # Match function: evaluates how much the input matches the category template
        return np.sum(np.minimum(input_vector, weights)) / np.sum(input_vector)

    def train(self, input_vectors):
        print("=== Training Start ===")
        for i, input_vector in enumerate(input_vectors):
            print(f"\nTraining on Input {i+1}: {input_vector}")
            input_cc = self._complement_coding(input_vector)

            # If no weights exist, initialize first category
            if len(self.weights) == 0:
                print("No existing weights, creating first category.")
                self.weights.append(input_cc.copy())
                continue

            # Calculate choice function for all existing categories
            choice_values = []
            for j, w in enumerate(self.weights):
                Tj = self._choice_function(input_cc, w)
                choice_values.append((j, Tj))
                print(f"Choice Function T[{j}] = {Tj:.4f}")

            # Sort by choice value in descending order
            sorted_choices = sorted(choice_values, key=lambda x: -x[1])
            matched = False

            # Try to find a matching category that meets the vigilance criterion
            for j, _ in sorted_choices:
                match = self._match_function(input_cc, self.weights[j])
                print(f"Match Function M[{j}] = {match:.4f}")
                if match >= self.rho:
                    print(f"Match found with Class {j}, updating weights.")
                    # Update weights using learning rule
                    self.weights[j] = self.beta * np.minimum(input_cc, self.weights[j]) + (1 - self.beta) * self.weights[j]
                    print(f"Updated Weights for Class {j}: {self.weights[j]}")
                    matched = True
                    break
                else:
                    print(f"Class {j} rejected due to low match.")

            # If no category matches, create a new one
            if not matched:
                print("No match met vigilance. Creating new class.")
                self.weights.append(input_cc.copy())
        print("=== Training Complete ===")

    def predict(self, input_vector):
        print(f"\nPredicting for Input: {input_vector}")
        input_cc = self._complement_coding(input_vector)
        best_match = -1
        best_choice = -np.inf

        # Check each category for the best match
        for j, w in enumerate(self.weights):
            Tj = self._choice_function(input_cc, w)
            M = self._match_function(input_cc, w)
            print(f"Class {j} => Choice: {Tj:.4f}, Match: {M:.4f}")
            if M >= self.rho and Tj > best_choice:
                best_choice = Tj
                best_match = j

        if best_match == -1:
            print("No class match found.")
        else:
            print(f"Predicted Class: {best_match}")
        return best_match


# === Example Usage ===
training_data = np.array([
    [1, 0, 0, 1],    # Cluster A
    [1, 1, 0, 1],    # Cluster A
    [0, 0, 1, 0],    # Cluster B
    [0, 1, 1, 0],    # Cluster B
])

test_data = np.array([
    [1, 0, 0, 1],    # Should match A
    [0, 1, 1, 0],    # Should match B
    [1, 1, 1, 1],    # May not match any
])

# Create and train the Fuzzy ART model
art = FuzzyART(rho=0.75, alpha=0.01, beta=1.0)
art.train(training_data)

# Test the model with new data
print("\n=== Testing ===")
for test_vec in test_data:
    result = art.predict(test_vec)
    print(f"Input: {test_vec} => Predicted Class: {result}")
