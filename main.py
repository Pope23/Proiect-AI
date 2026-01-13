import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("1.Încărcare date Iris Dataset...")
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Date antrenare: {X_train.shape}, Date test: {X_test.shape}")
print(f"Număr clase: {y_encoded.shape[1]}")

class NeuralNetwork:
    def __init__(self, layers=[4, 5, 3]):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
            self.biases.append(np.random.randn(layers[i+1]) * 0.1)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)-1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            activations.append(a)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        output = self.softmax(z_output)
        activations.append(output)
        return activations

    def predict(self, X):
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)
