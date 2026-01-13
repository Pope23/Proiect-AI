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
    
    class EvolutionaryAlgorithm:
        def __init__(self, network_structure, pop_size=30, generations=50):
            self.network_structure = network_structure
            self.pop_size = pop_size
            self.generations = generations
            self.population = []
            self.best_accuracy = 0
            self.best_network = None

    def create_individual(self):
        return NeuralNetwork(self.network_structure)

    def initialize_population(self):
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def fitness(self, network, X_train, y_train):
        return network.accuracy(X_train, y_train)

    def evolve(self, X_train, y_train):
        self.initialize_population()
        history = []

        print("\n3. Algoritm Evolutiv în execuție...")

        for gen in range(self.generations):
            fitness_scores = [self.fitness(net, X_train, y_train) for net in self.population]
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]

            if best_fitness > self.best_accuracy:
                self.best_accuracy = best_fitness
                self.best_network = self.population[best_idx]

            history.append(best_fitness)

            new_population = []
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population.append(self.population[sorted_indices[0]])
            new_population.append(self.population[sorted_indices[1]])

            while len(new_population) < self.pop_size:
                fitness_array = np.array(fitness_scores) + 1e-10
                probs = fitness_array / fitness_array.sum()
                p1 = self.population[np.random.choice(len(self.population), p=probs)]
                p2 = self.population[np.random.choice(len(self.population), p=probs)]

                child = NeuralNetwork(self.network_structure)

                for i in range(len(child.weights)):
                    mask = np.random.rand(*child.weights[i].shape) > 0.5
                    child.weights[i] = np.where(mask, p1.weights[i], p2.weights[i])
                    bias_mask = np.random.rand(child.biases[i].shape[0]) > 0.5
                    child.biases[i] = np.where(bias_mask, p1.biases[i], p2.biases[i])

                if np.random.rand() < 0.2:
                    idx = np.random.randint(len(child.weights))
                    child.weights[idx] += np.random.randn(*child.weights[idx].shape) * 0.1
                    child.biases[idx] += np.random.randn(*child.biases[idx].shape) * 0.1

                new_population.append(child)

            self.population = new_population

            if gen % 10 == 0:
                print(f"  Generația {gen:3d}: Acuratețea maximă = {best_fitness:.4f}")

        return history

print("\n" + "="*60)
print("ANTRENARE REȚEA NEURONALĂ CU ALGORITM EVOLUTIV")
print("="*60)

network_structure = [4, 5, 3]

ea = EvolutionaryAlgorithm(network_structure, pop_size=30, generations=50)
history = ea.evolve(X_train, y_train)

test_accuracy = ea.best_network.accuracy(X_test, y_test)

print("\n" + "="*60)
print("4.REZULTATE FINALE:")
print("="*60)
print(f"Acuratețea pe antrenare: {ea.best_accuracy:.2%}")
print(f"Acuratețea pe testare:   {test_accuracy:.2%}")
print(f"Număr total de ponderi:  {sum(w.size for w in ea.best_network.weights)}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history, linewidth=2)
plt.xlabel('Generație')
plt.ylabel('Acuratețea maximă')
plt.title('Evoluția algoritmului genetic')
plt.grid(True)
plt.ylim([0, 1.1])

plt.subplot(1, 3, 2)
from sklearn.metrics import confusion_matrix
predictions = ea.best_network.predict(X_test)
true_labels = np.argmax(y_test, axis=1)
cm = confusion_matrix(true_labels, predictions)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cbar=False)
plt.title('Matrice de confuzie')
plt.xlabel('Prezis')
plt.ylabel('Adevărat')

plt.subplot(1, 3, 3)
correct = predictions == true_labels
plt.bar(['Corecte', 'Greșite'], [np.sum(correct), len(correct)-np.sum(correct)])
plt.title(f'Predicții pe setul de test ({test_accuracy:.1%})')

plt.tight_layout()
plt.show()

print("\nExemple de predicții:")
for i in range(min(10, len(X_test))):
    pred = ea.best_network.predict(X_test[i:i+1])[0]
    true = np.argmax(y_test[i])
    status = "CORECT" if pred == true else "GREȘIT"
    print(i+1, iris.target_names[pred], iris.target_names[true], status)
