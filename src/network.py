# network.py
# Stacks DenseLayers into a complete neural network.
# Handles the full training loop:
#   forward pass → loss → backward pass → weight updates

import numpy as np
from layers import DenseLayer


class NeuralNetwork:
    """
    A fully connected feedforward neural network built from scratch.
    No TensorFlow. No PyTorch. Just NumPy and math.
    
    Architecture is defined by a list of layer sizes.
    Example: [784, 128, 64, 10] = input → 2 hidden → output
    """

    def __init__(self, layer_sizes, activations=None, learning_rate=0.01):
        """
        Builds the network layer by layer.
        
        Args:
            layer_sizes: list of ints — neuron count per layer
            activations: list of activation names per hidden layer
            learning_rate: step size for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_history = []
        self.accuracy_history = []

        # Default all hidden layers to ReLU
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["sigmoid"]

        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)

    def forward(self, X):
        """
        Passes input through every layer in sequence.
        Output of each layer becomes input of the next.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_pred, y_true):
        """
        Backpropagates loss gradient through all layers in reverse.
        
        Starting gradient: derivative of binary cross-entropy loss
        dL/dy_pred = -(y/y_pred) + (1-y)/(1-y_pred)
        Then simplified: (y_pred - y_true) / m
        """
        m = y_true.shape[0]
        # Gradient of binary cross-entropy w.r.t output
        dA = (y_pred - y_true.reshape(-1, 1)) / m

        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.learning_rate)

    def compute_loss(self, y_pred, y_true):
        """
        Binary cross-entropy loss.
        
        Measures how far predictions are from truth.
        Perfect predictions = 0. Random guessing ≈ 0.693.
        Clipping prevents log(0) which would be undefined.
        """
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def compute_accuracy(self, y_pred, y_true):
        """Converts probabilities to class labels and measures accuracy."""
        predictions = (y_pred > 0.5).astype(int).flatten()
        return np.mean(predictions == y_true)

    def train(self, X, y, epochs=1000, print_every=100):
        """
        Full training loop.
        
        Each epoch:
            1. Forward pass — get predictions
            2. Compute loss
            3. Backward pass — compute gradients
            4. Weights update automatically inside each layer
        """
        print("Training started...\n")

        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)

            # Loss + accuracy
            loss = self.compute_loss(y_pred, y)
            accuracy = self.compute_accuracy(y_pred, y)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            # Backward
            self.backward(y_pred, y)

            if epoch % print_every == 0:
                print(f"Epoch {epoch:>5} | "
                      f"Loss: {loss:.4f} | "
                      f"Accuracy: {accuracy:.4f}")

        print(f"\nTraining complete.")
        print(f"Final Loss: {self.loss_history[-1]:.4f}")
        print(f"Final Accuracy: {self.accuracy_history[-1]:.4f}")

    def predict(self, X):
        """Returns class predictions (0 or 1) for new inputs."""
        return (self.forward(X) > 0.5).astype(int).flatten()

    def __repr__(self):
        arch = " → ".join(str(l) for l in self.layers)
        return f"NeuralNetwork({arch})"


if __name__ == "__main__":
    # Quick sanity check — XOR problem
    # A network that can't solve XOR is broken
    print("Running XOR sanity check...\n")

    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    nn = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        activations=["relu", "sigmoid"],
        learning_rate=0.1
    )

    nn.train(X, y, epochs=5000, print_every=1000)

    print("\nPredictions:")
    for xi, yi in zip(X, y):
        pred = nn.predict(xi.reshape(1, -1))[0]
        print(f"  Input: {xi} | Expected: {int(yi)} | Got: {pred}")
