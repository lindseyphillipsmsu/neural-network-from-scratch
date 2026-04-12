# layers.py
# A single dense (fully connected) layer.
# Each layer stores its own weights and biases,
# knows how to do a forward pass,
# and returns gradients during backpropagation.

import numpy as np
from activations import get_activation


class DenseLayer:
    """
    A fully connected neural network layer.
    
    During forward pass: computes Z = W·X + b, then A = activation(Z)
    During backward pass: computes gradients and returns them upstream
    """

    def __init__(self, input_size, output_size, activation="relu"):
        """
        Initializes weights using He initialization.
        
        Why He initialization?
        Random weights prevent symmetry breaking failure —
        if all weights start at 0, every neuron learns
        the exact same thing and the network never improves.
        He scaling (sqrt(2/n)) keeps gradients stable with ReLU.
        """
        self.activation_name = activation
        self.activation_fn, self.activation_derivative = get_activation(activation)

        # He initialization — scales by sqrt(2 / input_size)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros((1, output_size))

        # Cache for backpropagation
        self.X = None
        self.Z = None
        self.A = None

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Forward pass through this layer.
        
        X → weighted sum (Z) → activation (A)
        Caches X and Z for use during backpropagation.
        """
        self.X = X
        self.Z = X @ self.W + self.b  # Matrix multiplication
        self.A = self.activation_fn(self.Z)
        return self.A

    def backward(self, dA, learning_rate):
        """
        Backward pass — computes gradients and updates weights.
        
        Chain rule applied:
            dZ = dA * activation'(Z)
            dW = X.T · dZ / m
            db = mean(dZ)
            dX = dZ · W.T  (passed back to previous layer)
        """
        m = self.X.shape[0]  # batch size

        dZ = dA * self.activation_derivative(self.Z)
        self.dW = (self.X.T @ dZ) / m
        self.db = np.mean(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T

        # Gradient descent weight update
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return dX  # Pass gradient back to previous layer

    def __repr__(self):
        return (f"DenseLayer(in={self.W.shape[0]}, "
                f"out={self.W.shape[1]}, "
                f"activation={self.activation_name})")
