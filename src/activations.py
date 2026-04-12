# activations.py
# Activation functions and their derivatives.
# These are the "decision gates" of a neural network —
# they decide whether a neuron fires and how strongly.
# Derivatives are required for backpropagation.

import numpy as np


def sigmoid(z):
    """
    Squishes any value into a range between 0 and 1.
    Used in output layer for binary classification.
    
    Formula: 1 / (1 + e^-z)
    
    Problem: vanishing gradients on deep networks —
    gradients shrink toward zero, killing learning.
    This is why ReLU replaced it in hidden layers.
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid — needed for backpropagation.
    Formula: sigmoid(z) * (1 - sigmoid(z))
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    Rectified Linear Unit — the most widely used activation.
    Returns z if positive, 0 if negative.
    
    Formula: max(0, z)
    
    Why it works: computationally cheap, no vanishing gradient
    problem in positive range, creates sparse activations.
    
    Problem: dying ReLU — neurons stuck at 0 never recover.
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU.
    1 where z > 0, 0 where z <= 0.
    This is what causes the dying ReLU problem —
    dead neurons have zero gradient and stop learning.
    """
    return (z > 0).astype(float)


def softmax(z):
    """
    Converts raw scores into a probability distribution.
    All outputs sum to 1.0 — used for multiclass output layer.
    
    Formula: e^z / sum(e^z)
    
    The exp shift (z - max) prevents numerical overflow.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_activation(name):
    """Returns activation function and its derivative by name."""
    activations = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "relu": (relu, relu_derivative),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose: {list(activations.keys())}")
    return activations[name]
