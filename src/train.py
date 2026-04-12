# train.py
# Loads MNIST, trains the neural network, and plots results.
# MNIST = 70,000 handwritten digit images (0-9)
# This is the "hello world" of neural networks.
# 
# Run this file to train the full network:
#   python src/train.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from network import NeuralNetwork


def load_mnist():
    """
    Loads MNIST dataset via scikit-learn.
    Returns normalized X (pixels) and binary y (labels).
    
    784 features = one per pixel in a 28x28 image.
    Values normalized 0-1 so gradients stay stable.
    """
    print("Loading MNIST dataset (this may take a moment)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype("float32") / 255.0  # Normalize 0-1
    y = mnist.target.astype(int)

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
    return X, y


def binary_task(X, y, digit_a=0, digit_b=1):
    """
    Filters MNIST to a binary classification task.
    Default: distinguish 0s from 1s.
    
    Why binary first?
        Our network uses sigmoid output (0 or 1).
        Full 10-class needs softmax + categorical cross-entropy.
        Binary is the right starting point — walk before you run.
    """
    mask = (y == digit_a) | (y == digit_b)
    X_filtered = X[mask]
    y_filtered = (y[mask] == digit_b).astype(int)  # digit_b = 1, digit_a = 0
    print(f"Binary task: {digit_a} vs {digit_b} | Samples: {len(y_filtered)}\n")
    return X_filtered, y_filtered


def plot_training_curves(loss_history, accuracy_history):
    """
    Plots loss and accuracy over training epochs.
    Saved to results/ folder.
    
    What to look for:
        Loss decreasing = network is learning
        Accuracy increasing = predictions improving
        Loss plateauing = try lower learning rate or more epochs
        Loss spiking = learning rate too high
    """
    epochs = range(len(loss_history))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, loss_history, color="#f38ba8", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, accuracy_history, color="#cba6f7", linewidth=2)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.suptitle("Neural Network From Scratch — MNIST Training", y=1.02)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Training curves saved to results/")


def visualize_predictions(X_test, y_test, network, n=10):
    """
    Shows n test images with predicted vs actual labels.
    Visual sanity check — lets you see what the network gets right and wrong.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    indices = np.random.choice(len(X_test), n, replace=False)

    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        pred = network.predict(X_test[idx].reshape(1, -1))[0]
        actual = y_test[idx]
        correct = pred == actual

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"Pred: {pred} | True: {actual}",
            color="#a6e3a1" if correct else "#f38ba8",
            fontsize=9
        )
        axes[i].axis("off")

    plt.suptitle("Sample Predictions (green = correct, red = wrong)")
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png", dpi=150)
    plt.show()
    print("Sample predictions saved to results/")


if __name__ == "__main__":

    # ── Config ───────────────────────────────────────
    LEARNING_RATE = 0.01
    EPOCHS = 1000
    PRINT_EVERY = 100
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    # ─────────────────────────────────────────────────

    # Load + prepare data
    X, y = load_mnist()
    X, y = binary_task(X, y, digit_a=0, digit_b=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    # Build network
    # 784 inputs (pixels) → 128 → 64 → 1 output
    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 1],
        activations=["relu", "relu", "sigmoid"],
        learning_rate=LEARNING_RATE
    )

    print(f"Network: {nn}\n")

    # Train
    nn.train(X_train, y_train, epochs=EPOCHS, print_every=PRINT_EVERY)

    # Evaluate
    y_pred = nn.forward(X_test)
    test_accuracy = nn.compute_accuracy(y_pred, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_training_curves(nn.loss_history, nn.accuracy_history)
    visualize_predictions(X_test, y_test, nn)
