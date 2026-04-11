# ⚡ Neural Network From Scratch

> A fully connected neural network built using only NumPy — no TensorFlow, no PyTorch, no shortcuts. Just math, code, and understanding.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## What This Is

This is a feedforward neural network built entirely from scratch using only NumPy. No ML frameworks. No `.fit()`. No abstraction layers hiding the actual math.

Every piece of this — forward propagation, backpropagation, gradient descent, weight updates — is implemented by hand so I actually understand what's happening inside every model I build from here on out.

Trained and tested on the MNIST handwritten digit dataset.

---

## Why I Built This

Most people learning ML skip straight to Keras and TensorFlow. You can get results fast that way, but you become dependent on the black box. You can't debug it when it breaks, you can't explain it in an interview, and you can't innovate beyond what the framework already does.

I'm building toward ML engineering roles in neurotech and healthcare imaging — fields where understanding the math underneath isn't optional. This project exists because I needed to know, not just use.

---

## The Math Behind It

If you want to understand what this code is actually doing:

**Forward Pass**
```
Z = W · X + b         # weighted sum
A = sigmoid(Z)        # activation function
```

**Loss (Binary Cross-Entropy)**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Backpropagation**
```
dL/dW = (1/m) · X · (A - Y)ᵀ     # gradient w.r.t weights
dL/db = (1/m) · Σ(A - Y)          # gradient w.r.t bias
```

**Weight Update (Gradient Descent)**
```
W = W - α · dL/dW
b = b - α · dL/db
```

Where α is the learning rate. That's it. That's the whole thing. Every neural network you've ever seen is a scaled version of exactly this.

---

## Project Structure

```
neural-network-from-scratch/
│
├── notebooks/
│   ├── 01_math_walkthrough.ipynb     # Step-by-step math before any code
│   ├── 02_single_neuron.ipynb        # Build one neuron first
│   └── 03_full_network.ipynb         # Full network on MNIST
│
├── src/
│   ├── activations.py     # Sigmoid, ReLU, softmax + their derivatives
│   ├── layers.py          # Dense layer implementation
│   ├── network.py         # Full network: forward pass, backprop, update
│   ├── loss.py            # Loss functions
│   └── train.py           # Training loop
│
├── data/
│   └── (MNIST loads automatically via fetch_openml)
│
├── results/
│   └── loss_accuracy_curves.png
│
├── requirements.txt
└── README.md
```

---

## How To Run

**1. Clone the repo**
```bash
git clone https://github.com/lindseyphillipsmsu/neural-network-from-scratch.git
cd neural-network-from-scratch
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
*(Only NumPy, Matplotlib, and scikit-learn for data loading — nothing else)*

**3. Run training**
```bash
python src/train.py
```

**4. Or walk through the notebooks in order**
Start with `01_math_walkthrough.ipynb` — it explains every line before you see the code.

---

## Network Configuration

```python
# Default setup
layer_sizes = [784, 128, 64, 10]   # input → hidden → hidden → output
learning_rate = 0.01
epochs = 1000
activation = 'relu'                 # hidden layers
output_activation = 'softmax'       # multiclass output
loss = 'cross_entropy'
```

784 input neurons = one per pixel in a 28×28 MNIST image.
10 output neurons = one per digit (0–9).

---

## Results

| Configuration | Test Accuracy |
|--------------|---------------|
| 1 hidden layer, sigmoid | *updating* |
| 2 hidden layers, ReLU | *updating* |
| 2 hidden layers, ReLU + dropout | *updating* |

---

## What Each File Actually Does

**`activations.py`**
Implements sigmoid, ReLU, and softmax — and critically, their derivatives, which are what backprop actually uses.

**`layers.py`**
A single dense layer class. Stores its own weights and biases. Knows how to do a forward pass and return the gradients from a backward pass.

**`network.py`**
Stacks layers together. Runs the full forward pass, computes loss, runs backprop through every layer in reverse, updates all weights.

**`train.py`**
The training loop. Loads MNIST, runs epochs, prints loss and accuracy, saves results.

---

## What I Learned

- Backpropagation is just the chain rule applied repeatedly — once you see it, you can't unsee it
- Why weight initialization matters (starting at zero kills learning completely)
- The difference between vanishing gradients with sigmoid vs. dying ReLU — both are real and both will wreck training if you're not watching
- Why batch size affects both speed and the quality of gradient estimates
- How learning rate is genuinely one of the most important decisions in the whole process

---

## What's Next

- [ ] Add momentum and Adam optimizer
- [ ] Implement dropout layer properly
- [ ] Extend to a CNN (convolutional layers from scratch)
- [ ] Visualize what each neuron is actually responding to

---

## Connect

Built by **Lindsey Phillips** — AI student at Mississippi State University, working toward ML engineering in neurotech and healthcare imaging.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-lindseyphillipsmsu-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/lindseyphillipsmsu)
[![GitHub](https://img.shields.io/badge/GitHub-lindseyphillipsmsu-181717?style=flat&logo=github)](https://github.com/lindseyphillipsmsu)

---

*If you're learning ML and you haven't built one of these yet — do it. Everything clicks differently after.*
