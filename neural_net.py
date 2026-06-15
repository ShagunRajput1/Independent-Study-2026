# neural_net.py
import numpy as np

class NeuralNet:
    """
    A 2-layer neural network built from scratch.
    Architecture: 13 inputs → 16 hidden → 8 hidden → 1 output
    """

    def __init__(self, input_size=13, hidden1=16, hidden2=8):
        # Xavier initialization — keeps gradients stable
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2 / hidden2)
        self.b3 = np.zeros((1, 1))

    # ── Activation functions ──────────────────────────────────────────

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # ── Forward pass ─────────────────────────────────────────────────

    def forward(self, X):
        """Run input through the network, return prediction (0-1)."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.output = self.sigmoid(self.z3)
        return self.output

    # ── Backward pass (backpropagation) ──────────────────────────────

    def backward(self, X, y, learning_rate=0.01):
        """Calculate gradients and update weights."""
        m = X.shape[0]  # batch size

        # Output layer gradient
        dL_dout = self.output - y                          # loss derivative
        dout_dz3 = self.output * (1 - self.output)        # sigmoid derivative
        delta3 = dL_dout * dout_dz3

        dW3 = self.a2.T @ delta3 / m
        db3 = np.sum(delta3, axis=0, keepdims=True) / m

        # Hidden layer 2 gradient
        delta2 = (delta3 @ self.W3.T) * self.relu_derivative(self.z2)
        dW2 = self.a1.T @ delta2 / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # Hidden layer 1 gradient
        delta1 = (delta2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = X.T @ delta1 / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    # ── Loss ─────────────────────────────────────────────────────────

    def binary_cross_entropy(self, y_pred, y_true):
        eps = 1e-8  # prevent log(0)
        return -np.mean(y_true * np.log(y_pred + eps) +
                        (1 - y_true) * np.log(1 - y_pred + eps))

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path="model_weights.npz"):
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3)
        print(f" Model saved to {path}")

    def load(self, path="model_weights.npz"):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        print(f" Model loaded from {path}")