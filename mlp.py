import numpy as np


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr

        # Weight initialization (small random values)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    # ---------- Activation Functions ----------
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z_stable)
        return exp / np.sum(exp, axis=1, keepdims=True)

    # ---------- Forward Pass ----------
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    # ---------- Loss ----------
    def cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

    # ---------- Backpropagation ----------
    def backward(self, X, y):
        m = X.shape[0]

        dZ2 = self.A2 - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # ---------- Training ----------
    def train(self, X, y, epochs=200, batch_size=32):
        losses = []

        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                preds = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            full_preds = self.forward(X)
            loss = self.cross_entropy(y, full_preds)
            losses.append(loss)

        return losses