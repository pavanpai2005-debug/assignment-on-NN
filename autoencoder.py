import numpy as np


class Autoencoder:
    def __init__(self, input_dim, hidden_dim, latent_dim, lr=0.01, sparsity_lambda=0.0):
        self.lr = lr
        self.sparsity_lambda = sparsity_lambda

        # Encoder
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b2 = np.zeros((1, latent_dim))

        # Decoder
        self.W3 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b3 = np.zeros((1, hidden_dim))

        self.W4 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b4 = np.zeros((1, input_dim))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.latent = self.Z2  # Bottleneck

        self.Z3 = self.latent @ self.W3 + self.b3
        self.A3 = self.relu(self.Z3)

        self.Z4 = self.A3 @ self.W4 + self.b4
        self.output = self.Z4

        return self.output

  
    def mse(self, X, X_hat):
        return np.mean((X - X_hat) ** 2)

    def sparse_penalty(self):
        return self.sparsity_lambda * np.mean(np.abs(self.latent))

 
    def backward(self, X):
        m = X.shape[0]

        dZ4 = (self.output - X) / m
        dW4 = self.A3.T @ dZ4
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self.relu_derivative(self.Z3)
        dW3 = self.latent.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dLatent = dZ3 @ self.W3.T
        dLatent += self.sparsity_lambda * np.sign(self.latent)

        dW2 = self.A1.T @ dLatent
        db2 = np.sum(dLatent, axis=0, keepdims=True)

        dA1 = dLatent @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update
        for W, dW in zip(
            [self.W1, self.W2, self.W3, self.W4],
            [dW1, dW2, dW3, dW4]
        ):
            W -= self.lr * dW
-
    def train(self, X, epochs=300):
        losses = []

        for _ in range(epochs):
            recon = self.forward(X)
            loss = self.mse(X, recon) + self.sparse_penalty()
            losses.append(loss)
            self.backward(X)

        return losses