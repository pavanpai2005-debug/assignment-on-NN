import numpy as np


class RBM:
    def __init__(self, visible_dim, hidden_dim, lr=0.01):
        self.lr = lr
        self.W = np.random.randn(visible_dim, hidden_dim) * 0.01
        self.h_bias = np.zeros(hidden_dim)
        self.v_bias = np.zeros(visible_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, v):
        prob = self.sigmoid(v @ self.W + self.h_bias)
        return prob

    def sample_visible(self, h):
        prob = self.sigmoid(h @ self.W.T + self.v_bias)
        return prob

    def train(self, X, epochs=100):
        errors = []

        for _ in range(epochs):
            # Positive phase
            h_prob = self.sample_hidden(X)

            # Reconstruction
            v_recon = self.sample_visible(h_prob)
            h_recon = self.sample_hidden(v_recon)

            # Contrastive Divergence update
            self.W += self.lr * (X.T @ h_prob - v_recon.T @ h_recon) / X.shape[0]

            error = np.mean((X - v_recon) ** 2)
            errors.append(error)

        return errors