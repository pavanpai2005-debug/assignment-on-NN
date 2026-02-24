import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from mlp import MLP
from autoencoder import Autoencoder
from rbm import RBM


# ---------- Dataset ----------
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ---------- MLP ----------
mlp = MLP(input_dim=2, hidden_dim=16, output_dim=2, lr=0.1)
mlp_losses = mlp.train(X_train, y_train)

# ---------- Autoencoder ----------
auto = Autoencoder(input_dim=2, hidden_dim=8, latent_dim=1,
                   lr=0.01, sparsity_lambda=0.001)
ae_losses = auto.train(X_train)

# Outlier Detection
reconstructed = auto.forward(X_train)
errors = np.mean((X_train - reconstructed) ** 2, axis=1)
threshold = np.mean(errors) + 2 * np.std(errors)

print("Outlier threshold:", threshold)

# ---------- RBM ----------
rbm = RBM(visible_dim=2, hidden_dim=4)
rbm_errors = rbm.train(X_train)

# ---------- Plot Example ----------
plt.plot(mlp_losses)
plt.title("MLP Loss")
plt.show()