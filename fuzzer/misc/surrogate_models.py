"""
Surrogate Models for SAMOTA — adapted from ICSE-SAMOTA.

Three surrogate models + weighted ensemble:
  - RBFSurrogate: Radial Basis Function neural network (Keras)
  - KrigingSurrogate: Gaussian Process Regression
  - PolynomialSurrogate: Polynomial Regression

These models operate on flattened scenario feature vectors extracted from
ScenarioConfig objects, NOT on the raw configs directly.

The ensemble combines predictions using inverse-MAE weighting.
"""

import copy
import numpy as np
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBF_Kernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════
#  Kriging (Gaussian Process)
# ═══════════════════════════════════════════════════════════════════════════

class KrigingSurrogate:
    """Gaussian Process surrogate model."""

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.model = None
        self.mae = float('inf')

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train on feature matrix X (n, d) and targets y (n,)."""
        y = np.clip(y, 0, 1)
        X_scaled = self.scaler.fit_transform(X)
        kernel = 1.0 * RBF_Kernel(1.0)
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        self.model.fit(X_scaled, y)

    def predict(self, x: np.ndarray) -> float:
        """Predict for a single feature vector (d,)."""
        x = np.array(x).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled)
        return float(np.clip(pred.flatten()[0], 0, 1))

    def test(self, X: np.ndarray, y: np.ndarray):
        """Compute MAE on test set."""
        y = np.clip(y, 0, 1)
        total = 0.0
        for i in range(len(X)):
            pred = self.predict(X[i])
            total += abs(y[i] - pred)
        self.mae = total / max(len(X), 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Polynomial Regression
# ═══════════════════════════════════════════════════════════════════════════

class PolynomialSurrogate:
    """Polynomial Regression surrogate model."""

    def __init__(self, degree=2):
        self.degree = degree
        self.scaler = preprocessing.StandardScaler()
        self.poly_features = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        self.mae = float('inf')

    def train(self, X: np.ndarray, y: np.ndarray):
        y = np.clip(y, 0, 1)
        X_scaled = self.scaler.fit_transform(X)
        X_poly = self.poly_features.fit_transform(X_scaled)
        self.model.fit(X_poly, y)

    def predict(self, x: np.ndarray) -> float:
        x = np.array(x).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        x_poly = self.poly_features.transform(x_scaled)
        pred = self.model.predict(x_poly)
        return float(np.clip(pred.flatten()[0], 0, 1))

    def test(self, X: np.ndarray, y: np.ndarray):
        y = np.clip(y, 0, 1)
        total = 0.0
        for i in range(len(X)):
            pred = self.predict(X[i])
            total += abs(y[i] - pred)
        self.mae = total / max(len(X), 1)


# ═══════════════════════════════════════════════════════════════════════════
#  RBF Neural Network (simplified, using sklearn instead of Keras for
#  portability — avoids TF dependency in the fuzzer venv)
# ═══════════════════════════════════════════════════════════════════════════

class RBFSurrogate:
    """RBF surrogate using sklearn's Kernel Ridge Regression as a
    lightweight replacement for the Keras RBF network."""

    def __init__(self, n_neurons=10):
        self.n_neurons = n_neurons
        self.scaler = preprocessing.StandardScaler()
        self.mae = float('inf')
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.kernel_ridge import KernelRidge
        y = np.clip(y, 0, 1)
        X_scaled = self.scaler.fit_transform(X)
        self.model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
        self.model.fit(X_scaled, y)

    def predict(self, x: np.ndarray) -> float:
        x = np.array(x).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled)
        return float(np.clip(pred.flatten()[0], 0, 1))

    def test(self, X: np.ndarray, y: np.ndarray):
        y = np.clip(y, 0, 1)
        total = 0.0
        for i in range(len(X)):
            pred = self.predict(X[i])
            total += abs(y[i] - pred)
        self.mae = total / max(len(X), 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Ensemble Model
# ═══════════════════════════════════════════════════════════════════════════

class EnsembleSurrogate:
    """Weighted ensemble of RBF + Polynomial + Kriging.

    Weights are computed as inverse-MAE (models with lower MAE get more weight).
    Uncertainty is estimated as max pairwise prediction difference.
    """

    def __init__(self, objective_index: int, poly_degree=2, n_rbf_neurons=10):
        self.objective_index = objective_index
        self.rbf = RBFSurrogate(n_rbf_neurons)
        self.poly = PolynomialSurrogate(poly_degree)
        self.kriging = KrigingSurrogate()
        self.w_rbf = 1 / 3
        self.w_poly = 1 / 3
        self.w_kriging = 1 / 3

    def train(self, X: np.ndarray, y: np.ndarray, test_ratio=0.2):
        """Train all three models and compute ensemble weights."""
        n = len(X)
        n_test = max(1, int(n * test_ratio))
        indices = np.random.permutation(n)
        train_idx, test_idx = indices[n_test:], indices[:n_test]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train
        try:
            self.rbf.train(X_train, y_train)
            self.rbf.test(X_test, y_test)
        except Exception as e:
            logger.warning(f"RBF training failed: {e}")
            self.rbf.mae = float('inf')

        try:
            self.poly.train(X_train, y_train)
            self.poly.test(X_test, y_test)
        except Exception as e:
            logger.warning(f"Poly training failed: {e}")
            self.poly.mae = float('inf')

        try:
            self.kriging.train(X_train, y_train)
            self.kriging.test(X_test, y_test)
        except Exception as e:
            logger.warning(f"Kriging training failed: {e}")
            self.kriging.mae = float('inf')

        # Compute weights (inverse MAE)
        total_mae = self.rbf.mae + self.poly.mae + self.kriging.mae
        if total_mae > 0 and total_mae < float('inf'):
            self.w_rbf = 0.5 * (total_mae - self.rbf.mae) / total_mae
            self.w_poly = 0.5 * (total_mae - self.poly.mae) / total_mae
            self.w_kriging = 0.5 * (total_mae - self.kriging.mae) / total_mae
        else:
            self.w_rbf = self.w_poly = self.w_kriging = 1 / 3

        logger.debug(f"Ensemble obj={self.objective_index}: "
                     f"w_rbf={self.w_rbf:.3f} w_poly={self.w_poly:.3f} w_kr={self.w_kriging:.3f}")

    def predict(self, x: np.ndarray):
        """Return (prediction, uncertainty) tuple."""
        try:
            y_rbf = self.rbf.predict(x)
        except Exception:
            y_rbf = 0.5
        try:
            y_poly = self.poly.predict(x)
        except Exception:
            y_poly = 0.5
        try:
            y_kr = self.kriging.predict(x)
        except Exception:
            y_kr = 0.5

        prediction = y_rbf * self.w_rbf + y_poly * self.w_poly + y_kr * self.w_kriging
        uncertainty = max(abs(y_rbf - y_poly), abs(y_rbf - y_kr), abs(y_poly - y_kr))

        return float(np.clip(prediction, 0, 1)), float(uncertainty)
