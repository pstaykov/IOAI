import numpy as np
def _paintings():
    from sklearn.datasets import make_moons
    rng = np.random.default_rng(2025)
    P, y = make_moons(n_samples=1800, noise=0.09, random_state=7)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    X = P @ Q[:, :2].T + rng.normal(0, 0.06, size=(1800, 6))
    X = X * np.array([1.0, 40.0, 1.0, 0.02, 1.0, 5.0]) + np.array([0.0, 300.0, -2.0, 0.5, 10.0, 0.0])
    y = np.where(y == 1, 1, -1)
    idx = rng.permutation(1800)
    X, y = X[idx], y[idx]
    y_train = y[:600].copy()
    known = rng.random(600) < 0.03
    y_train[~known] = 0
    return X[:600], y_train, X[600:1200], X[1200:], y[600:1200], y[1200:]
