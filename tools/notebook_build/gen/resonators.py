import numpy as np
def _make(n, seed):
    rng = np.random.default_rng(seed)
    fs, T, C = 64, 128, 6
    t = np.arange(T) / fs
    f = rng.uniform(2, 20, n); A = rng.uniform(0.5, 3, n); g = rng.uniform(0.2, 2.0, n)
    gains = rng.uniform(0.5, 1.5, (n, C)); gains /= gains.mean(1, keepdims=True)
    phase = rng.uniform(0, 2 * np.pi, (n, C))
    X = (A[:, None, None] * gains[:, :, None] * np.exp(-g[:, None, None] * t)
         * np.sin(2 * np.pi * f[:, None, None] * t + phase[:, :, None]))
    X = X + rng.normal(0, 0.15, X.shape)
    bad = rng.random(n) < 0.25
    ch = rng.integers(0, C, n)
    dead = rng.random(n) < 0.5
    for i in np.where(bad)[0]:
        if dead[i]:
            X[i, ch[i]] = 0.0
        else:
            k = rng.choice(T, 6, replace=False)
            X[i, ch[i], k] += rng.normal(0, 8, 6)
    Y = np.stack([f, A, g], 1)
    return X.astype(np.float32), Y.astype(np.float32)
