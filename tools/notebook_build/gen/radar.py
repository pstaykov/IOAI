import numpy as np
def _radar(n, seed, H=48, W=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    X = rng.gamma(2.0, 0.1, (n, 4, H, W)).astype(np.float32)
    Y = -np.ones((n, H, W), np.int64)
    SIG = {0: [1.0, 0.25, 0.7, 0.2], 1: [0.9, 0.3, 0.6, 0.3], 2: [0.5, 0.9, 0.4, 0.8], "ghost": [0.4, 0.45, 0.9, 0.4]}
    SIZE = {0: ((1.0, 2.0), (1.5, 2.5)), 1: ((1.5, 2.5), (3.0, 5.5)), 2: ((2.5, 4.0), (1.5, 2.8)), "ghost": ((1.2, 3.0), (1.5, 4.5))}
    for i in range(n):
        X[i] += 0.15 * np.sin(yy / rng.uniform(4, 12) + rng.uniform(0, 6))[None] * rng.uniform(0, 1, (4, 1, 1))
        for _ in range(rng.integers(0, 2)):                         # walls: dashed horizontal bands, class 3
            r = rng.integers(2, H - 3); t = rng.integers(1, 3)
            dash = (np.sin(xx[0] / rng.uniform(2, 5) + rng.uniform(0, 6)) > -0.3)
            band = np.zeros((H, W), bool); band[r:r + t, :] = dash[None, :]
            X[i][:, band] += (np.array([0.9, 0.1, 0.9, 0.1]) * rng.uniform(0.2, 0.45))[:, None]
            Y[i][band] = 3
        items = [int(c) for c in rng.integers(0, 3, rng.integers(1, 5))] + ["ghost"] * int(rng.integers(0, 3))
        for c in items:
            (a0, a1), (b0, b1) = SIZE[c]
            cy, cx = rng.uniform(4, H - 4), rng.uniform(6, W - 6)
            blob = ((yy - cy) / rng.uniform(a0, a1)) ** 2 + ((xx - cx) / rng.uniform(b0, b1)) ** 2 <= 1
            X[i][:, blob] += (rng.uniform(0.2, 0.6) * np.array(SIG[c]) * rng.uniform(0.7, 1.3, 4))[:, None]
            if c != "ghost": Y[i][blob] = c
    return np.clip(X / X.max(axis=(2, 3), keepdims=True), 0, 1), Y
