# Reference implementation for drill: 02 · Feature engineering for a fixed model (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Image summary features ⏱ 8 min
def image_features(imgs):
    imgs = np.asarray(imgs, dtype=float)
    rows, cols = imgs.sum(2), imgs.sum(1)
    lr = np.abs(imgs - imgs[:, :, ::-1]).sum((1, 2))
    tb = np.abs(imgs - imgs[:, ::-1, :]).sum((1, 2))
    return np.column_stack([rows, cols, lr, tb])

# %% [s2] 2. Channel-permutation-invariant features ⏱ 8 min
def invariant_features(X):
    X = np.asarray(X, dtype=float)
    per_ch = np.stack([X.mean(2), X.std(2), np.abs(X).max(2), (np.diff(np.sign(X), axis=2) != 0).sum(2)], 2)  # (n, C, 4)
    return np.concatenate([per_ch.mean(1), per_ch.max(1), per_ch.min(1), np.sort(per_ch[:, :, 1], 1)], 1)

# %% [s3] 3. Dominant frequency ⏱ 8 min
def dominant_freq(X, fs):
    X = np.asarray(X, dtype=float)
    P = np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1)) ** 2
    P[:, 0] = 0
    k = np.clip(P.argmax(1), 1, P.shape[1] - 2)
    i = np.arange(len(P))
    a, b_, c = P[i, k - 1], P[i, k], P[i, k + 1]
    delta = 0.5 * (a - c) / (a - 2 * b_ + c + 1e-12)          # parabolic peak interpolation
    return (k + delta) * fs / X.shape[1]

# %% [s4] 4. PCA features without leakage ⏱ 6 min
def pca_features(Xtr, Xte, k):
    p = PCA(n_components=k).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)

# %% [s5] 5. All-axis-permutation augmentation ⏱ 8 min
def augment_axes(X, y):
    perms = list(itertools.permutations([1, 2, 3]))
    return np.concatenate([X.transpose((0, *p)) for p in perms]), np.tile(y, len(perms))

# %% [s6] 6. Beat a frozen `LinearRegression` on Friedman #1 ⏱ 12 min
def friedman_features(X):
    return np.column_stack([np.sin(np.pi * X[:, 0] * X[:, 1]), (X[:, 2] - 0.5) ** 2, X[:, 3], X[:, 4]])
