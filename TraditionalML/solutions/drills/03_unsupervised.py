# Reference implementation for drill: 03 · Unsupervised and semi-supervised learning (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. K-means from scratch ⏱ 12 min
def kmeans(X, k, n_iter=100, seed=0):
    r = np.random.default_rng(seed)
    centers = [X[r.integers(len(X))]]
    for _ in range(1, k):                                   # k-means++ seeding
        d2 = ((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(1)
        centers.append(X[r.choice(len(X), p=d2 / d2.sum())])
    centers = np.array(centers, dtype=float)
    labels = None
    for _ in range(n_iter):
        new = ((X[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        if labels is not None and np.array_equal(new, labels): break
        labels = new
        for j in range(k):
            pts = X[labels == j]
            centers[j] = pts.mean(0) if len(pts) else X[r.integers(len(X))]
    return centers, labels

# %% [s2] 2. PCA from scratch ⏱ 10 min
def pca_fit(X, k):
    mean = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k]

def pca_transform(X, mean, components):
    return (X - mean) @ components.T

# %% [s3] 3. Best cluster-to-class matching ⏱ 8 min
from scipy.optimize import linear_sum_assignment
def cluster_accuracy(y_true, clusters):
    y_true, clusters = np.asarray(y_true), np.asarray(clusters)
    cs, ks = np.unique(clusters), np.unique(y_true)
    M = np.array([[np.sum((clusters == c) & (y_true == t)) for t in ks] for c in cs])
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(y_true)

# %% [s4] 4. Clustering two moons ⏱ 7 min
from sklearn.cluster import SpectralClustering
def moons_clusters(X):
    return SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=10, random_state=0).fit_predict(X)

# %% [s5] 5. Semi-supervised with 5 % labels ⏱ 10 min
from sklearn.semi_supervised import LabelSpreading
def semi_supervised_predict(X, y_partial):
    return LabelSpreading(kernel="knn", n_neighbors=10, max_iter=200).fit(X / 16.0, y_partial).transduction_

# %% [s6] 6. A 2-D embedding that keeps the classes apart ⏱ 8 min
from sklearn.manifold import TSNE
def embed_2d(X):
    return TSNE(2, init="pca", random_state=0).fit_transform(PCA(30, random_state=0).fit_transform(X / 16.0))
