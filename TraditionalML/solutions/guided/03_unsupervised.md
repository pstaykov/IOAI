# Solutions: 03 · Unsupervised and semi-supervised learning (guided)

One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).

## g1: smallest k with cumulative explained variance >= 0.90

Placeholder:
```python
k = ____
```
Answer:
```python
k = int(np.searchsorted(cum, 0.90) + 1)
```

## g2: 10 clusters, 10 random restarts (k-means only finds a local optimum), random_state=0

Placeholder:
```python
km = ____.fit(Z)
```
Answer:
```python
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Z)
```

## g3: similarity between true digits y and the cluster labels, invariant to renaming clusters

Placeholder:
```python
ari = ____
```
Answer:
```python
ari = adjusted_rand_score(y, km.labels_)
```

## g4: most frequent label among labels_in_c (fall back to -1 if the cluster has no known label)

Placeholder:
```python
mapping[c] = ____ if len(labels_in_c) else -1
```
Answer:
```python
mapping[c] = int(np.bincount(labels_in_c, minlength=10).argmax()) if len(labels_in_c) else -1
```

## g5: t-SNE has no separate .transform(): fit and embed in one call

Placeholder:
```python
emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).____
```
Answer:
```python
emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Z)
```

## g6: how many points did DBSCAN mark as noise?

Placeholder:
```python
n_noise = ____
```
Answer:
```python
n_noise = int((db.labels_ == -1).sum())
```

## g7: mark every sample we do NOT know (~known) as unlabelled, sklearn style

Placeholder:
```python
y_semi[~known] = ____
```
Answer:
```python
y_semi[~known] = -1
```

## g8: k-NN graph label spreading with 10 neighbours

Placeholder:
```python
ls = ____.fit(Z, y_semi)
```
Answer:
```python
ls = LabelSpreading(kernel="knn", n_neighbors=10).fit(Z, y_semi)
```
