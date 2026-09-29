# Solutions: 02 · Feature engineering for a fixed model (guided)

One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).

## g1: stack the four 1-D arrays as COLUMNS -> shape (n, 4)

Placeholder:
```python
return np.stack([sin_term, sq_term, X[:, 3], X[:, 4]], ____)
```
Answer:
```python
return np.stack([sin_term, sq_term, X[:, 3], X[:, 4]], axis=1)
```

## g2: turn the flat (n, 64) rows back into (n, 8, 8) images

Placeholder:
```python
imgs_tr = ____
```
Answer:
```python
imgs_tr = Xd_tr.reshape(-1, 8, 8)
```

## g3: total ink in each ROW of each image -> shape (n, 8)

Placeholder:
```python
row_mass = imgs.sum(____)
```
Answer:
```python
row_mass = imgs.sum(axis=2)
```

## g4: left-right asymmetry: |image - mirrored image| summed over each whole image -> shape (n,)

Placeholder:
```python
lr_asym = ____
```
Answer:
```python
lr_asym = np.abs(imgs - imgs[:, :, ::-1]).sum(axis=(1, 2))
```

## g5: project the TEST images with the PCA fitted on train (no refitting)

Placeholder:
```python
Z_te = ____
```
Answer:
```python
Z_te = pca.transform(Xd_te)
```

## g6: stack every permuted copy of the whole training set (axis 0 stays the sample axis)

Placeholder:
```python
X_aug = np.concatenate([____ for p in perms], axis=0)
```
Answer:
```python
X_aug = np.concatenate([Xc_tr.transpose((0, *p)) for p in perms], axis=0)
```

## g7: labels for X_aug: the block order is [all samples perm0, all samples perm1, ...]

Placeholder:
```python
y_aug = ____
```
Answer:
```python
y_aug = np.tile(yc_tr, len(perms))
```

## g8: magnitude spectrum along the time axis -> shape (n, T//2 + 1)

Placeholder:
```python
spec = ____
```
Answer:
```python
spec = np.abs(np.fft.rfft(sig, axis=1))
```
