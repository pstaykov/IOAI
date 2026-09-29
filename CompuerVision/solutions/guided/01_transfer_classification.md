# Solutions: 01 · Transfer learning for image classification (guided)

One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).

## g1: convert uint8 0..255 to float32 0..1

Placeholder:
```python
v2.ToDtype(____),
```
Answer:
```python
v2.ToDtype(torch.float32, scale=True),
```

## g2: normalise with the weights' ImageNet statistics

Placeholder:
```python
____,
```
Answer:
```python
v2.Normalize(mean=MEAN, std=STD),
```

## g3: stack the per-image tensors into (B, 3, H, W) and the labels into an int64 tensor

Placeholder:
```python
return ____
```
Answer:
```python
return torch.stack([r["x"] for r in rows]), torch.tensor([r["y"] for r in rows])
```

## g4: replace the 1000-class ImageNet layer with a fresh 10-class layer (keep its input size)

Placeholder:
```python
model.fc = ____
```
Answer:
```python
model.fc = nn.Linear(model.fc.in_features, 10)
```

## g5: test-time augmentation: average with the logits of the horizontally flipped batch (NCHW: width is dim 3)

Placeholder:
```python
return (model(xb) + model(____)) / 2
```
Answer:
```python
return (model(xb) + model(torch.flip(xb, dims=[3]))) / 2
```

## g6: the GradScaler versions of backward / step, plus the scale update

Placeholder:
```python
____
```
Answer:
```python
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

## g7: freeze every parameter, then unfreeze only the new head

Placeholder:
```python
____
```
Answer:
```python
model.requires_grad_(False); model.fc.requires_grad_(True)
```

## g8: per-class accuracy (recall) = correct predictions of a class / number of samples of that class

Placeholder:
```python
per_class = ____
```
Answer:
```python
per_class = cm.diagonal() / cm.sum(axis=1)
```
