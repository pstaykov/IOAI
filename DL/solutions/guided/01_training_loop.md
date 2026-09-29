# Solutions: 01 · The PyTorch training loop (guided)

One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).

## g1: first Linear layer: how many inputs does a flattened 28x28 image have?

Placeholder:
```python
nn.Linear(____, 256),
```
Answer:
```python
nn.Linear(28 * 28, 256),
```

## g2: the classification loss for raw logits + int64 class indices

Placeholder:
```python
loss_fn = ____
```
Answer:
```python
loss_fn = nn.CrossEntropyLoss()
```

## g3: move the batch to the same device as the model

Placeholder:
```python
xb, yb = ____
```
Answer:
```python
xb, yb = xb.to(device), yb.to(device)
```

## g4: the three optimizer lines, in the right order

Placeholder:
```python
____
```
Answer:
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## g5: switch dropout/batch-norm to inference behaviour

Placeholder:
```python
____
```
Answer:
```python
model.eval()
```

## g6: don't build the autograd graph while predicting

Placeholder:
```python
with ____:
```
Answer:
```python
with torch.no_grad():
```

## g7: fraction of rows whose highest logit is the true class (a Python float)

Placeholder:
```python
return ____
```
Answer:
```python
return (logits.argmax(dim=1) == y).float().mean().item()
```

## g8: snapshot the weights so later epochs cannot overwrite them

Placeholder:
```python
best_acc, best_state = val_acc, ____
```
Answer:
```python
best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
```
