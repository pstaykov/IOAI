# Solutions: 03 · Fine-tuning and extending a deployed model (guided)

One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).

## g1: load a state dict that does NOT contain every key of the model (it returns missing/unexpected keys)

Placeholder:
```python
res = ____
```
Answer:
```python
res = model.load_state_dict(backbone_state, strict=False)
```

## g2: copy the old head's weight and bias into rows 0..4 of the new head

Placeholder:
```python
____
```
Answer:
```python
model.head.weight[:5] = state["head.weight"]; model.head.bias[:5] = state["head.bias"]
```

## g3: stop gradients for every parameter of frozen.features

Placeholder:
```python
for p in frozen.features.parameters():
____
```
Answer:
```python
for p in frozen.features.parameters():
    p.requires_grad = False
```

## g4: one weight per SAMPLE = 1 / (number of samples of its class)

Placeholder:
```python
weights = ____
```
Answer:
```python
weights = 1.0 / counts[y_mix]
```

## g5: the teacher is a fixed target: inference mode and no gradients

Placeholder:
```python
____
```
Answer:
```python
teacher.eval(); teacher.requires_grad_(False)
```

## g6: KL(teacher || student) on the OLD-class logits, temperature T, scaled by T*T

Placeholder:
```python
return ____
```
Answer:
```python
return F.kl_div(F.log_softmax(logits[:, :5] / T, dim=1), F.softmax(t_logits / T, dim=1), reduction="batchmean") * T * T
```

## g7: two parameter groups: features at lr 1e-4, head at lr 1e-3

Placeholder:
```python
opt = torch.optim.Adam(____)
```
Answer:
```python
opt = torch.optim.Adam([{"params": student.features.parameters(), "lr": 1e-4}, {"params": student.head.parameters(), "lr": 1e-3}])
```

## g8: LoRA rank 8 (alpha 16) on the backbone's Linear "features.9"; keep "head" fully trainable

Placeholder:
```python
config = LoraConfig(____)
```
Answer:
```python
config = LoraConfig(r=8, lora_alpha=16, target_modules=["features.9"], modules_to_save=["head"])
```
