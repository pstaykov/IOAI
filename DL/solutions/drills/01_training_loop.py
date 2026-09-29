# Reference implementation for drill: 01 · The PyTorch training loop (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Data loaders ⏱ 6 min
from torchvision import datasets, transforms
tf = transforms.ToTensor()
train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# %% [s2] 2. A small CNN ⏱ 7 min
def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 10))

# %% [s3] 3. One epoch ⏱ 6 min
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n

# %% [s4] 4. Evaluation ⏱ 5 min
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = n = 0
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu()
        correct += (pred == yb).sum().item(); n += len(yb)
    return correct / n

# %% [s5] 5. `fit` with a schedule and best-epoch checkpoint ⏱ 12 min
def fit(model, train_loader, val_loader, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = -1, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = evaluate(model, val_loader)
        if acc > best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
    return best_state

# %% [s6] 6. Backprop by hand ⏱ 12 min
def manual_mlp_grads(X, y, W1, b1, W2, b2):
    z = X @ W1 + b1
    h = z.clamp(min=0)
    logits = h @ W2 + b2
    logp = logits - logits.logsumexp(1, keepdim=True)
    n = X.shape[0]
    loss = -logp[torch.arange(n), y].mean()
    d_logits = logp.exp()
    d_logits[torch.arange(n), y] -= 1
    d_logits /= n
    dW2 = h.T @ d_logits; db2 = d_logits.sum(0)
    dh = d_logits @ W2.T
    dz = dh * (z > 0).float()
    dW1 = X.T @ dz; db1 = dz.sum(0)
    return loss, dW1, db1, dW2, db2
