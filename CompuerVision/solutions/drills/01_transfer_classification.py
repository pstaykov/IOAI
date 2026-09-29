# Reference implementation for drill: 01 · Transfer learning for image classification (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Train and eval transforms ⏱ 8 min
from torchvision.transforms import v2
_t = models.ResNet18_Weights.DEFAULT.transforms()
train_tf = v2.Compose([v2.ToImage(), v2.RandomResizedCrop(112, scale=(0.6, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                       v2.RandomVerticalFlip(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(_t.mean, _t.std)])
eval_tf = v2.Compose([v2.ToImage(), v2.Resize(112, antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(_t.mean, _t.std)])

# %% [s2] 2. Pretrained model with a new head ⏱ 5 min
def make_model(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# %% [s3] 3. Loaders and a mixed-precision epoch ⏱ 10 min
def _collate(tf):
    return lambda rows: (torch.stack([tf(r["image"].convert("RGB")) for r in rows]), torch.tensor([r["label"] for r in rows]))
train_loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=_collate(train_tf))
val_loader = DataLoader(val, batch_size=128, collate_fn=_collate(eval_tf))

def train_epoch_amp(model, loader, optimizer, scaler):
    model.train(); total = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total += loss.item() * len(xb); n += len(xb)
    return total / n

# %% [s4] 4. Fine-tune to ≥ 88 % validation accuracy ⏱ 12 min
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); c = n = 0
    for xb, yb in loader:
        c += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item(); n += len(yb)
    return c / n
for _ in range(3):
    train_epoch_amp(m, train_loader, opt, scaler)

# %% [s5] 5. Test-time augmentation ⏱ 6 min
@torch.no_grad()
def predict_tta(model, x):
    model.eval()
    views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
    return torch.stack([model(v).softmax(1) for v in views]).mean(0)
