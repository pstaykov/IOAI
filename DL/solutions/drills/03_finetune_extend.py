# Reference implementation for drill: 03 · Fine-tuning and extending a deployed model (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Grow the head ⏱ 8 min
def grow_head(old_model, n_total):
    new = Net(n_total).to(next(old_model.parameters()).device)
    sd = old_model.state_dict()
    new.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.")}, strict=False)
    k = old_model.head.out_features
    with torch.no_grad():
        new.head.weight[:k] = sd["head.weight"]; new.head.bias[:k] = sd["head.bias"]
    return new

# %% [s2] 2. Freeze the backbone ⏱ 4 min
def freeze_backbone(model):
    for p in model.features.parameters(): p.requires_grad_(False)
    for p in model.head.parameters(): p.requires_grad_(True)
    return model

# %% [s3] 3. Distillation loss ⏱ 6 min
def kd_loss(student_logits, teacher_logits, T):
    return F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1), reduction="batchmean") * T * T

# %% [s4] 4. LoRA from scratch ⏱ 12 min
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

# %% [s5] 5. Class-incremental training ⏱ 15 min
from torch.utils.data import WeightedRandomSampler
def train_incremental(deployed, X_old, y_old, X_new, y_new, epochs=6, T=2.0):
    student, teacher = grow_head(deployed, 10), copy.deepcopy(deployed).eval().requires_grad_(False)
    X, y = torch.cat([X_old, X_new]), torch.cat([y_old, y_new])
    w = 1.0 / torch.bincount(y, minlength=10).float()[y]
    loader = DataLoader(TensorDataset(X, y), batch_size=64, sampler=WeightedRandomSampler(w, len(w), replacement=True))
    opt = torch.optim.Adam([{"params": student.features.parameters(), "lr": 1e-4},
                            {"params": student.head.parameters(), "lr": 1e-3}])
    for _ in range(epochs):
        student.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = student(xb)
            with torch.no_grad(): t = teacher(xb)
            loss = F.cross_entropy(logits, yb) + kd_loss(logits[:, :5], t, T)
            opt.zero_grad(); loss.backward(); opt.step()
    return student
