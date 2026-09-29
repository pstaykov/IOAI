# Reference implementation for drill: 02 · Segmentation and density counting (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Dataset ⏱ 4 min
from torch.utils.data import Dataset
class MapDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i].long() + 1

# %% [s2] 2. U-Net ⏱ 12 min
def _blk(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())
class UNet(nn.Module):
    def __init__(self, in_ch, n_classes, w=32):
        super().__init__()
        self.e1, self.e2, self.m = _blk(in_ch, w), _blk(w, 2 * w), _blk(2 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), _blk(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), _blk(2 * w, w)
        self.out = nn.Conv2d(w, n_classes, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); m = self.m(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(m), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)

# %% [s3] 3. Weighted pixel score and Dice ⏱ 7 min
def radar_score(pred, target, bonus=50):
    pred, target = torch.as_tensor(pred), torch.as_tensor(target)
    ok, bg = pred == target, target == -1
    return float(((ok & bg).sum() + bonus * (ok & ~bg).sum()) / (bg.sum() + bonus * (~bg).sum()))
def dice(pred_mask, true_mask, eps=1e-6):
    p, t = torch.as_tensor(pred_mask).bool(), torch.as_tensor(true_mask).bool()
    return float((2 * (p & t).sum() + eps) / (p.sum() + t.sum() + eps))

# %% [s4] 4. Train the segmenter ⏱ 12 min
seg = UNet(3, 3).to(device)
opt = torch.optim.AdamW(seg.parameters(), lr=2e-3)
lf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 50.0, 50.0], device=device))
for _ in range(8):
    seg.train()
    for xb, yb in DataLoader(MapDataset(X_tr, Y_tr), batch_size=32, shuffle=True):
        loss = lf(seg(xb.to(device)), yb.to(device)); opt.zero_grad(); loss.backward(); opt.step()

# %% [s5] 5. Counting by density ⏱ 14 min
cnet = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                     nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv2d(64, 1, 1), nn.Softplus()).to(device)
target4 = F.avg_pool2d(den_tr, 4) * 16 * 100
opt = torch.optim.Adam(cnet.parameters(), 1e-3)
for _ in range(700):
    idx = torch.randint(0, len(img_tr), (32,))
    loss = F.mse_loss(cnet(img_tr[idx].to(device)), target4[idx].to(device)); opt.zero_grad(); loss.backward(); opt.step()
@torch.no_grad()
def count_images(imgs):
    return (cnet(imgs.to(device)) / 100).sum((1, 2, 3)).cpu()
