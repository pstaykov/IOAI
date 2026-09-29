# Reference implementation for drill: 04 · Embeddings, prototypes, k-NN and autoencoders (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Frozen ResNet18 embedder ⏱ 10 min
from torchvision import models
_w = models.ResNet18_Weights.DEFAULT
_net = models.resnet18(weights=_w); _net.fc = nn.Identity(); _net = _net.to(device).eval()
_pre = _w.transforms()
@torch.no_grad()
def embed_images(pil_images, bs=200):
    out = []
    for i in range(0, len(pil_images), bs):
        xb = torch.stack([_pre(im.convert("RGB")) for im in pil_images[i:i + bs]]).to(device)
        out.append(_net(xb).cpu())
    return torch.cat(out)

# %% [s2] 2. Prototype classifier ⏱ 6 min
def fit_prototypes(E, y, n_classes):
    Z = F.normalize(E, dim=1)
    return torch.stack([Z[y == c].mean(0) for c in range(n_classes)])
def predict_prototypes(E, P):
    return (F.normalize(E, dim=1) @ F.normalize(P, dim=1).T).argmax(1)

# %% [s3] 3. k-NN from scratch ⏱ 8 min
def knn_predict(E_train, y_train, E_test, k):
    sims = F.normalize(E_test, dim=1) @ F.normalize(E_train, dim=1).T
    nn_idx = sims.topk(k, dim=1).indices
    votes = F.one_hot(y_train[nn_idx], int(y_train.max()) + 1).sum(1)
    return votes.argmax(1)

# %% [s4] 4. Autoencoder features ⏱ 12 min
def train_autoencoder(X, code_dim=32, steps=1500):
    X = X.to(device)
    enc = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, code_dim)).to(device)
    dec = nn.Sequential(nn.Linear(code_dim, 128), nn.ReLU(), nn.Linear(128, X.shape[1])).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3)
    for _ in range(steps):
        xb = X[torch.randint(0, len(X), (256,), device=device)]
        loss = F.mse_loss(dec(enc(xb)), xb); opt.zero_grad(); loss.backward(); opt.step()
    return enc.cpu().eval()

# %% [s5] 5. Seven classes from a frozen five-way head ⏱ 12 min
def seven_way(E):
    Z = F.normalize(E, dim=1)
    P = torch.stack([F.normalize(E_old[y_old == c], dim=1).mean(0) for c in range(5)] +
                    [F.normalize(E_few[y_few == c], dim=1).mean(0) for c in (5, 6)])
    route = (Z @ F.normalize(P, dim=1).T).argmax(1)
    return torch.where(route >= 5, route, head(Z).argmax(1))
