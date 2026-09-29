# Reference implementation for drill: 02 · Sequences: padding, masks, LSTMs, pooling (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Vocabulary and encoding ⏱ 5 min
def build_vocab(words):
    return {c: i + 1 for i, c in enumerate(sorted({c for w in words for c in w}))}
def encode(word, vocab):
    return torch.tensor([vocab.get(c, 0) for c in word], dtype=torch.long)

# %% [s2] 2. Padding collate ⏱ 7 min
from torch.nn.utils.rnn import pad_sequence
def collate(batch):
    seqs = [encode(w, vocab) for w, _ in batch]
    ys = [torch.tensor(l, dtype=torch.float32) for _, l in batch]
    return (pad_sequence(seqs, batch_first=True), pad_sequence(ys, batch_first=True),
            torch.tensor([len(s) for s in seqs], dtype=torch.long))

# %% [s3] 3. Padding-invariant BiLSTM tagger ⏱ 12 min
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Segmenter(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=96):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)
    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.emb(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
        return self.out(h).squeeze(-1)

# %% [s4] 4. Masked BCE ⏱ 5 min
def masked_bce(logits, y, lengths):
    mask = torch.arange(logits.shape[1], device=logits.device)[None] < lengths.to(logits.device)[:, None]
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])

# %% [s5] 5. Train to segment-F1 ≥ 0.85 ⏱ 12 min
seg_model = Segmenter(len(vocab) + 1).to(device)
opt = torch.optim.AdamW(seg_model.parameters(), lr=3e-3)
loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate)
for epoch in range(3):
    seg_model.train()
    for ids, y, lengths in loader:
        loss = masked_bce(seg_model(ids.to(device), lengths), y.to(device), lengths)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def predict_boundaries(model, words):
    model.eval(); out = []
    for i in range(0, len(words), 512):
        chunk = words[i:i + 512]
        ids, _, lengths = collate([(w, [0] * len(w)) for w in chunk])
        pred = (model(ids.to(device), lengths) > 0).int().cpu()
        for j, n in enumerate(lengths.tolist()):
            p = pred[j, :n].tolist(); p[-1] = 1; out.append(p)
    return out

# %% [s6] 6. An Elman RNN by hand ⏱ 8 min
def rnn_forward(x, h0, W_ih, W_hh, b_ih, b_hh):
    h, outs = h0, []
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        outs.append(h)
    return torch.stack(outs, 1), h
