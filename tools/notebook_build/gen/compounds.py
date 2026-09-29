import numpy as np
def _compounds(words):
    rng = np.random.default_rng(2025)
    words = sorted(set(words))
    perm = rng.permutation(len(words))
    seen = [words[i] for i in perm[: int(0.8 * len(words))]]
    unseen = [words[i] for i in perm[int(0.8 * len(words)):]]
    used = set()
    def make(n, p_unseen):
        out = {}
        while len(out) < n:
            k = rng.integers(2, 5)
            parts = []
            for _ in range(k):
                pool = unseen if rng.random() < p_unseen else seen
                parts.append(pool[rng.integers(len(pool))])
            w = "".join(parts)
            if w in used: continue
            used.add(w)
            lab = []
            for p in parts: lab += [0] * (len(p) - 1) + [1]
            out[w] = lab
        return out
    train = make(60000, 0.0)
    val = make(8000, 0.3)
    test = make(8000, 0.3)
    return train, val, test
