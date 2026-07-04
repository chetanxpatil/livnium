"""
modeling_noun_collapse.py — standalone loader + encoder for the pure-collapse
Wikipedia noun embeddings. Needs only torch (+ the checkpoint).

The whole "model" is: one 256-d well per word, a start state, and two scalars
(strength, temp). No MLP, no attention, no output layer. Meaning is READ OUT
of the geometry — the same wells that pull are the wells you look up.

    from modeling_noun_collapse import NounCollapse
    m = NounCollapse.from_pretrained("noun_collapse_pure.pt")

    m.vector("physics")                      # the 256-d embedding of one word
    m.neighbors("cat", k=8)                  # nearest NOUNS by cosine
    m.encode(["a cat sat on the mat"])       # collapse a context -> one state
"""

import torch
import torch.nn.functional as F

PAD = 0


class NounCollapse:
    def __init__(self, wells, stoi, noun_ids, start, strength, temp, window):
        self.A = F.normalize(wells, dim=-1)          # unit wells (embeddings)
        self.stoi = stoi
        self.itos = {i: w for w, i in stoi.items()}
        self.noun_ids = torch.tensor(noun_ids)
        self.noun_set = set(noun_ids)
        self.AN = self.A[self.noun_ids]              # noun-only sub-table
        self.start = start
        self.strength = float(strength)
        self.temp = float(temp)
        self.window = window

    @classmethod
    def from_pretrained(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location)
        cfg = ck.get("config", {})
        return cls(ck["wells"], ck["stoi"], ck["noun_ids"], ck["start"],
                   ck["strength"], ck["temp"], cfg.get("window", 5))

    # -- word-level --------------------------------------------------------
    def vector(self, word):
        """Unit embedding of a single word (None if out of vocab)."""
        i = self.stoi.get(word)
        return None if i is None else self.A[i]

    def similarity(self, a, b):
        va, vb = self.vector(a), self.vector(b)
        if va is None or vb is None:
            return None
        return float(va @ vb)

    def neighbors(self, word, k=8):
        """Nearest NOUNS to `word` by cosine (word may be any vocab word)."""
        v = self.vector(word)
        if v is None:
            return []
        sims = self.AN @ v
        if word in self.noun_set:
            sims[list(self.noun_ids).index(self.stoi[word])] = -1e9
        top = sims.topk(min(k, sims.numel()))
        return [(self.itos[int(self.noun_ids[i])], float(s))
                for s, i in zip(top.values, top.indices)]

    # -- context-level (the collapse READ path) ----------------------------
    def encode(self, sentences):
        """Collapse each whitespace-tokenized sentence to ONE state vector.
        Lowercase + split on spaces (use your own tokenizer for parity with
        training; this is the minimal version). Returns (B, dim), unit-norm."""
        rows = []
        for s in sentences:
            rows.append([self.stoi.get(t, PAD) for t in s.lower().split()])
        L = max((len(r) for r in rows), default=1)
        ids = torch.full((len(rows), L), PAD, dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask = ids != PAD
        h = self.start.expand(ids.size(0), -1).contiguous()
        s = self.strength
        for i in range(L):
            t = self.A[ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            align = (F.normalize(h, dim=-1) * t).sum(-1)
            away = F.normalize(h - t, dim=-1)
            h = h + m * (-s * (1.0 - align).unsqueeze(-1) * away)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return F.normalize(h, dim=-1)


if __name__ == "__main__":
    import sys
    m = NounCollapse.from_pretrained(sys.argv[1] if len(sys.argv) > 1
                                     else "noun_collapse_pure.pt")
    for w in ("cat", "physics", "war", "india"):
        print(f"{w:10s} ->", "  ".join(f"{n}({s:.2f})" for n, s in m.neighbors(w)))
