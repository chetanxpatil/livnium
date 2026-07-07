"""
experiment/pure_reply.py — 100% Clean-Room Pure Geometric Chat Collapse Engine

This is a parameter-free (MLP-free, linear-free) chat reply generator.
The state trajectory and decode step dynamics are driven entirely by closed-form
energy gradients of V(h) = -cos(h, T). All neural projection layers are removed.
"""

import sys
import os
import argparse
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add chat directory to python path for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../chat"))

from chat_typer import encode_batch, MAXLEN, PAD, SEED
from char_fingerprint import letter_anchors, char_fingerprint
from chat_reply import (
    SPECIALS, CTX_WORDS,
    read_pairs, shrink_vocab, semantic_init, meaning_weights,
    CharMinter, WordMinter, decode, encode_ctx
)

# Default paths pointing to chat/ assets
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "../chat/data/chat_context.tsv")
DEFAULT_TYPER_CKPT = os.path.join(os.path.dirname(__file__), "../chat/model/chat_typer.pt")
DEFAULT_FAST_CKPT = os.path.join(os.path.dirname(__file__), "../chat/model/fast_reader.pt")
DEFAULT_SEMANTIC_INIT = os.path.join(os.path.dirname(__file__), "../chat/model/noun_collapse_pure.pt")
CKPT_OUT_DIR = os.path.join(os.path.dirname(__file__), "model")
CKPT_OUT = os.path.join(CKPT_OUT_DIR, "chat_reply_pure.pt")


# ---------------------------------------------------------------- the model

class PureReplyBrain(nn.Module):
    """100% Pure Geometric Chat Generator. No MLPs, no attention keys/queries."""

    def __init__(self, n_words, dim, eos, warm=None, pos_well=True, align=True):
        super().__init__()
        self.eos, self.dim, self.n_words = eos, dim, n_words
        self.pos_well = bool(pos_well)
        self.align = bool(align)
        
        # training levers, set from outside:
        self.sample_p = 0.0       # scheduled sampling probability
        self.neg_samples = 0      # sampled-softmax negatives (0 = full CE)
        self.word_w = None        # per-word loss weights (meaning weighting)
        self.fast_alpha = None    # distilled reader taps
        self.oov_wells = None     # frozen char wells for OOV words (READ only)

        # Word Anchors: warm-started from the Wikipedia pre-trained collapse wells
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) / dim ** 0.5)
        if warm is not None:
            with torch.no_grad():
                self.word_anchors.copy_(warm)
                self.word_anchors[PAD].zero_()

        # Reader: learned start vector + collapse strength
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength_read = nn.Parameter(torch.tensor(2.2))

        # Writer: position wells + alignment temp + strength + temp
        if self.pos_well:
            self.pos_anchor = nn.Parameter(torch.randn(MAXLEN + 2, dim) * 0.05)
        if self.align:
            self.log_align_temp = nn.Parameter(torch.tensor(0.0))
            
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    @property
    def strength_read(self):
        return torch.sigmoid(self.log_strength_read)

    @property
    def align_temp(self):
        return F.softplus(self.log_align_temp) + 1e-3

    # -- reading ------------------------------------------------------------

    def read(self, ids, A):
        """Sequential context reader using pure analytical energy gradient."""
        B, L = ids.shape
        mask = (ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        s = self.strength_read
        states = []
        for i in range(L):
            target = A[ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            
            h_norm = h.norm(dim=-1, keepdim=True)
            h_n = h / (h_norm + 1e-8)
            align = (h_n * target).sum(-1, keepdim=True)
            
            # Analytical energy gradient of V(h) = -cos(h, target)
            grad = -(target - h_n * align) / (h_norm + 1e-8)
            h = h + m * (-s * grad)
            
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
            
        S = torch.stack(states, dim=1)
        return F.normalize(S, dim=-1), mask, h

    def read_fast(self, ids, A):
        """Distilled convolution reader."""
        mask = (ids != PAD)
        wells = A[ids] * mask.unsqueeze(-1)
        B, L, D = wells.shape
        K = self.fast_alpha.numel()
        x = F.pad(wells.transpose(1, 2), (K - 1, 0))
        w = self.fast_alpha.flip(0).view(1, 1, K).expand(D, 1, K)
        S = F.conv1d(x, w, groups=D).transpose(1, 2)
        lens = mask.sum(1).clamp(min=1) - 1
        hT = S[torch.arange(B, device=ids.device), lens]
        return F.normalize(S + 1e-8, dim=-1), mask, hT

    def read_context(self, ids, A):
        if self.fast_alpha is not None:
            return self.read_fast(ids, A)
        return self.read(ids, A)

    def read_table(self, A):
        if self.oov_wells is None or self.oov_wells.numel() == 0:
            return A
        return torch.cat([A, self.oov_wells], dim=0)

    # -- writing ------------------------------------------------------------

    def collapse_step(self, h, target):
        """Sequential writing step using analytical energy gradient."""
        h_norm = h.norm(dim=-1, keepdim=True)
        h_n = h / (h_norm + 1e-8)
        align = (h_n * target).sum(-1, keepdim=True)
        
        # Analytical energy gradient of V(h) = -cos(h, target)
        grad = -(target - h_n * align) / (h_norm + 1e-8)
        h = h - self.strength * grad
        
        n = h.norm(dim=-1, keepdim=True)
        return torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)

    def reply_nll(self, msg_ids, rep_ids, reduce_mean=True):
        """Teacher-forced NLL of the reply under pure geometric equations."""
        A = F.normalize(self.word_anchors, dim=-1)
        AT = self.read_table(A)
        
        # READ: Context -> trajectory of states & final state
        _, _, hread = self.read_context(msg_ids, AT)
        if self.align:
            Cwells = AT[msg_ids]
            cmask = (msg_ids != PAD)
            
        # THINK: Zero-parameter semantic bridge
        z = hread
        h = z
        
        B, L = rep_ids.shape
        tok_nll = torch.zeros(B, device=rep_ids.device)
        tok_cnt = torch.zeros(B, device=rep_ids.device)
        sampled = self.training and self.neg_samples > 0
        
        for t in range(L):
            hn = F.normalize(h, dim=-1)
            q = hn + z
            
            if self.pos_well:
                q = q + self.pos_anchor[min(t, self.pos_anchor.size(0) - 1)]
                
            if self.align:
                probe = F.normalize(hn + z, dim=-1).unsqueeze(1)
                sc = torch.bmm(probe, Cwells.transpose(1, 2)).squeeze(1) / self.align_temp
                a = torch.softmax(sc.masked_fill(~cmask, float("-inf")), dim=-1)
                q = q + torch.bmm(a.unsqueeze(1), Cwells).squeeze(1)
                
            query = F.normalize(q, dim=-1)
            tgt = rep_ids[:, t]
            
            if sampled:
                pos = (query * A[tgt]).sum(-1, keepdim=True) / self.temp
                neg_ids = torch.randint(1, self.n_words, (self.neg_samples,), device=rep_ids.device)
                neg = (query @ A[neg_ids].t()) / self.temp
                cand = torch.cat([pos, neg], dim=1)
                ll = F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long, device=rep_ids.device), reduction="none")
            else:
                ll = F.cross_entropy((query @ A.t()) / self.temp, tgt, reduction="none")
                
            mask = (tgt != PAD).float()
            if self.training and self.word_w is not None:
                mask = mask * self.word_w[tgt]
                
            tok_nll += ll * mask
            tok_cnt += mask
            
            walk = tgt
            if self.training and self.sample_p > 0:
                coin = (torch.rand(B, device=rep_ids.device) < self.sample_p) & (tgt != PAD)
                if bool(coin.any()):
                    with torch.no_grad():
                        lg = query[coin] @ A.t()
                        lg[:, PAD] = float("-inf")
                        walk = tgt.clone()
                        walk[coin] = lg.argmax(-1)
                        
            h = self.collapse_step(h, A[walk])
            
        if reduce_mean:
            return (tok_nll / tok_cnt.clamp(min=1)).mean()
        return tok_nll, tok_cnt

    @torch.no_grad()
    def generate(self, msg_ids, max_len, unk=None, ban=(),
                 rep_penalty=1.0, no_repeat_bigram=False, temperature=0.0):
        """Free-running decode using pure geometric collapse."""
        A = F.normalize(self.word_anchors, dim=-1)
        AT = self.read_table(A)
        
        traj, tmask, hread = self.read_context(msg_ids, AT)
        z = hread
        h = z
        
        if self.align:
            Cwells = AT[msg_ids]
            cmask = tmask
            
        B = msg_ids.size(0)
        toks = []
        done = torch.zeros(B, dtype=torch.bool, device=msg_ids.device)
        prev = None
        seen_bigrams = [set() for _ in range(B)]
        ar = torch.arange(B, device=msg_ids.device)
        
        for t in range(max_len):
            hn = F.normalize(h, dim=-1)
            q = hn + z
            
            if self.pos_well:
                q = q + self.pos_anchor[min(t, self.pos_anchor.size(0) - 1)]
                
            if self.align:
                probe = F.normalize(hn + z, dim=-1).unsqueeze(1)
                sc = torch.bmm(probe, Cwells.transpose(1, 2)).squeeze(1) / self.align_temp
                a = torch.softmax(sc.masked_fill(~cmask, float("-inf")), dim=-1)
                q = q + torch.bmm(a.unsqueeze(1), Cwells).squeeze(1)
                
            query = F.normalize(q, dim=-1)
            logits = (query @ A.t()) / self.temp
            
            if rep_penalty != 1.0 and toks:
                hist = torch.stack(toks, dim=1)
                g = torch.gather(logits, 1, hist)
                g = torch.where(g > 0, g / rep_penalty, g * rep_penalty)
                logits.scatter_(1, hist, g)
                
            logits[:, PAD] = float("-inf")
            if unk is not None:
                logits[:, unk] = float("-inf")
            for b in ban:
                logits[:, b] = float("-inf")
            if prev is not None:
                logits[ar, prev] = float("-inf")
            if no_repeat_bigram and prev is not None:
                for b in range(B):
                    for (p, y) in seen_bigrams[b]:
                        if p == int(prev[b]):
                            logits[b, y] = float("-inf")
                            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
                
            if prev is not None:
                for b in range(B):
                    seen_bigrams[b].add((int(prev[b]), int(nxt[b])))
                    
            prev = nxt
            toks.append(nxt)
            h = self.collapse_step(h, A[nxt])
            done = done | (nxt == self.eos)
            if bool(done.all()):
                break
                
        T = torch.stack(toks, dim=1)
        return T, None


# ---------------------------------------------------------------- data loader

def load_wells_from_typer(device, path=DEFAULT_TYPER_CKPT):
    ck = torch.load(path, map_location=device)
    warm, stoi, itos = ck["word_anchors"].to(device), dict(ck["stoi"]), dict(ck["itos"])
    n_words, dim = ck["config"]["n_words"], ck["config"]["dim"]
    for tok in SPECIALS:
        stoi[tok] = n_words
        itos[n_words] = tok
        warm = torch.cat([warm, torch.randn(1, dim, device=device) / dim ** 0.5], dim=0)
        n_words += 1
    extras = {"start": ck.get("start"), "strength": ck.get("strength")}
    return warm, stoi, itos, ck["unk"], ck["eos"], n_words, dim, extras


# ---------------------------------------------------------------- chat loop

def chat_loop(args, device):
    ck = torch.load(args.ckpt, map_location=device)
    cfg, stoi, itos = ck["config"], ck["stoi"], ck["itos"]
    unk, eos = ck["unk"], ck["eos"]
    ctx_words = cfg.get("ctx_words", CTX_WORDS)
    
    model = PureReplyBrain(cfg["n_words"], cfg["dim"], eos,
                           pos_well=cfg.get("pos_well", True),
                           align=cfg.get("align", True)).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    
    if ck.get("fast_alpha") is not None:
        model.fast_alpha = ck["fast_alpha"].to(device)
    model.eval()
    
    ban = [stoi[t] for t in SPECIALS if t in stoi]
    oov_src = args.oov_words or DEFAULT_SEMANTIC_INIT
    if os.path.exists(oov_src):
        minter = WordMinter(cfg["dim"], cfg["n_words"], device, oov_src)
    else:
        minter = CharMinter(cfg["dim"], cfg["n_words"], device)
        
    print(f"loaded {args.ckpt}   ctx {ctx_words} words   device {device}")
    print("Pure Collapse Chat: :reset to wipe, :q to quit\n")
    
    from prep_chat_context import clean as clean_text
    history = []
    
    while True:
        try:
            line = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line == ":q":
            break
        if line == ":reset":
            history = []; print("[history wiped]"); continue
            
        clean_line = clean_text(line)
        history.append(f"<you> {clean_line}")
        ctx_str = " ".join(history)
        
        ids = encode_batch([ctx_str], stoi, unk, eos, ctx_words, minter).to(device)
        model.oov_wells = minter.table()
        
        reply_ids, _ = model.generate(ids, max_len=64, unk=unk, ban=ban,
                                      rep_penalty=args.rep_penalty,
                                      no_repeat_bigram=args.no_repeat_bigram,
                                      temperature=args.temperature)
        
        reply_words = decode(reply_ids[0], itos, eos)
        reply = " ".join(reply_words)
        print(f"ai    > {reply}")
        history.append(f"<me> {reply}")
        
        # limit history size
        if len(history) > 20:
            history = history[-20:]


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--typer-ckpt", default=DEFAULT_TYPER_CKPT)
    ap.add_argument("--fast-ckpt", default=DEFAULT_FAST_CKPT)
    ap.add_argument("--ckpt", default=CKPT_OUT)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-lines", type=int, default=0)
    
    # model options
    ap.add_argument("--no-pos-well", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    
    # training params
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--semantic-init", default=DEFAULT_SEMANTIC_INIT)
    ap.add_argument("--oov-words", default="")
    ap.add_argument("--neg-samples", type=int, default=512)
    ap.add_argument("--meaning-w", type=float, default=1.0)
    ap.add_argument("--fast-reader", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    
    # decode options
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--no-repeat-bigram", action="store_true")
    
    args = ap.parse_args()
    device = torch.device(args.device)
    
    if args.chat:
        chat_loop(args, device)
        return
        
    print(f"starting pure collapse chat reply training pipeline on {device}...")
    
    # 1. Load context pairs
    pairs = read_pairs(args.data, args.max_lines)
    print(f"loaded {len(pairs):,} pairs")
    
    random.seed(SEED); random.shuffle(pairs)
    n_dev = max(10, int(len(pairs) * 0.05))
    train_pairs, dev_pairs = pairs[:-n_dev], pairs[-n_dev:]
    
    # 2. Setup Vocabulary & Wells from Typer
    warm, stoi, itos, unk, eos, n_words, dim, extras = load_wells_from_typer(device, args.typer_ckpt)
    
    # Shrink vocabulary to keep negative sampling fast and dense
    warm, stoi, itos, unk, eos, n_words, minted = shrink_vocab(
        train_pairs, warm, stoi, itos, unk, eos, args.min_freq
    )
    print(f"vocab cut: {len(warm):,} wells (min-freq {args.min_freq}; {minted} fresh letter-minted wells)")
    
    # Initialize shared model architecture
    model = PureReplyBrain(n_words, dim, eos, warm=warm,
                           pos_well=not args.no_pos_well,
                           align=not args.no_align).to(device)
                           
    # Load start states and strength priors
    if extras.get("start") is not None:
        with torch.no_grad():
            model.start.copy_(extras["start"])
            
    # Load checkpoint if resuming
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["state_dict"], strict=False)
        print(f"resumed checkpoint: {args.resume}")
    elif os.path.exists(args.semantic_init):
        semantic_init(model, stoi, args.semantic_init, device)
        
    # Read distilled causal reader taps
    if args.fast_reader and os.path.exists(args.fast_ckpt):
        fck = torch.load(args.fast_ckpt, map_location=device)
        model.fast_alpha = fck["alpha"].to(device)
        print(f"loaded causal fast-reader: {args.fast_ckpt}")
        
    # Setup Minter for OOV context words
    oov_src = args.oov_words or args.semantic_init
    if oov_src and os.path.exists(oov_src):
        minter = WordMinter(dim, n_words, device, oov_src)
    else:
        minter = CharMinter(dim, n_words, device)

    # Prepare meaning weights
    # Prepare meaning weights
    train_rep = encode_batch([r for _, r in train_pairs], stoi, unk, eos)
    if args.meaning_w > 0:
        w, most_common = meaning_weights(train_rep, n_words, args.meaning_w)
        model.word_w = w.to(device)
        print(f"meaning weights enabled: common word weight {w[most_common]:.2f}")
        
    # Set model config
    model.neg_samples = args.neg_samples
    
    # Prepare batch indexes
    train_msg = encode_ctx([m for m, _ in train_pairs], stoi, unk, eos, CTX_WORDS, minter=minter)
    train_rep = train_rep.contiguous()
    
    dev_msg = encode_ctx([m for m, _ in dev_pairs], stoi, unk, eos, CTX_WORDS, minter=minter)
    dev_rep = encode_batch([r for _, r in dev_pairs], stoi, unk, eos)
    
    if minter is not None:
        model.oov_wells = minter.table()
        if isinstance(minter, WordMinter):
            print(f"oov wells: {minter.hits:,} real wells <- {oov_src} | {minter.miss:,} fallback")
        else:
            print(f"char wells: minted {len(minter.rows):,} read-wells for OOV context words")
            
    print(f"train size: {train_msg.size(0)} | dev size: {dev_msg.size(0)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_nll = float("inf")
    patience_cnt = 0
    
    # Ensure checkpoint output folder exists
    os.makedirs(CKPT_OUT_DIR, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Annealing schedules
        model.sample_p = min(0.25, 0.25 * (epoch / 10.0))
        
        perm = torch.randperm(train_msg.size(0))
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        
        for i in range(0, train_msg.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            b_msg, b_rep = train_msg[idx].to(device), train_rep[idx].to(device)
            
            optimizer.zero_grad()
            loss = model.reply_nll(b_msg, b_rep)
            loss.backward()
            
            # Clip gradients to keep system stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        # Evaluation pass
        model.eval()
        with torch.no_grad():
            dev_nll = model.reply_nll(dev_msg.to(device), dev_rep.to(device)).item()
            
        dt = time.time() - t0
        print(f"ep {epoch:02d} | loss {epoch_loss/n_batches:.4f} | dev nll {dev_nll:.4f} | sample_p {model.sample_p:.2f} | {dt:.1f}s")
        
        # Print check examples
        if epoch % 5 == 0 or epoch == 1:
            print("\n  --- Dev Check Example ---")
            with torch.no_grad():
                demo_ids = dev_msg[:2].to(device)
                rep_ids, _ = model.generate(demo_ids, max_len=30, unk=unk)
                for b in range(min(2, demo_ids.size(0))):
                    prompt = decode(demo_ids[b], itos, eos)
                    real = decode(dev_rep[b], itos, eos)
                    gen = decode(rep_ids[b], itos, eos)
                    print(f"   you   : {' '.join(prompt[-15:])}")
                    print(f"   real  : {' '.join(real)}")
                    print(f"   ai    : {' '.join(gen)}")
            print()
            
        # Early stopping and checkpointing
        if dev_nll < best_nll:
            best_nll = dev_nll
            patience_cnt = 0
            
            # Save state checkpoint
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "config": {
                    "n_words": n_words,
                    "dim": dim,
                    "pos_well": model.pos_well,
                    "align": model.align,
                    "ctx_words": CTX_WORDS
                },
                "stoi": list(stoi.items()),
                "itos": list(itos.items()),
                "unk": unk,
                "eos": eos,
                "best": {"epoch": epoch, "nll": dev_nll}
            }
            if model.fast_alpha is not None:
                state["fast_alpha"] = model.fast_alpha
            torch.save(state, args.ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"early stopping triggered (best dev nll {best_nll:.4f})")
                break
                
    print(f"best model = model saved to {args.ckpt}")


if __name__ == "__main__":
    main()
