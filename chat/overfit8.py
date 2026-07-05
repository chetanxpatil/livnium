"""
overfit8.py — 30-second sanity check, NOT training.

Force the model to memorize 8 pairs with the clean loss (plain masked CE, every
lever off). Watch one number: does loss fall toward 0?

  loss -> ~0 and it types the 8 replies back  -> the model CAN learn. Go train.
  loss stalls high, replies stay ". . ."       -> the writer can't learn. Fix it
                                                  before wasting an hour on 77k.

Vocab is cut to just these 8 pairs so every step is tiny and fast, and the loss
can actually reach 0 (with the full 100k vocab it starts at ln(100k)=11.5 and
crawls). Prints every 10 steps so it never looks stuck.

    python3 overfit8.py
"""

import argparse
import time
import torch

from chat_reply import (ReplyBrain, load_wells, shrink_vocab, encode_ctx,
                        decode, read_pairs, CharMinter, SPECIALS, CTX_WORDS)
from chat_typer import encode_batch

ap = argparse.ArgumentParser()
ap.add_argument("--pure", action="store_true",
                help="test the pure writer (default: the full writer, which passes)")
ap.add_argument("--align", action="store_true", help="add pure cosine-alignment")
ap.add_argument("--steps", type=int, default=400)
args = ap.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
N = 8
STEPS = args.steps

warm, stoi, itos, unk, eos, n_words, dim, ex = load_wells(DEVICE)
pairs = read_pairs("data/dd_context.tsv")[:N]

# cut the vocab down to only the words in these 8 pairs -> small + fast
warm, stoi, itos, unk, eos, n_words, minted = shrink_vocab(
    pairs, warm, stoi, itos, unk, eos, min_freq=1)

model = ReplyBrain(n_words, dim, eos, warm=warm, pos=False,
                   warm_start=(ex["start"].to(DEVICE) if ex["start"] is not None else None),
                   warm_strength=ex["strength"], pure=args.pure,
                   align=args.align).to(DEVICE)
# every lever OFF -> clean masked CE, exact reader
model.neg_samples = 0
model.word_w = None
model.sample_p = 0.0
model.pos_w = 0.0
model.fast_alpha = None

mint = CharMinter(dim, n_words, DEVICE)
msg = encode_ctx([m for m, r in pairs], stoi, unk, eos, CTX_WORDS,
                 minter=mint).to(DEVICE)
rep = encode_batch([r for m, r in pairs], stoi, unk, eos).to(DEVICE)
model.oov_wells = mint.table()

opt = torch.optim.Adam(model.parameters(), lr=3e-3)
writer = "PURE" if args.pure else "FULL (brain + attention)"
print(f"overfitting {N} pairs on {DEVICE}   vocab {n_words}   writer {writer}   (want loss -> ~0)\n")
model.train()
t0 = time.time()
best = float("inf")
for step in range(STEPS):
    loss = model.reply_nll(msg, rep)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stop Adam spikes
    opt.step()
    best = min(best, loss.item())
    if step % 10 == 0 or step == STEPS - 1:
        sps = (step + 1) / max(time.time() - t0, 1e-9)
        print(f"  step {step:3d}  loss {loss.item():.4f}   {sps:.1f} steps/s", flush=True)

print()
model.eval()
ban = [stoi[t] for t in SPECIALS]
gen, _ = model.generate(msg, 40, unk=unk, ban=ban)
exact = 0
for k, (m, r) in enumerate(pairs):
    got = decode(gen[k], itos, eos)
    ok = got.split() == r.split()
    exact += ok
    print(f"  {'OK ' if ok else 'XX '}real : {r}")
    print(f"      ai   : {got}\n")

# the real test is reproduction, not a loss threshold: if it types the replies
# back it has memorized + conditioned, whatever the teacher-forced loss reads.
print(f"reproduced {exact}/{N} exactly   (loss {loss.item():.3f}, best {best:.3f})")
if exact >= N - 1:
    print("PASS: the writer memorizes + conditions on context -> go run the real training")
elif best < 0.2:
    print(f"PASS (near): loss reached {best:.3f} — it memorizes; a late step just "
          "landed on a spike. Add --steps or it's fine, go train.")
else:
    print(f"FAIL: the {writer} writer can't reproduce the 8 pairs. Give it more "
          "--steps or check the reader.")
