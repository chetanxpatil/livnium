"""
img_collapse_pure.py — image embeddings from raw pixels, PURE vector collapse.

Every attractor is a pixel of a fixed-size image. The wells table IS an
S x S grid: one well per pixel POSITION. An image is read as an ordered
trajectory over its own pixels (raster order), exactly the way
noun_collapse_pure.py reads a context as an ordered word trajectory:

    ENCODE (pure collapse, one attraction step per pixel):
        h <- h - v * strength * (1 - cos(h, W_pix)) * norm(h - W_pix)
    (v = pixel intensity in [0,1] — dark pixels barely pull, bright ones do)

    READOUT (pure geometry, no network):
        logits = cos(h, noun wells) / temp  -> CE against a noun from the
        image's own COCO caption.

Learnable things, in full (the noun_collapse_pure.py discipline):
    - one well per pixel position   (S*S wells — the fixed-size image)
    - one well per caption noun
    - a start state
    - one scalar strength, one scalar temp
    MLP: none.  SVD: none.  Conv: none.  Readout matrix: none.

Compute notes (smarter/faster/cleaner):
    - Images are pre-resized once to S x S grayscale uint8 and cached
      (5000 imgs at S=64 = 20 MB). The JPEG decode never happens twice.
    - The collapse is inherently sequential (S*S steps), so backprop
      memory is bounded with gradient checkpointing in --ckpt-chunk
      pixel chunks instead of storing every step.

Usage (from repo root):
    python3 vision/img_collapse_pure.py --train
    python3 vision/img_collapse_pure.py --probe images/coco/val2017/000000179765.jpg
    python3 vision/img_collapse_pure.py --render dog --render-out dog_map.png

Output: vision/model/img_collapse_pure.pt
    { pix_wells, noun_wells, nouns, start, strength, temp, config }
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

OUT = "vision/model/img_collapse_pure.pt"
_WORD = re.compile(r"[a-z]+")


# ------------------------------------------------------------ data plumbing
# torch-free on purpose: testable without the heavy import.

def caption_words(text):
    return _WORD.findall(text.lower())


def noun_set():
    """WordNet's noun lexicon as a lowercase lookup set (same as noun_embed)."""
    try:
        from nltk.corpus import wordnet as wn
        nouns = {l.name().lower().replace("_", " ")
                 for s in wn.all_synsets(pos="n") for l in s.lemmas()}
        return {n for n in nouns if " " not in n and n.isalpha()}
    except LookupError:
        sys.exit("wordnet missing:  python3 -c \"import nltk; nltk.download('wordnet')\"")
    except ImportError:
        sys.exit("needs nltk:  pip3 install nltk")


def build_targets(captions_path, min_noun_count, max_nouns, no_noun_filter=False):
    """captions json -> (nouns list, image_id -> [noun slot ids], id -> filename)."""
    with open(captions_path) as f:
        d = json.load(f)
    fname = {im["id"]: im["file_name"] for im in d["images"]}
    per_image = defaultdict(list)
    freq = Counter()
    for a in d["annotations"]:
        ws = caption_words(a["caption"])
        per_image[a["image_id"]].extend(ws)
        freq.update(ws)
    keep = set() if no_noun_filter else noun_set()
    nouns = [w for w, c in freq.most_common()
             if c >= min_noun_count and (no_noun_filter or w in keep)]
    nouns = nouns[:max_nouns]
    slot = {w: i for i, w in enumerate(nouns)}
    targets = {}
    for img_id, ws in per_image.items():
        ids = sorted({slot[w] for w in ws if w in slot})
        if ids:
            targets[img_id] = ids
    print(f"  nouns {len(nouns):,} (min-count {min_noun_count})   "
          f"images with targets {len(targets):,}", flush=True)
    return nouns, targets, fname


def build_image_cache(img_dir, ids, fname, size, cache_path):
    """Resize-once cache: (N, S, S) uint8 grayscale + the id order."""
    import torch
    from PIL import Image
    if os.path.exists(cache_path):
        ck = torch.load(cache_path, map_location="cpu")
        if ck["size"] == size and ck["ids"] == ids:
            print(f"  image cache hit: {cache_path}", flush=True)
            return ck["imgs"]
    imgs = torch.empty(len(ids), size, size, dtype=torch.uint8)
    for n, img_id in enumerate(ids):
        with Image.open(os.path.join(img_dir, fname[img_id])) as im:
            im = im.convert("L").resize((size, size), Image.BILINEAR)
            imgs[n] = torch.frombuffer(bytearray(im.tobytes()),
                                       dtype=torch.uint8).view(size, size)
        if (n + 1) % 1000 == 0:
            print(f"  cached {n + 1:,}/{len(ids):,}", flush=True)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save({"imgs": imgs, "ids": ids, "size": size}, cache_path)
    print(f"  image cache -> {cache_path} ({imgs.numel() / 1e6:.0f} MB)", flush=True)
    return imgs


# ------------------------------------------------------------ the model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images/coco/val2017")
    ap.add_argument("--captions",
                    default="images/coco/annotations/captions_val2017.json")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--size", type=int, default=64,
                    help="fixed image size S: the wells table is S*S attractors")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--min-noun-count", type=int, default=5)
    ap.add_argument("--max-nouns", type=int, default=5000)
    ap.add_argument("--no-noun-filter", action="store_true",
                    help="skip WordNet: every frequent caption word is a target")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--neg", type=int, default=512,
                    help="sampled-softmax noun negatives (0 = all nouns)")
    ap.add_argument("--ckpt-chunk", type=int, default=256,
                    help="pixels per gradient-checkpoint chunk (memory bound)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--probe", nargs="*", default=None,
                    help="image path(s): print top-8 caption nouns")
    ap.add_argument("--render", default=None,
                    help="a noun: render cos(pixel well, noun well) as an S x S image")
    ap.add_argument("--render-out", default="render.png")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(0)
    S = args.size

    # ---------------- pure collapse over the pixel grid --------------------
    def collapse_span(h, wells_raw, vals, s):
        """One attraction step per pixel in [span]. h (B,D), wells (K,D), vals (B,K)."""
        A = F.normalize(wells_raw, dim=-1)
        for i in range(A.size(0)):
            t = A[i]                                       # (D,)
            v = vals[:, i:i + 1]                           # (B,1)
            align = (F.normalize(h, dim=-1) * t).sum(-1, keepdim=True)
            away = F.normalize(h - t, dim=-1)
            h = h - v * s * (1.0 - align) * away
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def encode(pix_wells, start, log_strength, imgs_u8, grad=True):
        """imgs_u8 (B,S,S) -> final state (B,D). Raster-order trajectory."""
        vals = imgs_u8.to(device).float().view(imgs_u8.size(0), -1) / 255.0
        h = start.expand(vals.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        C = args.ckpt_chunk
        for c0 in range(0, S * S, C):
            c1 = min(c0 + C, S * S)
            if grad:
                h = checkpoint(collapse_span, h, pix_wells[c0:c1],
                               vals[:, c0:c1], s, use_reentrant=False)
            else:
                h = collapse_span(h, pix_wells[c0:c1], vals[:, c0:c1], s)
        return h

    # ---------------- probe / render an existing model ---------------------
    if args.probe is not None or args.render:
        ck = torch.load(args.out, map_location=device)
        nouns = ck["nouns"]
        pix = ck["pix_wells"].to(device)
        AN = F.normalize(ck["noun_wells"].to(device), dim=-1)
        start, log_s = ck["start"].to(device), ck["log_strength"].to(device)
        cs = ck["config"]["size"]
        if args.render:
            if args.render not in nouns:
                sys.exit(f"'{args.render}' is not a trained noun")
            from PIL import Image
            sim = (F.normalize(pix, dim=-1) @ AN[nouns.index(args.render)])
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            img = (sim.view(cs, cs) * 255).byte().cpu().numpy()
            Image.fromarray(img, mode="L").resize((256, 256), Image.NEAREST) \
                 .save(args.render_out)
            print(f"'{args.render}' attractor map -> {args.render_out}")
        for p in args.probe or []:
            from PIL import Image
            with Image.open(p) as im:
                im = im.convert("L").resize((cs, cs), Image.BILINEAR)
                u8 = torch.frombuffer(bytearray(im.tobytes()),
                                      dtype=torch.uint8).view(1, cs, cs)
            with torch.no_grad():
                h = F.normalize(encode(pix, start, log_s, u8, grad=False), dim=-1)
                top = (h @ AN.t()).squeeze(0).topk(8)
            print(f"  {os.path.basename(p):30s} -> " + "  ".join(
                f"{nouns[i]}({v:.2f})" for v, i in zip(top.values, top.indices)))
        return
    if not args.train:
        sys.exit("need --train (or --probe / --render)")

    # ---------------- data -------------------------------------------------
    print("pass 1/2: caption nouns ...", flush=True)
    nouns, targets, fname = build_targets(
        args.captions, args.min_noun_count, args.max_nouns, args.no_noun_filter)
    ids = sorted(targets)
    if args.max_images:
        ids = ids[:args.max_images]
    print("pass 2/2: image cache ...", flush=True)
    cache = os.path.join(os.path.dirname(args.out) or ".", f"img_cache_{S}.pt")
    imgs = build_image_cache(args.images, ids, fname, S, cache)

    # (image index, noun slot) pairs — one example per noun occurrence
    pairs = torch.tensor([(n, t) for n, img_id in enumerate(ids)
                          for t in targets[img_id]], dtype=torch.long)
    print(f"  {len(pairs):,} (image, noun) examples", flush=True)

    # ---------------- the whole model: wells + start + 2 scalars -----------
    N = len(nouns)
    pix_wells = torch.nn.Parameter(torch.randn(S * S, args.dim, device=device)
                                   / args.dim ** 0.5)
    noun_wells = torch.nn.Parameter(torch.randn(N, args.dim, device=device)
                                    / args.dim ** 0.5)
    start = torch.nn.Parameter(torch.randn(args.dim, device=device) * 0.05)
    log_strength = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    params = [pix_wells, noun_wells, start, log_strength, log_temp]
    print(f"pure model: {S}x{S} pixel wells + {N:,} noun wells x {args.dim} "
          f"+ start + strength + temp   ({sum(p.numel() for p in params):,} numbers)"
          f"   device {device}", flush=True)

    def loss_fn(img_idx, tgt):
        h = F.normalize(encode(pix_wells, start, log_strength, imgs[img_idx]), dim=-1)
        A = F.normalize(noun_wells, dim=-1)
        temp = F.softplus(log_temp) + 1e-3
        if args.neg > 0:
            pos = (h * A[tgt]).sum(-1, keepdim=True) / temp
            neg = torch.randint(0, N, (args.neg,), device=device)
            cand = torch.cat([pos, (h @ A[neg].t()) / temp], dim=1)
            return F.cross_entropy(
                cand, torch.zeros(cand.size(0), dtype=torch.long, device=device))
        return F.cross_entropy((h @ A.t()) / temp, tgt)

    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"pix_wells": pix_wells.detach().cpu(),
                    "noun_wells": noun_wells.detach().cpu(),
                    "nouns": nouns,
                    "start": start.detach().cpu(),
                    "log_strength": log_strength.detach().cpu(),
                    "strength": torch.sigmoid(log_strength).item(),
                    "temp": (F.softplus(log_temp) + 1e-3).item(),
                    "config": {"size": S, "dim": args.dim}}, args.out)

    # ---------------- train -------------------------------------------------
    import time
    opt = torch.optim.Adam(params, lr=args.lr)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        for b0 in range(0, len(pairs) - args.batch + 1, args.batch):
            batch = pairs[torch.randint(0, len(pairs), (args.batch,))] \
                if ep or b0 else pairs[b0:b0 + args.batch]
            img_idx, tgt = batch[:, 0], batch[:, 1].to(device)
            loss = loss_fn(img_idx, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(f"ep {ep} step {step:6d}  loss {loss.item():.4f}  "
                      f"strength {torch.sigmoid(log_strength).item():.3f}  "
                      f"| {time.time() - t0:.0f}s", flush=True)
            if args.save_every and step % args.save_every == 0:
                save(); print(f"  [checkpoint -> {args.out}]", flush=True)
            if args.max_steps and step >= args.max_steps:
                print(f"  [step budget {args.max_steps} reached]", flush=True)
                save(); print(f"saved -> {args.out}"); return
    save()
    print(f"done: {step:,} steps\nsaved -> {args.out}")
    print("probe:   python3 vision/img_collapse_pure.py --probe <image.jpg>")
    print("render:  python3 vision/img_collapse_pure.py --render dog")


if __name__ == "__main__":
    main()
