"""
img_from_state_pure.py — can ONE collapsed state hold a whole image?

Level 1 (FROZEN): pixel_color_pure.pt classifies every pixel of an S x S RGB
image into 13 color names — all S*S pixels in one parallel batch. The image
becomes a color map: S*S symbols from a 13-word alphabet.

Level 2 (trained here): 4096 position wells + 13 color wells + start +
strength + temp. ENCODE: h collapses through the image's pixels, each pulling
toward the composite well  norm(W_pos[i] + W_color[label_i]) — position bound
to content by vector ADDITION, no matrix. DECODE (pure geometry): at every
position i, the predicted color is  argmax_c cos(H, norm(W_pos[i] + W_color[c])).
The image must be regenerated from H alone — the wells are shared across all
images, so H is the only place a specific image can live.

This is a CAPACITY experiment. S=64 means 4,096 symbols (~15k bits) squeezed
into one D-dim state; superposition memories hold ~D items, so expect lossy,
low-frequency reconstruction — the interesting result is HOW MUCH survives,
measured against the positional prior (predict each position's most common
color across the dataset, no H at all).

    python3 vision/img_from_state_pure.py --train --device mps            # S=64
    python3 vision/img_from_state_pure.py --train --size 16 --steps 500   # smoke
    python3 vision/img_from_state_pure.py --recon 0 3 7   # side-by-side PNGs
"""

import argparse
import os
import sys

OUT = "vision/model/img_from_state_pure.pt"
L1 = "vision/model/pixel_color_pure.pt"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for pixel_color_pure


def keep_awake():
    if sys.platform == "darwin":
        import subprocess
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass


def build_rgb_cache(img_dir, size, max_images, cache_path):
    """Resize-once RGB cache: (N, S, S, 3) uint8."""
    import torch
    from PIL import Image
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))[:max_images]
    if os.path.exists(cache_path):
        ck = torch.load(cache_path, map_location="cpu")
        if ck["size"] == size and ck["files"] == files:
            print(f"  rgb cache hit: {cache_path}", flush=True)
            return ck["imgs"]
    imgs = torch.empty(len(files), size, size, 3, dtype=torch.uint8)
    for n, f in enumerate(files):
        with Image.open(os.path.join(img_dir, f)) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            imgs[n] = torch.frombuffer(bytearray(im.tobytes()),
                                       dtype=torch.uint8).view(size, size, 3)
        if (n + 1) % 1000 == 0:
            print(f"  cached {n + 1:,}/{len(files):,}", flush=True)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save({"imgs": imgs, "files": files, "size": size}, cache_path)
    print(f"  rgb cache -> {cache_path}", flush=True)
    return imgs


def label_pixels(imgs, l1_path, device):
    """Frozen level 1: (N, S, S, 3) uint8 -> (N, S*S) color labels. Parallel."""
    import torch
    import torch.nn.functional as F
    ck = torch.load(l1_path, map_location=device)
    N, S = imgs.size(0), imgs.size(1)
    rgb = imgs.view(-1, 3).float().to(device) / 255.0            # (N*S*S, 3)
    s = torch.sigmoid(ck["log_strength"])
    A = F.normalize(ck["ch_wells"].to(device), dim=-1)
    cw = F.normalize(ck["color_wells"].to(device), dim=-1)
    out = torch.empty(rgb.size(0), dtype=torch.uint8)
    for b0 in range(0, rgb.size(0), 262144):
        x = rgb[b0:b0 + 262144]
        h = ck["start"].to(device).expand(x.size(0), -1).contiguous()
        for c in range(3):
            t, v = A[c], x[:, c:c + 1]
            align = (F.normalize(h, dim=-1) * t).sum(-1, keepdim=True)
            h = h - v * s * (1.0 - align) * F.normalize(h - t, dim=-1)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        out[b0:b0 + 262144] = (F.normalize(h, dim=-1) @ cw.t()).argmax(1).to(torch.uint8).cpu()
    return out.view(N, S * S), ck["names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images/coco/val2017")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pixel-chunk", type=int, default=64)
    ap.add_argument("--pos-mode", choices=["free", "factored"], default="free",
                    help="factored: W_pos[x,y] = W_row[y] + W_col[x] — 2*S "
                         "structured wells instead of S*S independent ones, so "
                         "neighboring pixels share components (positional "
                         "understanding by construction)")
    ap.add_argument("--pos-init", choices=["rand", "fourier"], default="rand",
                    help="fourier (factored only): init row/col wells with 1D "
                         "sinusoids so METRIC adjacency exists from step 0 — "
                         "nearby indices start similar, decaying with distance")
    ap.add_argument("--relations", action="store_true",
                    help="neighbor transition wells: each pixel's pull also "
                         "carries its RIGHT and DOWN neighbor's color — 2D "
                         "bigrams, so edges/transitions become first-class "
                         "directions in H instead of being invisible")
    ap.add_argument("--scan", choices=["raster", "shuffled"], default="raster",
                    help="shuffled: fixed random pixel order — breaks the "
                         "P-aligned row-chunk confound (P=S means every raster "
                         "chunk IS one image row, so rows get averaged)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--recon", nargs="*", type=int, default=None,
                    help="image indices: save input-vs-reconstruction PNGs")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(0)
    S = args.size
    SS = S * S
    P = args.pixel_chunk

    def composites(pos_wells, color_wells):
        """(SS, 13, D) normalized position+color composite wells."""
        return F.normalize(pos_wells.unsqueeze(1) + color_wells.unsqueeze(0), dim=-1)

    perm = None  # set below if --scan shuffled (fixed, seeded, saved in config)

    def build_traj(pos, cw, lab, wr=None, wd=None):
        """(B, SS, D) trajectory composites: position + color, and with
        relation wells also the RIGHT and DOWN neighbor's color — 2D bigrams,
        so a transition ('brown meets gray here') is its own well direction."""
        T = pos.unsqueeze(0) + cw[lab.long()]                    # (B, SS, D)
        if wr is not None:
            l2 = lab.view(-1, S, S).long()
            labr = torch.cat([l2[:, :, 1:], l2[:, :, -1:]], 2).reshape(-1, SS)
            labd = torch.cat([l2[:, 1:, :], l2[:, -1:, :]], 1).reshape(-1, SS)
            T = T + wr[labr] + wd[labd]
        T = F.normalize(T, dim=-1)
        return T if perm is None else T[:, perm]

    def encode(T, start, log_strength):
        """T (B, SS, D) trajectory -> H (B, D)."""
        h = start.expand(T.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        for c0 in range(0, SS, P):
            t = T[:, c0:c0 + P]                                  # (B, P, D)
            align = (F.normalize(h, dim=-1).unsqueeze(1) * t).sum(-1)   # (B, P)
            away = F.normalize(h.unsqueeze(1) - t, dim=-1)
            h = h - ((s * (1.0 - align)).unsqueeze(-1) * away).sum(1)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def decode_logits(H, M, log_temp):
        temp = F.softplus(log_temp) + 1e-3
        D = H.size(-1)
        return (F.normalize(H, dim=-1) @ M.view(SS * 13, D).t()).view(-1, SS, 13) / temp

    # ---------------- data: frozen level 1 labels every pixel ---------------
    cache = os.path.join(os.path.dirname(args.out) or ".", f"img_cache_rgb_{S}.pt")
    imgs = build_rgb_cache(args.images, S, args.max_images, cache)
    print("labeling pixels with frozen pixel_color_pure ...", flush=True)
    labels, names = label_pixels(imgs, L1, device)               # (N, SS)
    N, C = labels.size(0), len(names)
    labels = labels.to(device)

    # positional prior: per-position most common color (the no-H baseline).
    # computed on CPU: aten::mode isn't implemented on MPS.
    prior = torch.mode(labels.long().cpu(), dim=0).values.to(device)   # (SS,)
    prior_acc = (labels.long() == prior.unsqueeze(0)).float().mean().item()
    print(f"  {N:,} images  {SS} positions  {C} colors   "
          f"positional-prior acc {prior_acc:.3f}  <- beat this or H holds nothing",
          flush=True)

    # ---------------- reconstruct from an existing model --------------------
    def make_perm(scan):
        if scan != "shuffled":
            return None
        g = torch.Generator().manual_seed(0)                     # deterministic
        return torch.randperm(SS, generator=g).to(device)

    if args.recon is not None:
        from PIL import Image
        from pixel_color_pure import COLORS
        ck = torch.load(args.out, map_location=device)
        perm = make_perm(ck["config"].get("scan", "raster"))     # match training
        pw, cw2 = ck["pos_wells"], ck["color_wells2"]
        wr, wd = ck.get("trans_right"), ck.get("trans_down")     # relation wells
        M = composites(pw, cw2)
        palette = (torch.tensor(list(COLORS.values())) * 255).byte()
        for idx in args.recon:
            lab = labels[idx:idx + 1]
            T = build_traj(pw, cw2, lab, wr, wd)
            H = encode(T, ck["start"], ck["log_strength"])
            pred = decode_logits(H, M, ck["log_temp"]).argmax(-1).squeeze(0)  # (SS,)
            acc = (pred == lab.squeeze(0).long()).float().mean().item()
            left = palette[lab.squeeze(0).long().cpu()].view(S, S, 3)
            right = palette[pred.cpu()].view(S, S, 3)
            side = torch.cat([left, torch.full((S, 2, 3), 255, dtype=torch.uint8), right], dim=1)
            out = f"recon_{idx}.png"
            Image.fromarray(side.numpy()).resize(((2 * S + 2) * 4, S * 4), Image.NEAREST).save(out)
            print(f"  img {idx}: recon acc {acc:.3f} -> {out} (left=target, right=from H)")
        return
    if not args.train:
        sys.exit("need --train or --recon")

    # ---------------- level-2 model: wells + start + 2 scalars --------------
    keep_awake()
    perm = make_perm(args.scan)
    if args.pos_mode == "factored":
        # positional understanding by construction: pixel (x, y)'s well is
        # W_row[y] + W_col[x], so neighbors share a component and the geometry
        # knows about 2D adjacency. 2*S wells instead of S*S.
        def pos_init():
            if args.pos_init == "fourier":
                # 1D sinusoids: nearby indices similar, decaying with distance
                # -> METRIC adjacency from step 0 (rand gives only membership)
                i = torch.arange(S, device=device).float().unsqueeze(1)
                f = torch.exp(torch.linspace(0, torch.log(torch.tensor(S / 2.0)),
                                             args.dim // 2, device=device))
                pe = torch.cat([torch.sin(i * f * 2 * torch.pi / S),
                                torch.cos(i * f * 2 * torch.pi / S)], dim=1)
                return F.normalize(pe, dim=-1)
            return torch.randn(S, args.dim, device=device) / args.dim ** 0.5

        row_wells = torch.nn.Parameter(pos_init())
        col_wells = torch.nn.Parameter(pos_init())
        pos_desc = f"{S}+{S} row/col wells (factored)"
        pos_params = [row_wells, col_wells]

        def get_pos():
            return (row_wells.unsqueeze(1) + col_wells.unsqueeze(0)).reshape(SS, args.dim)
    else:
        pos_wells = torch.nn.Parameter(torch.randn(SS, args.dim, device=device) / args.dim ** 0.5)
        pos_desc = f"{SS} free position wells"
        pos_params = [pos_wells]

        def get_pos():
            return pos_wells

    color_wells2 = torch.nn.Parameter(torch.randn(C, args.dim, device=device) / args.dim ** 0.5)
    start = torch.nn.Parameter(torch.randn(args.dim, device=device) * 0.05)
    log_strength = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    if args.relations:
        # 2D bigram wells: "my right neighbor is color c" / "below me is c".
        # Small init so the base (position+color) physics dominates at first.
        trans_right = torch.nn.Parameter(torch.randn(C, args.dim, device=device) * 0.02)
        trans_down = torch.nn.Parameter(torch.randn(C, args.dim, device=device) * 0.02)
        rel_params, rel_desc = [trans_right, trans_down], f" + 2x{C} transition wells"
    else:
        trans_right = trans_down = None
        rel_params, rel_desc = [], ""
    params = pos_params + rel_params + [color_wells2, start, log_strength, log_temp]
    print(f"level-2 model: {pos_desc} + {C} color wells{rel_desc} x {args.dim} "
          f"+ start + strength + temp   ({sum(p.numel() for p in params):,} numbers)"
          f"   device {device}", flush=True)

    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        ck = {"pos_wells": get_pos().detach().cpu(),   # materialized either way
              "color_wells2": color_wells2.detach().cpu(),
              "start": start.detach().cpu(),
              "log_strength": log_strength.detach().cpu(),
              "log_temp": log_temp.detach().cpu(),
              "names": names,
              "config": {"size": S, "dim": args.dim, "pixel_chunk": P,
                         "pos_mode": args.pos_mode, "pos_init": args.pos_init,
                         "relations": args.relations, "scan": args.scan}}
        if args.relations:
            ck["trans_right"] = trans_right.detach().cpu()
            ck["trans_down"] = trans_down.detach().cpu()
        torch.save(ck, args.out)

    import time
    opt = torch.optim.Adam(params, lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device=device)
        lab = labels[idx]
        pos = get_pos()
        M = composites(pos, color_wells2)
        T = build_traj(pos, color_wells2, lab, trans_right, trans_down)
        H = encode(T, start, log_strength)
        logits = decode_logits(H, M, log_temp)
        loss = F.cross_entropy(logits.reshape(-1, C), lab.reshape(-1).long())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == lab.long()).float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  recon acc {acc:.3f}  "
                  f"(prior {prior_acc:.3f})  strength "
                  f"{torch.sigmoid(log_strength).item():.3f}  | {time.time() - t0:.0f}s",
                  flush=True)
        if args.save_every and step % args.save_every == 0:
            save()
    save()
    print(f"saved -> {args.out}\nrecon:  python3 vision/img_from_state_pure.py "
          f"--size {S} --recon 0 1 2")


if __name__ == "__main__":
    main()
