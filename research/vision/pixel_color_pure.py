"""
pixel_color_pure.py — the overfit8 of vision: ONE pixel, pure vector collapse.

The minimal mechanism test. Input is a single RGB pixel; the trajectory is its
3 channels in fixed order (R -> G -> B), each pulling h toward a learned
CHANNEL well, gated by that channel's intensity — exactly the
img_collapse_pure.py physics with S*S=3 wells:

    h <- h - v * s * (1 - cos(h, W_ch)) * norm(h - W_ch)

Readout is pure geometry: cos(h, color wells) / temp -> CE over color names.

Learnable things, in full:
    - 3 channel wells (R, G, B)
    - one well per color name
    - a start state
    - one scalar strength, one scalar temp
    MLP: none.  Conv: none.  Readout matrix: none.

Data is synthetic and infinite: sample RGB uniform in [0,1]^3, label = nearest
canonical color anchor. A pixel either lands in the right basin or the
mechanism can't separate the RGB cube — there is nothing else to blame.

    python3 research/vision/pixel_color_pure.py            # train + report accuracy
    python3 research/vision/pixel_color_pure.py --probe 255 40 40   # classify one RGB
"""

import argparse
import os
import sys

from vision_paths import model_path

OUT = model_path("pixel_color_pure.pt")

# canonical anchors: name -> RGB in [0,1]
COLORS = {
    "black":   (0.0, 0.0, 0.0),
    "white":   (1.0, 1.0, 1.0),
    "gray":    (0.5, 0.5, 0.5),
    "red":     (1.0, 0.0, 0.0),
    "green":   (0.0, 0.8, 0.0),
    "blue":    (0.0, 0.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0),
    "cyan":    (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange":  (1.0, 0.5, 0.0),
    "purple":  (0.5, 0.0, 0.8),
    "pink":    (1.0, 0.6, 0.7),
    "brown":   (0.4, 0.25, 0.1),
}


def keep_awake():
    """macOS: stop idle sleep for the life of this process (no-op elsewhere)."""
    if sys.platform == "darwin":
        import subprocess
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--probe", nargs=3, type=int, default=None,
                    metavar=("R", "G", "B"), help="classify one 0-255 pixel")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = torch.device(args.device)
    torch.manual_seed(0)
    names = list(COLORS)
    anchors = torch.tensor(list(COLORS.values()), device=device)   # (C, 3)
    C = len(names)

    def label(rgb):  # (B, 3) -> (B,) nearest canonical anchor
        return torch.cdist(rgb, anchors).argmin(dim=1)

    def encode(ch_wells, start, log_strength, rgb):
        """(B, 3) pixel -> (B, D). One collapse step per channel, R -> G -> B."""
        h = start.expand(rgb.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        A = F.normalize(ch_wells, dim=-1)                          # (3, D)
        for c in range(3):
            t = A[c]
            v = rgb[:, c:c + 1]
            align = (F.normalize(h, dim=-1) * t).sum(-1, keepdim=True)
            away = F.normalize(h - t, dim=-1)
            h = h - v * s * (1.0 - align) * away
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    if args.probe:
        ck = torch.load(args.out, map_location=device)
        rgb = torch.tensor([[c / 255.0 for c in args.probe]], device=device)
        h = F.normalize(encode(ck["ch_wells"], ck["start"], ck["log_strength"], rgb), dim=-1)
        cw = F.normalize(ck["color_wells"], dim=-1)
        top = (h @ cw.t()).squeeze(0).topk(3)
        print("  ".join(f"{ck['names'][i]}({v:.2f})" for v, i in zip(top.values, top.indices)))
        return

    keep_awake()
    ch_wells = torch.nn.Parameter(torch.randn(3, args.dim, device=device) / args.dim ** 0.5)
    color_wells = torch.nn.Parameter(torch.randn(C, args.dim, device=device) / args.dim ** 0.5)
    start = torch.nn.Parameter(torch.randn(args.dim, device=device) * 0.05)
    log_strength = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    params = [ch_wells, color_wells, start, log_strength, log_temp]
    print(f"pure 1-pixel model: 3 channel wells + {C} color wells x {args.dim} "
          f"+ start + strength + temp   ({sum(p.numel() for p in params):,} numbers)")
    print(f"chance {1 / C:.3f}   prior-only (best constant) is barely better — "
          f"anything near 1.0 is the mechanism working")

    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(1, args.steps + 1):
        rgb = torch.rand(args.batch, 3, device=device)
        tgt = label(rgb)
        h = F.normalize(encode(ch_wells, start, log_strength, rgb), dim=-1)
        A = F.normalize(color_wells, dim=-1)
        temp = F.softplus(log_temp) + 1e-3
        loss = F.cross_entropy((h @ A.t()) / temp, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                rgb_v = torch.rand(4096, 3, device=device)
                hv = F.normalize(encode(ch_wells, start, log_strength, rgb_v), dim=-1)
                acc = ((hv @ A.t()).argmax(1) == label(rgb_v)).float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  val acc {acc:.3f}  "
                  f"strength {torch.sigmoid(log_strength).item():.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"ch_wells": ch_wells.detach().cpu(),
                "color_wells": color_wells.detach().cpu(),
                "start": start.detach().cpu(),
                "log_strength": log_strength.detach().cpu(),
                "names": names}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
