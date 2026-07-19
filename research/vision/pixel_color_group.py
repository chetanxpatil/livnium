"""
pixel_color_group.py — group-aware 1-pixel model: color AND color family.

pixel_color_pure answers "which of 13 colors is this pixel". This adds the
level a scene model actually wants: WHICH COLOR GROUP — so a region can be
described as "mostly warm" before (or without) naming exact colors.

Hierarchy by construction, not by an extra head:

    color well  =  W_group[group(c)] + W_offset[c]

Same-group colors share a component, so they cluster in a cone around their
group well. One collapsed h then reads out at two grains, both pure geometry:

    group:  argmax cos(h, W_group)            (+ softmax = BLEND percentages)
    color:  argmax cos(h, W_group[g] + W_offset[c])

Blends come free: a teal pixel lies between green/cyan/blue anchors, so its
group scores say "cool 0.8, neutral 0.15, ..." — exactly the mixed-membership
awareness a region summary needs.

Warm start: ch_wells / start / strength load from pixel_color_pure.pt,
offsets init to the OLD color wells and group wells to ~0 — step 1 IS the old
model; the group structure grows on top of it. The checkpoint saves the
MATERIALIZED color_wells (group+offset), so it is a drop-in replacement for
everything that reads pixel_color_pure.pt (label_pixels in img_from_state_*,
img_fovea) — those get group-clustered color wells with zero code change.

    python3 research/vision/pixel_color_group.py --train
    python3 research/vision/pixel_color_group.py --probe 255 40 40    # color + blend
    python3 research/vision/pixel_color_group.py --probe 0 128 128    # a blend case
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pixel_color_pure import COLORS, keep_awake  # noqa: E402
from vision_paths import model_path

BASE = model_path("pixel_color_pure.pt")
OUT = model_path("pixel_color_group.pt")

# color families over the known anchors — edit freely, retrain cheaply
GROUPS = {
    "neutral": ["black", "white", "gray"],
    "warm":    ["red", "orange", "yellow", "pink", "magenta", "brown"],
    "cool":    ["green", "blue", "cyan", "purple"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--group-loss", type=float, default=0.5,
                    help="weight of the group CE next to the color CE")
    ap.add_argument("--base", default=BASE, help="warm-start checkpoint")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--probe", nargs=3, type=int, default=None,
                    metavar=("R", "G", "B"), help="classify one 0-255 pixel")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = torch.device(args.device)
    torch.manual_seed(0)
    names = list(COLORS)
    gnames = list(GROUPS)
    C, G = len(names), len(gnames)
    anchors = torch.tensor(list(COLORS.values()), device=device)      # (C, 3)
    c2g = torch.tensor([gnames.index(next(g for g, cs in GROUPS.items()
                                          if n in cs)) for n in names],
                       device=device)                                  # (C,)

    def label(rgb):
        return torch.cdist(rgb, anchors).argmin(dim=1)

    def encode(ch_wells, start, log_strength, rgb):
        """Verbatim pixel_color_pure physics: 3 pulls, R -> G -> B."""
        h = start.expand(rgb.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        A = F.normalize(ch_wells, dim=-1)
        for c in range(3):
            t = A[c]
            v = rgb[:, c:c + 1]
            align = (F.normalize(h, dim=-1) * t).sum(-1, keepdim=True)
            away = F.normalize(h - t, dim=-1)
            h = h - v * s * (1.0 - align) * away
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def color_wells_of(group_wells, offset_wells):
        """color well = its group well + its own offset. The hierarchy."""
        return group_wells[c2g] + offset_wells

    # ---------------- probe: exact color + soft group blend -----------------
    if args.probe:
        ck = torch.load(args.out, map_location=device)
        rgb = torch.tensor([[c / 255.0 for c in args.probe]], device=device)
        h = F.normalize(encode(ck["ch_wells"], ck["start"],
                               ck["log_strength"], rgb), dim=-1)
        cw = F.normalize(ck["color_wells"], dim=-1)
        gw = F.normalize(ck["group_wells"], dim=-1)
        top = (h @ cw.t()).squeeze(0).topk(3)
        blend = torch.softmax((h @ gw.t()).squeeze(0)
                              / (F.softplus(ck["log_gtemp"]) + 1e-3), dim=0)
        print("color:", "  ".join(f"{ck['names'][i]}({v:.2f})"
                                  for v, i in zip(top.values, top.indices)))
        print("group:", ck["group_names"][ck["color_to_group"][top.indices[0]]],
              "(via color)   blend:", "  ".join(
                  f"{n} {blend[i]:.2f}" for i, n in enumerate(ck["group_names"])))
        return
    if not args.train:
        sys.exit("need --train or --probe")

    # ---------------- warm start over the pure model ------------------------
    keep_awake()
    base = torch.load(args.base, map_location=device)
    assert base["ch_wells"].size(1) == args.dim, "dim mismatch with base"
    ch_wells = torch.nn.Parameter(base["ch_wells"].to(device))
    start = torch.nn.Parameter(base["start"].to(device))
    log_strength = torch.nn.Parameter(base["log_strength"].to(device))
    # offsets = old color wells, groups ~0: step 1 IS pixel_color_pure
    offset_wells = torch.nn.Parameter(base["color_wells"].to(device))
    group_wells = torch.nn.Parameter(torch.randn(G, args.dim, device=device) * 0.02)
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_gtemp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    params = [ch_wells, start, log_strength, offset_wells, group_wells,
              log_temp, log_gtemp]
    print(f"group model: warm start {args.base} + {G} group wells x {args.dim}"
          f"   groups: " + "  ".join(f"{g}[{len(cs)}]" for g, cs in GROUPS.items()),
          flush=True)

    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(1, args.steps + 1):
        rgb = torch.rand(args.batch, 3, device=device)
        tgt = label(rgb)
        h = F.normalize(encode(ch_wells, start, log_strength, rgb), dim=-1)
        cw = F.normalize(color_wells_of(group_wells, offset_wells), dim=-1)
        gw = F.normalize(group_wells, dim=-1)
        temp = F.softplus(log_temp) + 1e-3
        gtemp = F.softplus(log_gtemp) + 1e-3
        loss = (F.cross_entropy((h @ cw.t()) / temp, tgt)
                + args.group_loss
                * F.cross_entropy((h @ gw.t()) / gtemp, c2g[tgt]))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                rv = torch.rand(4096, 3, device=device)
                tv = label(rv)
                hv = F.normalize(encode(ch_wells, start, log_strength, rv), dim=-1)
                cp = (hv @ cw.t()).argmax(1)
                ca = (cp == tv).float().mean().item()
                # via-color: the HARD group answer (color -> its family).
                # direct: single group well — bounded, since a family's basins
                # are a non-convex union; its real job is the soft blend
                gv = (c2g[cp] == c2g[tv]).float().mean().item()
                gd = ((hv @ gw.t()).argmax(1) == c2g[tv]).float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  color acc {ca:.3f}  "
                  f"group acc {gv:.3f} via-color ({gd:.3f} direct)  strength "
                  f"{torch.sigmoid(log_strength).item():.3f}", flush=True)

    # cone check: do same-group colors now cluster around their group well?
    with torch.no_grad():
        cw = F.normalize(color_wells_of(group_wells, offset_wells), dim=-1)
        gw = F.normalize(group_wells, dim=-1)
        sim = cw @ gw.t()                                          # (C, G)
        own = sim[torch.arange(C), c2g].mean().item()
        other = sim.sum(1).sub(sim[torch.arange(C), c2g]).div(G - 1).mean().item()
        print(f"hierarchy: color-to-OWN-group cos {own:.3f}  vs other groups "
              f"{other:.3f}   (gap = group structure is real)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"ch_wells": ch_wells.detach().cpu(),
                # materialized: drop-in for everything reading the pure ck
                "color_wells": color_wells_of(group_wells, offset_wells).detach().cpu(),
                "group_wells": group_wells.detach().cpu(),
                "offset_wells": offset_wells.detach().cpu(),
                "start": start.detach().cpu(),
                "log_strength": log_strength.detach().cpu(),
                "log_gtemp": log_gtemp.detach().cpu(),
                "names": names,
                "group_names": gnames,
                "color_to_group": c2g.cpu(),
                "groups": {g: list(cs) for g, cs in GROUPS.items()}}, args.out)
    print(f"saved -> {args.out}   (drop-in compatible with pixel_color_pure.pt "
          f"readers)\nprobe:  python3 research/vision/pixel_color_group.py --probe 0 128 128")


if __name__ == "__main__":
    main()
