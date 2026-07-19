"""
img_refine_loop.py — the TRUE loop: stage 1 frozen, stage 2 refines its output.

Stage 1 (FROZEN, img_from_state_pure.pt): encodes each image's color map into
H1. Computed once, never trained — its contribution stays measurable forever.

Stage 2 (trained here): a NEW model whose collapse STARTS AT H1 and then runs
through the raw RGB pixel trajectory with its own wells:

    t_i = norm( W2_pos[i] + r*CR + g*CG + b*CB )      (no labels — raw pixels)
    H2  = collapse(H1 through t_0 .. t_4095)

So stage 1's output is stage 2's input, concatenated with the same RGB pixels
— and with --loops k the refinement iterates: H <- refine(H, pixels) k times,
same wells each pass. Decode is 13 colors at 4096 positions from H alone,
same as stage 1, so accuracies compare directly.

Stage 2 starts as the IDENTITY on H1 (its strength initializes near zero —
the zero-init lesson from COLLAPSE_ENGINE_VERDICT.md), so step 1 scores what
frozen stage 1 scores, and every point gained is what the loop bought.

    python3 research/vision/img_refine_loop.py --train --device mps
    python3 research/vision/img_refine_loop.py --train --device mps --loops 2
    python3 research/vision/img_refine_loop.py --recon 0 1 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_from_state_pure import build_rgb_cache, keep_awake, label_pixels  # noqa: E402
from vision_paths import COCO_IMAGES, model_path

BASE = model_path("img_from_state_pure.pt")
OUT = model_path("img_refine_loop.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=COCO_IMAGES)
    ap.add_argument("--base", default=BASE, help="frozen stage-1 checkpoint")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loops", type=int, default=1, help="refinement passes")
    ap.add_argument("--err-weight", type=float, default=4.0,
                    help="loss weight on pixels stage 1 got WRONG (1 = off)")
    ap.add_argument("--pixel-chunk", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--recon", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(0)

    # ---------------- stage 1: FROZEN ---------------------------------------
    ck1 = torch.load(args.base, map_location=device)
    S = ck1["config"]["size"]
    SS, D, P = S * S, ck1["pos_wells"].size(1), args.pixel_chunk
    pw1, cw1 = ck1["pos_wells"], ck1["color_wells2"]
    M1 = F.normalize(pw1.unsqueeze(1) + cw1.unsqueeze(0), dim=-1)      # (SS,13,D)
    s1 = torch.sigmoid(ck1["log_strength"])
    temp1 = F.softplus(ck1["log_temp"]) + 1e-3

    def encode_stage1(lab):
        """Frozen: labels (B, SS) -> H1 (B, D)."""
        T = M1[torch.arange(SS, device=device), lab.long()]
        h = ck1["start"].expand(lab.size(0), -1).contiguous()
        for c0 in range(0, SS, P):
            t = T[:, c0:c0 + P]
            align = (F.normalize(h, dim=-1).unsqueeze(1) * t).sum(-1)
            away = F.normalize(h.unsqueeze(1) - t, dim=-1)
            h = h - ((s1 * (1.0 - align)).unsqueeze(-1) * away).sum(1)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    # ---------------- stage 2: the loop (trained) ---------------------------
    pos2 = torch.nn.Parameter(pw1.clone().to(device))
    col2 = torch.nn.Parameter(cw1.clone().to(device))
    # ch2 init by least squares: anchor_rgb @ ch2 ~ color_well, so the raw-RGB
    # pull starts as a CONTINUOUS version of the label trajectory stage 1
    # already uses. Random init deadlocks: noise trajectory -> gradient turns
    # the loop off -> ch2 never learns (observed: log_s2 driven -9 -> -9.95).
    from pixel_color_pure import COLORS
    _anchors = torch.tensor(list(COLORS.values()))                      # (13, 3)
    # lstsq on CPU: aten::linalg_lstsq not implemented on MPS
    ch2 = torch.nn.Parameter(
        torch.linalg.lstsq(_anchors, cw1.cpu()).solution.to(device))
    # ~identity init: the pull is applied SS=4096 times, so per-pixel strength
    # must be ~1e-4 for the total first-pass displacement to stay small.
    log_s2 = torch.nn.Parameter(torch.tensor(-9.0, device=device))
    log_t2 = torch.nn.Parameter(ck1["log_temp"].clone().to(device))
    params = [pos2, col2, ch2, log_s2, log_t2]

    def refine(h, rgb, gate):
        """One loop pass: H + raw pixels -> refined H.

        gate (B, SS) in [0,1]: only gated pixels pull. Trained with stage-1's
        error mask, so the loop works exclusively where stage 1 failed — the
        one place where the raw pixel carries information H1 provably lacks.
        """
        T = F.normalize(pos2.unsqueeze(0) + rgb @ ch2, dim=-1)          # (B,SS,D)
        s = torch.sigmoid(log_s2)
        for c0 in range(0, SS, P):
            t = T[:, c0:c0 + P]
            g = gate[:, c0:c0 + P]
            align = (F.normalize(h, dim=-1).unsqueeze(1) * t).sum(-1)
            away = F.normalize(h.unsqueeze(1) - t, dim=-1)
            h = h - ((s * g * (1.0 - align)).unsqueeze(-1) * away).sum(1)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def decode2(H):
        M2 = F.normalize(pos2.unsqueeze(1) + col2.unsqueeze(0), dim=-1)
        temp = F.softplus(log_t2) + 1e-3
        return (F.normalize(H, dim=-1) @ M2.view(-1, D).t()).view(-1, SS, 13) / temp

    # ---------------- data + frozen H1 for every image ----------------------
    cache = os.path.join(os.path.dirname(args.out) or ".", f"img_cache_rgb_{S}.pt")
    imgs = build_rgb_cache(args.images, S, args.max_images, cache)
    print("labeling pixels with frozen pixel_color_pure ...", flush=True)
    labels, names = label_pixels(imgs, model_path("pixel_color_pure.pt"), device)
    N = labels.size(0)
    labels = labels.to(device)
    rgb_all = imgs.view(N, SS, 3).to(device)

    print("encoding all images with frozen stage 1 ...", flush=True)
    with torch.no_grad():
        H1 = torch.cat([encode_stage1(labels[b:b + 64]) for b in range(0, N, 64)])
        # stage 1's own predictions -> the error mask that PUNISHES stage 2
        pred1 = ((F.normalize(H1, dim=-1) @ M1.view(-1, D).t())
                 .view(N, SS, 13).argmax(-1))                          # (N, SS)
        err1 = (pred1 != labels.long())                                # (N, SS)
        ref = (~err1).float().mean().item()
    prior = torch.mode(labels.long().cpu(), dim=0).values.to(device)
    prior_acc = (labels.long() == prior.unsqueeze(0)).float().mean().item()
    print(f"  {N:,} images  prior {prior_acc:.3f}  FROZEN stage-1 recon {ref:.3f}  "
          f"({err1.float().mean().item():.1%} of pixels wrong -> stage 2's job)",
          flush=True)

    def save():
        torch.save({"pos2": pos2.detach().cpu(), "col2": col2.detach().cpu(),
                    "ch2": ch2.detach().cpu(), "log_s2": log_s2.detach().cpu(),
                    "log_t2": log_t2.detach().cpu(), "names": names,
                    "config": {"size": S, "dim": D, "pixel_chunk": P,
                               "loops": args.loops, "err_weight": args.err_weight,
                               "base": args.base}}, args.out)

    if args.recon is not None:
        from PIL import Image
        from pixel_color_pure import COLORS
        ck2 = torch.load(args.out, map_location=device)
        with torch.no_grad():
            pos2.copy_(ck2["pos2"]); col2.copy_(ck2["col2"]); ch2.copy_(ck2["ch2"])
            log_s2.copy_(ck2["log_s2"]); log_t2.copy_(ck2["log_t2"])
            palette = (torch.tensor(list(COLORS.values())) * 255).byte()
            for idx in args.recon:
                h = H1[idx:idx + 1]
                rgb = rgb_all[idx:idx + 1].float() / 255.0
                gate = err1[idx:idx + 1].float()
                for _ in range(ck2["config"]["loops"]):
                    h = refine(h, rgb, gate)
                pred = decode2(h).argmax(-1).squeeze(0)
                lab = labels[idx].long()
                acc = (pred == lab).float().mean().item()
                left = palette[lab.cpu()].view(S, S, 3)
                right = palette[pred.cpu()].view(S, S, 3)
                side = torch.cat([left, torch.full((S, 2, 3), 255, dtype=torch.uint8),
                                  right], 1)
                out = f"recon_loop_{idx}.png"
                Image.fromarray(side.numpy()).resize(((2 * S + 2) * 4, S * 4),
                                                     Image.NEAREST).save(out)
                print(f"  img {idx}: loop recon acc {acc:.3f}  (frozen stage-1 ref "
                      f"{ref:.3f}) -> {out}")
        return
    if not args.train:
        sys.exit("need --train or --recon")

    # ---------------- train the loop ----------------------------------------
    keep_awake()
    import time
    # log_s2 starts at -9 (identity); at the shared lr it would take thousands
    # of steps to escape zero and the model just polishes the decoder instead.
    # Give the strength its own fast lr so the loop physics can switch on.
    opt = torch.optim.Adam([
        {"params": [pos2, col2, ch2, log_t2], "lr": args.lr},
        {"params": [log_s2], "lr": 0.05},
    ])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device=device)
        lab = labels[idx]
        rgb = rgb_all[idx].float() / 255.0
        err = err1[idx]                               # where stage 1 failed
        h = H1[idx]                                   # stage-1 output = input
        for _ in range(args.loops):
            h = refine(h, rgb, err.float())
        logits = decode2(h)
        # PUNISH stage 2 hardest on the pixels stage 1 got wrong
        ce = F.cross_entropy(logits.reshape(-1, 13), lab.reshape(-1).long(),
                             reduction="none")
        w = 1.0 + (args.err_weight - 1.0) * err.reshape(-1).float()
        loss = (ce * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                hit = logits.argmax(-1) == lab.long()
                fix = hit[err].float().mean().item()          # stage-1-wrong, now right
                keep = hit[~err].float().mean().item()        # stage-1-right, kept right
            print(f"step {step:5d}  loss {loss.item():.4f}  fix {fix:.3f}  "
                  f"keep {keep:.3f}  net {hit.float().mean().item():.3f}  "
                  f"(stage-1 {ref:.3f})  s2 "
                  f"{torch.sigmoid(log_s2).item():.4f}  | {time.time() - t0:.0f}s",
                  flush=True)
        if args.save_every and step % args.save_every == 0:
            save()
    save()
    print(f"saved -> {args.out}\nrecon:  python3 research/vision/img_refine_loop.py --recon 0 1 2")


if __name__ == "__main__":
    main()
