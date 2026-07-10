"""
img_from_state_rgb.py — train OVER the img_from_state weights, + raw RGB.

Warm start: every parameter of the trained img_from_state_pure.pt (position
wells, color wells, start, strength, temp) is loaded and keeps training.
NEW: 3 channel wells (R, G, B). Each pixel's pull target becomes

    t_i = norm( W_pos[i] + W_color[label_i] + r*CR + g*CG + b*CB )

i.e. the raw continuous RGB is concatenated into the composite — the shading
information the 13-color quantizer threw away flows back into the encoding.
Channel wells start near zero, so step 1 IS the old model; everything it
learns beyond that is what raw RGB buys.

DECODE is unchanged (13 colors at 4096 positions from H alone), so recon
accuracy is directly comparable to img_from_state_pure's (~0.56 vs prior 0.38).

    python3 vision/img_from_state_rgb.py --train --device mps
    python3 vision/img_from_state_rgb.py --recon 0 1 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_from_state_pure import build_rgb_cache, keep_awake, label_pixels  # noqa: E402

BASE = "vision/model/img_from_state_pure.pt"
OUT = "vision/model/img_from_state_rgb.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images/coco/val2017")
    ap.add_argument("--base", default=BASE, help="warm-start checkpoint")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
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

    # ---- warm start: ALL weights come from the trained level-2 model -------
    src = args.out if args.recon is not None and os.path.exists(args.out) else args.base
    ck = torch.load(src, map_location=device)
    S = ck["config"]["size"]
    SS, D, P = S * S, ck["pos_wells"].size(1), args.pixel_chunk
    pos_wells = torch.nn.Parameter(ck["pos_wells"].to(device))
    color_wells2 = torch.nn.Parameter(ck["color_wells2"].to(device))
    start = torch.nn.Parameter(ck["start"].to(device))
    log_strength = torch.nn.Parameter(ck["log_strength"].to(device))
    log_temp = torch.nn.Parameter(ck["log_temp"].to(device))
    ch_wells = torch.nn.Parameter(
        ck.get("ch_wells", torch.randn(3, D) * 0.02).to(device))  # NEW, ~zero init
    params = [pos_wells, color_wells2, start, log_strength, log_temp, ch_wells]
    print(f"warm start from {src}  (S={S}, dim={D})", flush=True)

    def encode(lab, rgb):
        """lab (B, SS) labels + rgb (B, SS, 3) raw pixels -> H (B, D)."""
        base = pos_wells[torch.arange(SS, device=device)].unsqueeze(0) \
            + color_wells2[lab.long()]                            # (B, SS, D)
        T = F.normalize(base + rgb @ ch_wells, dim=-1)            # rgb concat
        h = start.expand(lab.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        for c0 in range(0, SS, P):
            t = T[:, c0:c0 + P]
            align = (F.normalize(h, dim=-1).unsqueeze(1) * t).sum(-1)
            away = F.normalize(h.unsqueeze(1) - t, dim=-1)
            h = h - ((s * (1.0 - align)).unsqueeze(-1) * away).sum(1)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def decode_logits(H):
        M = F.normalize(pos_wells.unsqueeze(1) + color_wells2.unsqueeze(0), dim=-1)
        temp = F.softplus(log_temp) + 1e-3
        return (F.normalize(H, dim=-1) @ M.view(-1, D).t()).view(-1, SS, 13) / temp

    # ---- data ---------------------------------------------------------------
    cache = os.path.join(os.path.dirname(args.out) or ".", f"img_cache_rgb_{S}.pt")
    imgs = build_rgb_cache(args.images, S, args.max_images, cache)
    print("labeling pixels with frozen pixel_color_pure ...", flush=True)
    labels, names = label_pixels(imgs, "vision/model/pixel_color_pure.pt", device)
    N = labels.size(0)
    labels = labels.to(device)
    rgb_all = imgs.view(N, SS, 3).to(device)                      # uint8, tiny

    prior = torch.mode(labels.long().cpu(), dim=0).values.to(device)
    prior_acc = (labels.long() == prior.unsqueeze(0)).float().mean().item()
    print(f"  {N:,} images  {SS} positions  prior acc {prior_acc:.3f}", flush=True)

    def save():
        torch.save({"pos_wells": pos_wells.detach().cpu(),
                    "color_wells2": color_wells2.detach().cpu(),
                    "ch_wells": ch_wells.detach().cpu(),
                    "start": start.detach().cpu(),
                    "log_strength": log_strength.detach().cpu(),
                    "log_temp": log_temp.detach().cpu(),
                    "names": names,
                    "config": {"size": S, "dim": D, "pixel_chunk": P}}, args.out)

    if args.recon is not None:
        from PIL import Image
        from pixel_color_pure import COLORS
        palette = (torch.tensor(list(COLORS.values())) * 255).byte()
        with torch.no_grad():
            for idx in args.recon:
                lab = labels[idx:idx + 1]
                rgb = rgb_all[idx:idx + 1].float() / 255.0
                pred = decode_logits(encode(lab, rgb)).argmax(-1).squeeze(0)
                acc = (pred == lab.squeeze(0).long()).float().mean().item()
                left = palette[lab.squeeze(0).long().cpu()].view(S, S, 3)
                right = palette[pred.cpu()].view(S, S, 3)
                side = torch.cat([left, torch.full((S, 2, 3), 255, dtype=torch.uint8), right], 1)
                out = f"recon_rgb_{idx}.png"
                Image.fromarray(side.numpy()).resize(((2 * S + 2) * 4, S * 4),
                                                     Image.NEAREST).save(out)
                print(f"  img {idx}: recon acc {acc:.3f} -> {out}")
        return
    if not args.train:
        sys.exit("need --train or --recon")

    # ---- train over the warm-started weights --------------------------------
    keep_awake()
    import time
    opt = torch.optim.Adam(params, lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device=device)
        lab = labels[idx]
        rgb = rgb_all[idx].float() / 255.0
        logits = decode_logits(encode(lab, rgb))
        loss = F.cross_entropy(logits.reshape(-1, 13), lab.reshape(-1).long())
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
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
