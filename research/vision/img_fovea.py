"""
img_fovea.py — foveal level 2: the image as seen from an eye, not a raster.

Position is polar, relative to the focal center (31.5, 31.5 for S=64):

    radius -> ring well     angle -> direction well     ray = norm([dx,dy,focal])

    pixel address  =  W_ring[r_bin] * W_angle[a_bin] * sqrt(D)  +  ray @ W_ray

"brown, slightly below center, at this distance" instead of "pixel 2087 is
brown". ~R+A+3 wells replace S*S; pixels close in radius/angle share wells, so
2D adjacency exists by construction. Collapse physics and pure-geometry decode
are unchanged from img_from_state_pure — recon accuracy is directly comparable.

Lessons burned in (2026-07-07, see README):
  * bind MULTIPLICATIVELY: additive ring+angle addresses are so correlated the
    state collapses to one flat color per image
  * init with SMOOTHED RANDOM wells (gaussian blur along the index): local
    adjacency, orthogonal beyond ~2*sigma bins
  * CE training re-correlates addresses over time -> --addr-reg holds them
    orthogonal; --inspect measures drift on any checkpoint
  * recon acc flatters on dominant-color images -> watch uniq-colors

    python3 vision/img_fovea.py --selftest                # 1-min mechanics gate
    python3 vision/img_fovea.py --train --addr-reg 0.1 --device mps
    python3 vision/img_fovea.py --recon 0 1 2             # side-by-side PNGs
    python3 vision/img_fovea.py --inspect                 # address drift check

Level 1 is pluggable: --l1 vision/model/pixel_color_group.pt labels pixels
with the group-aware model (drop-in keys) and recon then also reports color-
GROUP shares (neutral/warm/cool) for the whole image and fovea vs periphery.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_from_state_pure import build_rgb_cache, keep_awake, label_pixels  # noqa: E402

OUT = "vision/model/img_fovea.pt"
L1 = "vision/model/pixel_color_pure.pt"


def polar_geometry(S, rings, angles, focal, device):
    """Per-pixel foveal coordinates in raster order: r_bin, a_bin (long),
    rays (SS, 3) unit view directions, r_norm (SS,) in [0, 1]."""
    import torch
    c = (S - 1) / 2.0
    y, x = torch.meshgrid(torch.arange(S, device=device).float(),
                          torch.arange(S, device=device).float(), indexing="ij")
    dx, dy = (x - c).reshape(-1), (y - c).reshape(-1)
    radius = torch.sqrt(dx * dx + dy * dy)
    r_norm = radius / radius.max()
    r_bin = (r_norm * rings).long().clamp(max=rings - 1)
    a_bin = ((torch.atan2(dy, dx) + math.pi) / (2 * math.pi)
             * angles).long().clamp(max=angles - 1)
    rays = torch.stack([dx, dy, torch.full_like(dx, focal)], dim=-1)
    return r_bin, a_bin, rays / rays.norm(dim=-1, keepdim=True), r_norm


def smooth_wells(n, dim, device, sigma=1.5, circular=False):
    """Random unit wells gaussian-blurred along the index: neighbor cos ~0.9,
    decorrelated beyond ~2*sigma bins. What lets mul binding keep adjacency
    AND addressing capacity."""
    import torch
    import torch.nn.functional as F
    i = torch.arange(n, device=device).float()
    d = (i.unsqueeze(0) - i.unsqueeze(1)).abs()
    if circular:
        d = torch.minimum(d, n - d)
    K = torch.exp(-0.5 * (d / sigma) ** 2)
    return F.normalize((K / K.sum(1, keepdim=True))
                       @ torch.randn(n, dim, device=device), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images/coco/val2017")
    ap.add_argument("--l1", default=L1,
                    help="level-1 pixel labeler (pixel_color_pure.pt or the "
                         "drop-in pixel_color_group.pt)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--rings", type=int, default=32)
    ap.add_argument("--angles", type=int, default=64)
    ap.add_argument("--focal", type=float, default=None,
                    help="ray focal distance (default S/2 ~ 90deg FOV)")
    ap.add_argument("--bind", choices=["mul", "add"], default="mul",
                    help="mul: elementwise ring*angle addresses (add is the "
                         "documented flat-color failure, kept for ablation)")
    ap.add_argument("--pos-init", choices=["smooth", "rand"], default="smooth")
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="smooth init blur radius in bins")
    ap.add_argument("--scan", choices=["spiral", "raster", "shuffled"],
                    default="spiral", help="spiral: fovea collapses first")
    ap.add_argument("--rgb", action="store_true",
                    help="raw RGB channel wells in the encode composite")
    ap.add_argument("--addr-reg", type=float, default=0.0,
                    help="cos^2 penalty holding addresses orthogonal (try 0.1)")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pixel-chunk", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--recon", nargs="*", type=int, default=None)
    ap.add_argument("--inspect", action="store_true",
                    help="print a checkpoint's address stats and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="no images needed: store 16 synthetic maps, gate on "
                         "structured recall >> prior. FAIL = code broken; "
                         "PASS = full-scale issues are capacity, not bugs")
    ap.add_argument("--energy-grad", action="store_true",
                    help="use analytical energy gradient descent instead of hand-designed away force")
    ap.add_argument("--h-reg", type=float, default=0.0,
                    help="cos^2 penalty holding batch states orthogonal to prevent state collapse")
    ap.add_argument("--outside-in", action="store_true",
                    help="reverse spiral scan to outside-in (periphery first, fovea last)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(0)

    if args.selftest:
        args.size, args.dim, args.rings, args.angles = 16, 128, 16, 32
        args.steps, args.batch, args.train, args.sigma = 500, 8, True, 1.0
        args.pixel_chunk, args.rgb, args.recon = 16, False, None
        args.out = os.path.join(os.path.dirname(args.out) or ".",
                                "img_fovea_selftest.pt")

    # recon/inspect must use the CHECKPOINT's geometry, not CLI defaults
    ck = None
    if args.recon is not None or args.inspect:
        ck = torch.load(args.out, map_location=device)
        cfg = ck["config"]
        args.size, args.dim, args.rings, args.angles = (
            cfg["size"], cfg["dim"], cfg["rings"], cfg["angles"])
        args.focal, args.scan = cfg["focal"], cfg["scan"]
        args.bind, args.pixel_chunk = cfg.get("bind", "add"), cfg["pixel_chunk"]
        args.rgb = cfg.get("rgb", False)
        args.l1 = cfg.get("l1", args.l1)
        args.energy_grad = cfg.get("energy_grad", False)
        args.h_reg = cfg.get("h_reg", 0.0)
        args.outside_in = cfg.get("outside_in", False)

    S, D, P = args.size, args.dim, args.pixel_chunk
    SS = S * S
    focal = args.focal if args.focal is not None else S / 2.0
    R, A = args.rings, args.angles

    r_bin, a_bin, rays, r_norm = polar_geometry(S, R, A, focal, device)
    if args.scan == "spiral":
        perm = torch.argsort(r_norm * (2 * A) + a_bin.float() / A, descending=args.outside_in)
    elif args.scan == "shuffled":
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(SS, generator=g).to(device)
    else:
        perm = None

    def get_pos(ring_w, angle_w, ray_w, bind):
        """(SS, D) pixel address wells from ~R+A+3 polar wells."""
        if bind == "mul":
            return ring_w[r_bin] * angle_w[a_bin] * D ** 0.5 + rays @ ray_w
        return ring_w[r_bin] + angle_w[a_bin] + rays @ ray_w

    def composites(pos, color_w):
        return F.normalize(pos.unsqueeze(1) + color_w.unsqueeze(0), dim=-1)

    def build_traj(pos, cw, lab, ch_w=None, rgb=None):
        T = pos.unsqueeze(0) + cw[lab.long()]
        if ch_w is not None:
            T = T + rgb @ ch_w
        T = F.normalize(T, dim=-1)
        return T if perm is None else T[:, perm]

    def encode(T, start, log_strength, log_falloff, energy_grad=False):
        h = start.expand(T.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        g = torch.exp(-F.softplus(log_falloff) * r_norm)   # foveal gain
        g = g if perm is None else g[perm]
        for c0 in range(0, SS, P):
            t = T[:, c0:c0 + P]
            h_norm = h.norm(dim=-1, keepdim=True).unsqueeze(1)
            h_n = h.unsqueeze(1) / (h_norm + 1e-8)
            align = (h_n * t).sum(-1, keepdim=True)
            
            if energy_grad:
                # Analytical energy gradient: grad = -(t - h_n * align) / ||h||
                grad = -(t - h_n * align) / (h_norm + 1e-8)
                h = h - (s * g[c0:c0 + P].unsqueeze(-1) * grad).sum(1)
            else:
                away = F.normalize(h.unsqueeze(1) - t, dim=-1)
                h = h - ((s * g[c0:c0 + P] * (1.0 - align.squeeze(-1))).unsqueeze(-1) * away).sum(1)
                
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def decode_logits(H, M, log_temp):
        temp = F.softplus(log_temp) + 1e-3
        C_ = M.size(1)
        return (F.normalize(H, dim=-1) @ M.view(SS * C_, D).t()).view(-1, SS, C_) / temp

    def pair_cos(pos, k=4000):
        pn = F.normalize(pos, dim=-1)
        i = torch.randint(0, SS, (k,), device=device)
        j = torch.randint(0, SS, (k,), device=device)
        xt = (pn[i] * pn[j]).sum(-1).abs().mean().item()
        nb = (pn[:-1] * pn[1:]).sum(-1).mean().item()
        return xt, nb

    if args.inspect:
        xt, nb = pair_cos(get_pos(ck["ring_wells"].to(device),
                                  ck["angle_wells"].to(device),
                                  ck["ray_wells"].to(device), args.bind))
        print(f"{args.out}  bind={args.bind}  mean|cos| random addr pairs "
              f"{xt:.3f} (free-well ref {1 / D ** 0.5:.3f})   "
              f"raster-neighbor cos {nb:.3f}")
        return

    # ---- data: level 1 labels every pixel ------------------------------------
    if args.selftest:
        # 8 structured (random-color quadrants + speckle) + 8 uniform-random
        g = torch.Generator().manual_seed(1)
        N, C = 16, 13
        names = [f"c{i}" for i in range(C)]
        labels = torch.randint(0, C, (N, SS), generator=g)
        for n in range(8):
            q = torch.randint(0, C, (4,), generator=g)
            m = labels[n].view(S, S)
            m[:S // 2, :S // 2], m[:S // 2, S // 2:] = q[0], q[1]
            m[S // 2:, :S // 2], m[S // 2:, S // 2:] = q[2], q[3]
            spk = torch.rand(S, S, generator=g) < 0.1
            m[spk] = torch.randint(0, C, (int(spk.sum()),), generator=g)
        labels, imgs, rgb_all = labels.to(device), None, None
        print(f"SELFTEST: {N} synthetic maps  S={S} dim={D} rings={R} "
              f"angles={A} bind={args.bind}", flush=True)
    else:
        cache = os.path.join(os.path.dirname(args.out) or ".",
                             f"img_cache_rgb_{S}.pt")
        imgs = build_rgb_cache(args.images, S, args.max_images, cache)
        print(f"labeling pixels with frozen {os.path.basename(args.l1)} ...",
              flush=True)
        labels, names = label_pixels(imgs, args.l1, device)
        N, C = labels.size(0), len(names)
        labels = labels.to(device)
        rgb_all = imgs.view(N, SS, 3).to(device) if args.rgb else None

    prior = torch.mode(labels.long().cpu(), dim=0).values.to(device)
    prior_acc = (labels.long() == prior.unsqueeze(0)).float().mean().item()
    print(f"  {N:,} images  {SS} positions  {C} colors   positional-prior acc "
          f"{prior_acc:.3f}  <- beat this or H holds nothing", flush=True)

    # ---- reconstruct from a checkpoint ---------------------------------------
    if args.recon is not None:
        from PIL import Image
        from pixel_color_pure import COLORS
        pos = get_pos(ck["ring_wells"].to(device), ck["angle_wells"].to(device),
                      ck["ray_wells"].to(device), args.bind)
        cw = ck["color_wells2"].to(device)
        M = composites(pos, cw)
        chw = ck.get("ch_wells")
        chw = chw.to(device) if chw is not None else None
        palette = (torch.tensor(list(COLORS.values())) * 255).byte()
        l1ck = torch.load(args.l1, map_location="cpu")
        c2g = l1ck.get("color_to_group")           # group-aware level 1?
        gnames = l1ck.get("group_names", [])

        def group_shares(lab_flat):
            gs = torch.bincount(c2g[lab_flat.long().cpu()],
                                minlength=len(gnames)).float() / lab_flat.numel()
            return "  ".join(f"{n} {gs[i]:.2f}" for i, n in enumerate(gnames))

        with torch.no_grad():
            for idx in args.recon:
                lab = labels[idx:idx + 1]
                rgb = (imgs.view(N, SS, 3)[idx:idx + 1].float().to(device) / 255.0
                       if chw is not None else None)
                H = encode(build_traj(pos, cw, lab, chw, rgb), ck["start"].to(device),
                           ck["log_strength"].to(device), ck["log_falloff"].to(device),
                           energy_grad=args.energy_grad)
                pred = decode_logits(H, M, ck["log_temp"].to(device)).argmax(-1).squeeze(0)
                acc = (pred == lab.squeeze(0).long()).float().mean().item()
                uniq = pred.unique().numel()
                left = palette[lab.squeeze(0).long().cpu()].view(S, S, 3)
                right = palette[pred.cpu()].view(S, S, 3)
                side = torch.cat([left, torch.full((S, 2, 3), 255,
                                                   dtype=torch.uint8), right], 1)
                out = f"recon_fovea_{idx}.png"
                Image.fromarray(side.numpy()).resize(((2 * S + 2) * 4, S * 4),
                                                     Image.NEAREST).save(out)
                print(f"  img {idx}: recon acc {acc:.3f}  uniq-colors {uniq}/{C}"
                      f"  -> {out}")
                if c2g is not None:
                    fov = r_norm < 0.5                     # fovea vs periphery
                    print(f"      groups true  | {group_shares(lab.squeeze(0))}")
                    print(f"      groups pred  | {group_shares(pred)}")
                    print(f"      pred fovea   | {group_shares(pred[fov])}")
                    print(f"      pred periph  | {group_shares(pred[~fov])}")
        return
    if not args.train:
        sys.exit("need --train, --recon, --inspect, or --selftest")

    # ---- model ----------------------------------------------------------------
    keep_awake()
    if args.pos_init == "smooth":
        ring_wells = torch.nn.Parameter(smooth_wells(R, D, device, args.sigma))
        angle_wells = torch.nn.Parameter(
            smooth_wells(A, D, device, args.sigma, circular=True))
    else:
        ring_wells = torch.nn.Parameter(torch.randn(R, D, device=device) / D ** 0.5)
        angle_wells = torch.nn.Parameter(torch.randn(A, D, device=device) / D ** 0.5)
    ray_wells = torch.nn.Parameter(torch.randn(3, D, device=device) * 0.02)
    color_wells2 = torch.nn.Parameter(torch.randn(C, D, device=device) / D ** 0.5)
    ch_wells = (torch.nn.Parameter(torch.randn(3, D, device=device) * 0.02)
                if args.rgb else None)
    start = torch.nn.Parameter(torch.randn(D, device=device) * 0.05)
    log_strength = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_falloff = torch.nn.Parameter(torch.tensor(-4.0, device=device))
    params = [ring_wells, angle_wells, ray_wells, color_wells2,
              start, log_strength, log_temp, log_falloff]
    if ch_wells is not None:
        params.append(ch_wells)
    print(f"fovea model ({args.bind} binding, {args.pos_init} init): {R} ring "
          f"+ {A} angle + 3 ray{' + 3 rgb' if args.rgb else ''} + {C} color "
          f"wells x {D} + 4 scalars  ({sum(p.numel() for p in params):,} numbers)"
          f"   scan {args.scan}  addr-reg {args.addr_reg}  device {device}",
          flush=True)

    def addr_stats(tag):
        with torch.no_grad():
            xt, nb = pair_cos(get_pos(ring_wells, angle_wells, ray_wells, args.bind))
        print(f"  [{tag}] addresses: mean|cos| random pairs {xt:.3f} "
              f"(free-well ref {1 / D ** 0.5:.3f})   neighbor cos {nb:.3f}",
              flush=True)
    addr_stats("init")

    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out = {"ring_wells": ring_wells.detach().cpu(),
               "angle_wells": angle_wells.detach().cpu(),
               "ray_wells": ray_wells.detach().cpu(),
               "color_wells2": color_wells2.detach().cpu(),
               "start": start.detach().cpu(),
               "log_strength": log_strength.detach().cpu(),
               "log_temp": log_temp.detach().cpu(),
               "log_falloff": log_falloff.detach().cpu(),
               "names": names,
               "config": {"size": S, "dim": D, "pixel_chunk": P, "rings": R,
                          "angles": A, "focal": focal, "scan": args.scan,
                          "pos_init": args.pos_init, "rgb": args.rgb,
                          "bind": args.bind, "sigma": args.sigma,
                          "l1": args.l1, "energy_grad": args.energy_grad,
                          "h_reg": args.h_reg, "outside_in": args.outside_in}}
        if ch_wells is not None:
            out["ch_wells"] = ch_wells.detach().cpu()
        torch.save(out, args.out)

    import time
    opt = torch.optim.Adam(params, lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randperm(N, device=device)[:args.batch]   # no dupes in batch
        lab = labels[idx]
        rgb = rgb_all[idx].float() / 255.0 if args.rgb else None
        pos = get_pos(ring_wells, angle_wells, ray_wells, args.bind)
        M = composites(pos, color_wells2)
        H = encode(build_traj(pos, color_wells2, lab, ch_wells, rgb),
                   start, log_strength, log_falloff, energy_grad=args.energy_grad)
        logits = decode_logits(H, M, log_temp)
        loss = F.cross_entropy(logits.reshape(-1, C), lab.reshape(-1).long())
        if args.h_reg > 0 and H.size(0) > 1:
            H_norm = F.normalize(H, dim=-1)
            h_cos = torch.matmul(H_norm, H_norm.t())
            h_reg_loss = (h_cos - torch.eye(H.size(0), device=device)).pow(2).mean()
            loss = loss + args.h_reg * h_reg_loss
        if args.addr_reg > 0:      # hold addresses orthogonal (CE drifts them)
            pn = F.normalize(pos, dim=-1)
            ii = torch.randint(0, SS, (512,), device=device)
            jj = torch.randint(0, SS, (512,), device=device)
            loss = loss + args.addr_reg * (pn[ii] * pn[jj]).sum(-1).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                pred = logits.argmax(-1)
                acc = (pred == lab.long()).float().mean().item()
                uniq = pred[0].unique().numel()   # 1 = flat-color failure, live
                hn = F.normalize(H, dim=-1)
                hx = (hn @ hn.t()).masked_fill(
                    torch.eye(hn.size(0), device=device).bool(), 0).abs().max().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  recon acc {acc:.3f}  "
                  f"(prior {prior_acc:.3f})  uniq-colors {uniq:2d}/{C}  "
                  f"H-crosscos {hx:.2f}  strength "
                  f"{torch.sigmoid(log_strength).item():.3f}  falloff "
                  f"{F.softplus(log_falloff).item():.3f}  | {time.time() - t0:.0f}s",
                  flush=True)
        if args.save_every and step % args.save_every == 0:
            save()
            addr_stats(f"step {step}")
    save()
    addr_stats("final")

    if args.selftest:
        with torch.no_grad():
            pos = get_pos(ring_wells, angle_wells, ray_wells, args.bind)
            M = composites(pos, color_wells2)
            accs = []
            for n in range(N):
                H = encode(build_traj(pos, color_wells2, labels[n:n + 1]),
                           start, log_strength, log_falloff, energy_grad=args.energy_grad)
                pred = decode_logits(H, M, log_temp).argmax(-1).squeeze(0)
                accs.append((pred == labels[n].long()).float().mean().item())
                print(f"  map {n:2d} ({'structured' if n < 8 else 'random    '})"
                      f"  acc {accs[-1]:.3f}  pred-colors "
                      f"{pred.unique().numel():2d}/{C}", flush=True)
        st, rd = sum(accs[:8]) / 8, sum(accs[8:]) / 8
        ok = st > prior_acc + 0.15 and min(accs[:8]) > prior_acc
        print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}: structured {st:.3f} vs "
              f"prior {prior_acc:.3f}  |  random {rd:.3f} (capacity probe: "
              f"{SS} symbols in {D} dims, not gated)", flush=True)
        sys.exit(0 if ok else 1)
    print(f"saved -> {args.out}\nrecon:  python3 vision/img_fovea.py "
          f"--out {args.out} --recon 0 1 2")


if __name__ == "__main__":
    main()
