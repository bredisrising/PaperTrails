import math
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import UNet
from ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpts/latest.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--out", default="sample.png")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    return p.parse_args()


@torch.no_grad()
def sample(net, n, steps, img_size, device, solver="heun"):
    x = torch.randn(n, 3, img_size, img_size, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_now = torch.full((n,), i * dt, device=device)
        v1 = net(x, t_now)
        if solver == "euler":
            x = x + dt * v1
        else:  # Heun's method (2nd order)
            x_mid = x + dt * v1
            t_next = torch.full((n,), (i + 1) * dt, device=device)
            v2 = net(x_mid, t_next)
            x = x + 0.5 * dt * (v1 + v2)
    return (x / 2.0 + 0.5).clamp(0.0, 1.0)


def save_grid(imgs, path):
    n = imgs.shape[0]
    side = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(side, side))
    for k, ax in enumerate(axes.flat):
        if k < n:
            ax.imshow(imgs[k].cpu().permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = UNet(img_size=args.img_size).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    key = "ema" if args.use_ema and "ema" in ck else "net"
    net.load_state_dict(ck[key])
    net.eval()
    print(f"loaded {args.ckpt} ({key}, step {ck.get('step', '?')})")

    imgs = sample(net, args.n, args.steps, args.img_size, device, solver=args.solver)
    save_grid(imgs, args.out)


if __name__ == "__main__":
    main()
