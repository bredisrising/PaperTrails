import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from model import UNet
from ema import EMA


device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ------------------ config (tuned for ~half-H100: 40GB / ~66 SMs) ------------------
# Model: base_channels=192 → ~70M params. Comfortably fits with BS=512 in <30GB bf16.
IMG_SIZE = 32
BASE_CHANNELS = 192
NUM_RES_BLOCKS = 2
BATCH_SIZE = 512                # large batch to keep SMs saturated
LR = 3e-4                       # sqrt(4)·2e-4 ≈ 4e-4; back off slightly for stability
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 2000
TOTAL_STEPS = 200_000           # large batches reach quality in fewer steps
GRAD_CLIP = 1.0
EMA_DECAY = 0.9999
LOG_EVERY = 50
CKPT_EVERY = 2000
SAMPLE_EVERY = 2000
NUM_SAMPLES = 16
SAMPLE_STEPS = 50
DROPOUT = 0.1
USE_BF16 = True
USE_CHANNELS_LAST = True
USE_COMPILE = True              # H100 compile gains are large (1.5-2x)
COMPILE_MODE = "max-autotune"
NUM_WORKERS = 8
# -----------------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

dataset = CIFAR10("../data", train=True, download=True, transform=transform)
dataloader = DataLoader(
    dataset,
    BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
    persistent_workers=True,
    prefetch_factor=4,
)

raw_net = UNet(
    img_size=IMG_SIZE,
    base_channels=BASE_CHANNELS,
    num_res_blocks=NUM_RES_BLOCKS,
    dropout=DROPOUT,
).to(device)
if USE_CHANNELS_LAST:
    raw_net = raw_net.to(memory_format=torch.channels_last)

n_params = sum(p.numel() for p in raw_net.parameters())
print(f"params: {n_params/1e6:.2f}M")

# EMA wraps the *uncompiled* model so deepcopy + state_dict keys stay clean.
ema = EMA(raw_net, decay=EMA_DECAY)

if USE_COMPILE and device == "cuda":
    net = torch.compile(raw_net, mode=COMPILE_MODE)
else:
    net = raw_net

optim = torch.optim.AdamW(raw_net.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

# fp16 needs a scaler, bf16 doesn't
amp_dtype = torch.bfloat16 if (USE_BF16 and device == "cuda") else torch.float16
use_scaler = amp_dtype == torch.float16
scaler = torch.amp.GradScaler(device) if use_scaler else None

lossfn = nn.MSELoss()
os.makedirs("samples", exist_ok=True)
os.makedirs("ckpts", exist_ok=True)


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    # cosine decay to 10% of LR
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def sample_grid(model, n=16, steps=50, img_size=32):
    was_training = model.training
    model.eval()
    x = torch.randn(n, 3, img_size, img_size, device=device)
    if USE_CHANNELS_LAST:
        x = x.to(memory_format=torch.channels_last)
    for i in range(steps):
        t = torch.full((n,), i / steps, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            v = model(x, t)
        x = x + v.float() * (1.0 / steps)
    if was_training:
        model.train()
    return (x.float() / 2.0 + 0.5).clamp(0.0, 1.0)


def save_grid(imgs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = imgs.shape[0]
    side = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(side, side))
    for k, ax in enumerate(axes.flat):
        if k < n:
            ax.imshow(imgs[k].cpu().permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ------------ resume if checkpoint exists ------------
start_step = 0
ckpt_latest = "ckpts/latest.pt"
if os.path.exists(ckpt_latest):
    ck = torch.load(ckpt_latest, map_location=device)
    raw_net.load_state_dict(ck["net"])
    ema.load_state_dict(ck["ema"])
    optim.load_state_dict(ck["optim"])
    start_step = ck["step"]
    print(f"resumed from step {start_step}")
# -----------------------------------------------------

step = start_step
running_loss = 0.0
running_count = 0
t0 = time.time()
net.train()
done = False
while not done:
    for images, _ in dataloader:
        for g in optim.param_groups:
            g["lr"] = lr_at(step)

        images = images.to(device, non_blocking=True)
        if USE_CHANNELS_LAST:
            images = images.to(memory_format=torch.channels_last)

        B = images.shape[0]
        t = torch.rand(B, device=device)
        noise = torch.randn_like(images)
        x_t = (1 - t.view(B, 1, 1, 1)) * noise + t.view(B, 1, 1, 1) * images
        target = images - noise

        optim.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            pred = net(x_t, t)
            loss = lossfn(pred, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optim.step()

        ema.update(raw_net)

        running_loss += loss.item()
        running_count += 1

        if step % LOG_EVERY == 0 and running_count > 0:
            avg = running_loss / running_count
            dt = time.time() - t0
            ips = LOG_EVERY * BATCH_SIZE / max(dt, 1e-6) if step > start_step else 0
            print(f"step {step:>7d} | loss {avg:.4f} | lr {lr_at(step):.2e} | {ips:.0f} imgs/s")
            running_loss = 0.0
            running_count = 0
            t0 = time.time()

        if step > 0 and step % CKPT_EVERY == 0:
            torch.save(
                {
                    "net": raw_net.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                },
                ckpt_latest,
            )

        if step > 0 and step % SAMPLE_EVERY == 0:
            grid = sample_grid(ema.model, n=NUM_SAMPLES, steps=SAMPLE_STEPS, img_size=IMG_SIZE)
            save_grid(grid, f"samples/step_{step:07d}.png")

        step += 1
        if step >= TOTAL_STEPS:
            done = True
            break

# final save
torch.save(
    {
        "net": net.state_dict(),
        "ema": ema.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
    },
    ckpt_latest,
)
print("done.")
