import torch
from model import UNet
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

EPOCHS = 10000
BATCH_SIZE = 512
eps = 1e-7


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
])

IMG_SIZE = 32
dataset = SVHN('../data', split='train', download=True, transform=transform)


dataloader = DataLoader(
    dataset, BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=4, pin_memory=(device == 'cuda'), persistent_workers=True, prefetch_factor=2
)

compile_mode = 'default'

net = UNet(img_size=IMG_SIZE).to(device)

if os.path.exists("ckpt.pt"):
    net.load_state_dict(torch.load("ckpt.pt", map_location=device))
    print("Generator Loaded")

from muon import SingleDeviceMuon

muon_params = [p for n, p in net.named_parameters() if p.ndim >= 2 and 'end' not in n]
aux_params  = [p for n, p in net.named_parameters() if p.ndim < 2 or 'end' in n]

muon_optim  = SingleDeviceMuon(muon_params, lr=0.02, momentum=0.95)
aux_optim   = torch.optim.AdamW(aux_params, lr=3e-4)

scaler = torch.amp.GradScaler(device)
lossfn = nn.MSELoss()

os.makedirs('samples', exist_ok=True)

def interpolate(noise, target, time):
    interpolated = (target - noise) * time + noise
    return interpolated 

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        muon_optim.zero_grad(set_to_none=True)
        aux_optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device):
            time = torch.rand((BATCH_SIZE), device=device).view(-1, 1, 1, 1)
            noise = torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), device=device)
            target = images - noise
            interpolated = target * time + noise
            preds = net(interpolated, time)
            loss = lossfn(preds, target)

        scaler.scale(loss).backward()
        scaler.unscale_(muon_optim)
        scaler.unscale_(aux_optim)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(muon_optim)
        scaler.step(aux_optim)
        scaler.update()

        if i % 5 == 0:
            torch.save(net.state_dict(), "ckpt.pt")


        print(f"Epoch {epoch} [{i}] Loss: {loss.item():.3f}")