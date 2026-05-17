import torch
from model import Generator, Discriminator
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

EPOCHS = 10000
BATCH_SIZE = 256
eps = 1e-7

IMG_SIZE = 32
mnist = CIFAR10('../data', download=True, transform=ToTensor())
dataloader = DataLoader(
    mnist, BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=4, pin_memory=(device == 'cuda'), persistent_workers=True, prefetch_factor=2
)

compile_mode = "reduce-overhead" if device == 'cuda' else 'default'
G = torch.compile(Generator(100).to(device), mode=compile_mode)
D = torch.compile(Discriminator().to(device), mode=compile_mode)

if os.path.exists("generator.pt"):
    G.load_state_dict(torch.load("generator.pt", map_location=device))
    print("Loaded generator.pt")
if os.path.exists("discriminator.pt"):
    D.load_state_dict(torch.load("discriminator.pt", map_location=device))
    print("Loaded discriminator.pt")

gen_optim = torch.optim.Adam(G.parameters(), lr=.0002, betas=(.5, .999))
discrim_optim = torch.optim.Adam(D.parameters(), lr=.0002, betas=(.5, .999))


d_scaler = torch.amp.GradScaler(device)
g_scaler = torch.amp.GradScaler(device)

os.makedirs('samples', exist_ok=True)

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dataloader):
        torch.compiler.cudagraph_mark_step_begin()
        images = images.to(device, non_blocking=True)
        # print(images.shape)

        discrim_optim.zero_grad()
        with torch.autocast(device_type=device):
            real_preds = D(images)
            fake_input = torch.rand((BATCH_SIZE, 100), device=device) * 2 - 1
            fake_gen = G(fake_input).detach()
            # print(fake_gen.shape)
            fake_preds = D(fake_gen)

            d_real_loss = torch.mean(-torch.log(real_preds.clamp(eps, 1 - eps)))
            d_fake_loss = torch.mean(-torch.log((1 - fake_preds).clamp(eps, 1 - eps)))
            d_loss = d_real_loss + d_fake_loss

        d_scaler.scale(d_loss).backward()
        d_scaler.step(discrim_optim)
        d_scaler.update()

        gen_optim.zero_grad()
        with torch.autocast(device_type=device):
            fake_generations = G(fake_input)
            fake_preds = D(fake_generations)
            g_loss = torch.mean(-torch.log(fake_preds.clamp(eps, 1-eps)))

        g_scaler.scale(g_loss).backward()
        g_scaler.step(gen_optim)
        g_scaler.update()

        if i % 10 == 0:
            img = fake_generations[0].cpu().permute(1, 2, 0).detach().float().numpy()
            plt.imsave(f'samples/dcgan.png', img.clip(0, 1))
            print(f"Epoch {epoch} [{i}] D_Real: {d_real_loss.item():.3f}, D_Fake: {d_fake_loss.item():.3f}, G: {g_loss.item():.3f}")

        if i % 50 == 0:
            torch.save(G.state_dict(), "generator.pt")
            torch.save(D.state_dict(), "discriminator.pt")