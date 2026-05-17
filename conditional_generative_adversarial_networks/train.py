import torch
from model import Generator, Discriminator
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

EPOCHS = 10000
BATCH_SIZE = 512
eps = 1e-7

IMG_SIZE = 32
cifar = CIFAR10('../data', download=True, transform=ToTensor())


dataloader = DataLoader(
    cifar, BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=4, pin_memory=(device == 'cuda'), persistent_workers=True, prefetch_factor=2
)

compile_mode = "default"

G = Generator(100).to(device)
D = Discriminator().to(device)

# G = torch.compile(G, mode=compile_mode)
# D = torch.compile(D, mode=compile_mode)

if os.path.exists("generator.pt"):
    G.load_state_dict(torch.load("generator.pt", map_location=device))
    print("Loaded generator.pt")
if os.path.exists("discriminator.pt"):
    D.load_state_dict(torch.load("discriminator.pt", map_location=device))
    print("Loaded discriminator.pt")

gen_optim = torch.optim.Adam(G.parameters(), lr=.0002, betas=(.5, .999))
discrim_optim = torch.optim.Adam(D.parameters(), lr=.00001, betas=(.5, .999))


d_scaler = torch.amp.GradScaler(device)
g_scaler = torch.amp.GradScaler(device)

os.makedirs('samples', exist_ok=True)

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        discrim_optim.zero_grad()
        with torch.autocast(device_type=device):
            real_preds = D(images, labels)
            fake_input = torch.randn((BATCH_SIZE, 100), device=device)
            fake_gen = G(fake_input, labels).detach()
            # print(fake_gen.shape)
            fake_preds = D(fake_gen, labels)

            d_real_loss = torch.mean(-torch.log(real_preds.clamp(eps, 1 - eps)) * 0.9)
            d_fake_loss = torch.mean(-torch.log((1 - fake_preds).clamp(eps, 1 - eps)))
            d_loss = d_real_loss + d_fake_loss

        d_scaler.scale(d_loss).backward()
        d_scaler.step(discrim_optim)
        d_scaler.update()

        gen_optim.zero_grad()
        with torch.autocast(device_type=device):
            fake_generations = G(fake_input, labels)
            fake_preds = D(fake_generations, labels)
            g_loss = torch.mean(-torch.log(fake_preds.clamp(eps, 1-eps)))

        g_scaler.scale(g_loss).backward()
        g_scaler.step(gen_optim)
        g_scaler.update()

        if i % 3 == 0:
            sample_labels = torch.arange(0, 10).to(device)
            noise = torch.randn((10, 100)).to(device)
            with torch.no_grad():
                samples = G(noise, sample_labels).cpu().float().permute(0, 2, 3, 1).numpy()

            fig, axes = plt.subplots(1, 10, figsize=(20, 2))
            for cls in range(10):
                axes[cls].imshow(samples[cls].clip(0, 1))
                axes[cls].set_title(CIFAR10_CLASSES[cls])
                axes[cls].axis('off')
            plt.tight_layout()
            plt.savefig('samples/cgan.png')
            plt.close()
            print(f"Epoch {epoch} [{i}] D_Real: {d_real_loss.item():.3f}, D_Fake: {d_fake_loss.item():.3f}, G: {g_loss.item():.3f}")

        if i % 50 == 0:
            torch.save(G.state_dict(), "generator.pt")
            torch.save(D.state_dict(), "discriminator.pt")