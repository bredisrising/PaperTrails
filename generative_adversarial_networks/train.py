import torch
from model import FullyConnectedGenerator, Discriminator
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

EPOCHS = 100
BATCH_SIZE = 512
eps = 1e-7

IMG_SIZE = 28
mnist = MNIST("../data", download=True, transform=ToTensor())
dataloader = DataLoader(
    mnist, BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=4, pin_memory=(device == "cuda"), persistent_workers=True, prefetch_factor=2
)

compile_mode = "reduce-overhead" if device == "cuda" else "default"
G = torch.compile(FullyConnectedGenerator(100, 2048, IMG_SIZE*IMG_SIZE).to(device), mode=compile_mode)
D = torch.compile(Discriminator(IMG_SIZE*IMG_SIZE, 512).to(device), mode=compile_mode)

generator_optimizer = torch.optim.SGD(G.parameters(), lr=.025, momentum=.5, foreach=True)
discriminator_optimizer = torch.optim.SGD(D.parameters(), lr=.003, momentum=.5, foreach=True)

d_scaler = torch.amp.GradScaler(device)
g_scaler = torch.amp.GradScaler(device)

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(torch.zeros(IMG_SIZE, IMG_SIZE).numpy(), cmap="gray", vmin=0, vmax=1)
ax.axis("off")

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dataloader):
        torch.compiler.cudagraph_mark_step_begin()
        images = images.to(device, non_blocking=True)

        discriminator_optimizer.zero_grad()
        with torch.autocast(device_type=device):
            real_preds = D(images)
            fake_input = torch.rand((BATCH_SIZE, 100), device=device) * 2 - 1
            fake_preds = D(G(fake_input).detach())

            d_real_loss = torch.mean(-torch.log(real_preds.clamp(eps, 1 - eps)))
            d_fake_loss = torch.mean(-torch.log((1 - fake_preds).clamp(eps, 1 - eps)))
            d_loss = d_real_loss + d_fake_loss

        d_scaler.scale(d_loss).backward()
        d_scaler.step(discriminator_optimizer)
        d_scaler.update()

        generator_optimizer.zero_grad()
        with torch.autocast(device_type=device):
            fake_generations = G(fake_input)
            fake_preds = D(fake_generations)
            g_loss = torch.mean(-torch.log(fake_preds.clamp(eps, 1 - eps)))

        g_scaler.scale(g_loss).backward()
        g_scaler.step(generator_optimizer)
        g_scaler.update()

        if i % 5 == 0:
            im.set_data(fake_generations[0].cpu().view(IMG_SIZE, IMG_SIZE).detach().float().numpy())
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            print(f"Epoch {epoch} [{i}] D_Real: {d_real_loss.item():.3f}, D_Fake: {d_fake_loss.item():.3f}, G: {g_loss.item():.3f}")
