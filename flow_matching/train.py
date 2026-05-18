import torch
from model import UNet
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
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
BATCH_SIZE = 128
eps = 1e-7

IMG_SIZE = 32
cifar = CIFAR10('../data', download=True, transform=ToTensor())


dataloader = DataLoader(
    cifar, BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=4, pin_memory=(device == 'cuda'), persistent_workers=True, prefetch_factor=2
)

compile_mode = 'default'

net = UNet()

# if os.path.

optim = torch.optim.Adam()
scaler = torch.amp.GradScaler(device)

os.makedirs('samples', exist_ok=True)

for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)




        pass