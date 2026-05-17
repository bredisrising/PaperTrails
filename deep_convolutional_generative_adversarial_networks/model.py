import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.projection = nn.Linear(input_size, 1024 * 2 * 2)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.view(-1, 1024, 2, 2)
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__() 

        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2 * 2 * 1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)