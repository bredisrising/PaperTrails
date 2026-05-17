import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.projection = nn.Linear(input_size + 10, 2048 * 1 * 1)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(2048, 2048, 4, 2, 1), # 2x2
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1), # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1), # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 3, 4, 2, 1), # 32x32
            nn.Sigmoid()
        )

    def forward(self, x, label):
        x = torch.cat([x, self.label_emb(label)], dim=1)
        x = self.projection(x)
        x = x.view(-1, 2048, 1, 1)
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(10, 32*32)
        self.model = nn.Sequential(
            nn.Conv2d(4, 128, 3, 2, 1), # 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), # 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), # 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1), # 2
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2 * 2 * 1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        one_hot = torch.zeros((x.shape[0], 10), device=x.device)
        one_hot[torch.arange(x.shape[0]), labels] = 1.0
        label_channel = self.projection(one_hot)
        label_channel = label_channel.view(-1, 1, 32, 32)

        x = torch.concat([x, label_channel], dim=1)

        return self.model(x)