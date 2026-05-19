import torch
import torch.nn as nn
from torchinfo import summary

class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1),
            nn.GroupNorm(8, output_channels),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
            nn.GroupNorm(8, output_channels),
            nn.LeakyReLU(),

        )
        # downsample
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.model(x)
        downed = self.down(x)
        return downed, x



class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, input_channels // 2, 2, 2)
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1),
            nn.GroupNorm(8, output_channels),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
            nn.GroupNorm(8, output_channels),
            nn.LeakyReLU(),
        )
    
    def forward(self, x, skipped):
        x = self.upsample(x)
        # print(x.shape, skipped.shape)
        x = torch.cat([x, skipped], dim=1) 
        x = self.model(x)
        return x


class UNet(nn.Module):
    def __init__(self, layers=3, img_size=32):
        super().__init__()
        self.layers = layers
        self.img_size = img_size

        self.t_projection = nn.Linear(1, img_size * img_size)

        self.channels = 64 
        self.initial = nn.Sequential(
            nn.Conv2d(4, self.channels, 3, 1, 1),
            nn.GroupNorm(8, self.channels),
            nn.LeakyReLU(),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1),
            nn.GroupNorm(8, self.channels),
            nn.LeakyReLU(),
        )

        self.downsamples = nn.ModuleList()
        for i in range(self.layers):
            self.downsamples.append(Downsample(self.channels, self.channels * 2))
            self.channels *= 2

        self.double_channel = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2, 3, 1, 1),
            nn.GroupNorm(8, self.channels * 2),
            nn.LeakyReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, 1, 1),
            nn.GroupNorm(8, self.channels * 2),
            nn.LeakyReLU(),
        )

        self.channels *= 2

        self.upsamples = nn.ModuleList()
        for i in range(self.layers):
            self.upsamples.append(Upsample(self.channels, self.channels // 2))
            self.channels //= 2


        self.end = nn.Sequential(
            nn.Conv2d(self.channels, 3, 3, 1, 1),
            # nn.Tanh(),
        )

    def forward(self, x, t):
        # t = self.t_projection(t).view(-1, 1, self.img_size, self.img_size)
        t = t.expand(-1, 1, self.img_size, self.img_size)
        init = torch.cat([x, t], dim=1)
        init = self.initial(init)
        x = init
        skips = []
        for downsample in self.downsamples:
            x, skip = downsample(x)
            # print(x.shape, skip.shape)
            skips.append(skip) 

        # print()
        x = self.double_channel(x)
        # print(x.shape)
        # print()

        for i, upsample in enumerate(self.upsamples):
            # print(i, x.shape, skips[self.layers - i - 1].shape) 
            x = upsample(x, skips[self.layers - i - 1])

        x = self.end(x)

        return x


if __name__ == "__main__":
    unet = UNet()
    summary(unet, input_size=(1, 3, 32, 32), device='cpu')