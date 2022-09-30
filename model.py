import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feats=64, n_res_blocks=9):
        super(Generator, self).__init__()

        # Input Conv Block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_feats, 7),
            nn.InstanceNorm2d(n_feats),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_feats = n_feats
        out_feats = in_feats*2
        for _ in range(2):
            layers += [
                nn.Conv2d(in_feats, out_feats, 3, 2, 1),
                nn.InstanceNorm2d(out_feats),
                nn.ReLU(inplace=True)
            ]
            in_feats = out_feats
            out_feats = in_feats*2

        # Residual Blocks
        for _ in range(n_res_blocks):
            layers += [ResidualBlock(in_feats)]

        # Upsampling
        out_feats = in_feats//2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(in_feats, out_feats, 3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(out_feats),
                nn.ReLU(inplace=True)
            ]
            in_feats = out_feats
            out_feats = in_feats//2

        # Output Conv Block
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_feats, out_channels, 7),
            nn.Tanh()
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, n_feats=64):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(in_channels, n_feats, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for _ in range(2):
            layers += [
                nn.Conv2d(n_feats, n_feats*2, 4, 2, 1),
                nn.InstanceNorm2d(n_feats*2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            n_feats = n_feats*2

        layers += [
            nn.Conv2d(n_feats, n_feats*2, 4, padding=1),
            nn.InstanceNorm2d(n_feats*2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        layers += [
            nn.Conv2d(n_feats*2, 1, 4, padding=1)
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return F.avg_pool2d(x, x.shape[2:]).view(x.shape[0], -1)


if __name__ == "__main__":
    generator = Generator()
    input = torch.randn(10, 3, 128, 128)
    output = generator(input)
    print(output.shape)

    discriminator = Discriminator()
    output = discriminator(output)
    print(output.shape)
