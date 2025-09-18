import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CVDBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, drop_out_rate=0.1):
        """
        Convolutional Vision Difference (CVD) Block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            drop_out_rate (float): Dropout rate
        """
        super().__init__()
        c = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(c),
            Swish()
        )

        self.conv_branch = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(c * 2),
            Swish(),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1, stride=1,
                      groups=c // 2, bias=False),
            nn.BatchNorm2d(c * 2)
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(c, in_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )

        self.sg = SimpleGate()

        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x_branch = self.conv_branch(x)
        x_branch = self.sg(x_branch)
        channel_attn = self.sca(x)
        x_branch = x_branch * channel_attn
        x = self.conv_final(x_branch)
        x = self.dropout(x)
        x = x + identity

        return x


class CVDNet(nn.Module):
    def __init__(self, channels=64, num_blocks=3):
        """
        CVD Neural Network

        Args:
            channels (int): Base number of channels
            num_blocks (int): Number of CVD blocks to stack
        """
        super(CVDNet, self).__init__()

        # Multiple CVD blocks
        self.cvd_blocks = nn.ModuleList([
            CVDBlock(out_channels=channels) for _ in range(num_blocks)
        ])

    def forward(self, img, img_cvd):
        features_img = img
        features_img_cvd = img_cvd

        for block in self.cvd_blocks:
            features_img = block(features_img)
            features_img_cvd = block(features_img_cvd)

        out = torch.cat([features_img, features_img_cvd], dim=1)

        return out


