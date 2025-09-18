import torch
from torch import nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CVDBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(c)

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=1, padding=0, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(c * 2)

        self.conv2 = nn.Conv2d(in_channels=c * 2, out_channels=c * 2, kernel_size=3, padding=1, stride=1,
                               groups=c // 2, bias=True)
        self.bn2 = nn.BatchNorm2d(c * 2)

        self.conv3= nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(c)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn4 = nn.BatchNorm2d(c)

        self.conv5 = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True)
        self.bn5 = nn.BatchNorm2d(3)

        self.swish = Swish()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True),
        )
        
        # Simplified Channel Attention
        self.sca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )
        self.sca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )

        self.sg = SimpleGate()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sg(x)
        
        x_avg, x_max = x.chunk(2, dim=1)
        x_avg = self.sca_avg(x_avg)*x_avg
        x_max = self.sca_max(x_max)*x_max
        x = torch.cat([x_avg, x_max], dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.swish(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.swish(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.swish(x)
        x = self.dropout1(x)

        return x

class CVDNet(nn.Module):
    def __init__(self, c):
        super(CVDNet, self).__init__()
        self.cvd_block = CVDBlock(c)
        self.proj = nn.Conv2d(3, 6, kernel_size=1)

    def forward(self, img, img_cvd):
        features_img = self.cvd_block(img)
        features_img_cvd = self.cvd_block(img_cvd)
        # out = torch.cat([features_img, features_img_cvd], dim=1)
        out = features_img - features_img_cvd
        out = self.proj(out)

        return out