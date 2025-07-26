# 频率注意力感知融合模块 Frequency Attention Aware Fusion Module FAAF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FAAF(nn.Module):
    def __init__(self,en_features, de_features, out_features):
        super(FAAF,self).__init__()
        self.sa=SpatialAttention()
        self.ca=ChannelAttention(out_features)
        self.dog = DoG_att(sigma1=1.0, sigma2=2.0)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.freq_weight = nn.Parameter(torch.ones(1, out_features, 1, 1))
        self.en=nn.Sequential(
            nn.Conv2d(en_features,en_features//4,kernel_size=1),
            nn.BatchNorm2d(en_features//4),
            nn.ReLU(True),

            nn.Conv2d(en_features//4, out_features, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_features),
            nn.Sigmoid()
        )
        self.de=nn.Sequential(
            nn.Conv2d(de_features, de_features//4, kernel_size=1),
            nn.BatchNorm2d(de_features//4),
            nn.ReLU(True),

            nn.Conv2d(de_features//4, out_features, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_features),
            nn.Sigmoid()
        )
        self.tail_conv=nn.Sequential(
            nn.Conv2d(out_features, out_features,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(out_features*2, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True)
        )
        self.conv_label=nn.Conv2d(out_features,1,1)

    def forward(self, x_en, x_de):

        x_e = self.en(x_en)
        x_d = self.de(x_de)
        if x_en.size(3) != x_de.size(3):
            x_d = F.interpolate(x_d, size=[x_e.size(2), x_e.size(3)], mode='bilinear', align_corners=True)

        x_ed = x_e + x_d

        # 使用DoG提取高频信息
        x_high = self.dog(x_ed)
        x_high = x_high * self.alpha

        # FFT变换
        x_ft = torch.fft.fft2(x_ed)
        x_ft = x_high + x_ft

        x_f=torch.fft.ifft2(x_ft).real

        x_ca = x_ed * self.ca(x_ed)
        x_sa = x_ca * self.sa(x_ca)

        x_ed1 = torch.cat((x_sa, x_f), dim=1)
        x_ed1 = self.conv(x_ed1)
        x_final=x_ed1+x_ed

        edge_pred=self.conv_label(x_f)

        return x_final,edge_pred


class DoG_att(nn.Module):
    def __init__(self, sigma1=1.0, sigma2=1.6):
        super(DoG_att, self).__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def gaussian_kernel(self, size, sigma):
        x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size))
        d = torch.sqrt(x * x + y * y)
        g = torch.exp(-(d ** 2 / (2.0 * sigma ** 2)))
        return g / g.sum()

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        kernel_size = 7
        g1 = self.gaussian_kernel(kernel_size, self.sigma1).to(device)
        g2 = self.gaussian_kernel(kernel_size, self.sigma2).to(device)

        g1 = g1.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
        g2 = g2.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)

        x1 = F.conv2d(x, g1, padding=kernel_size // 2, groups=c)
        x2 = F.conv2d(x, g2, padding=kernel_size // 2, groups=c)

        dog = x1 - x2
        return dog

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
