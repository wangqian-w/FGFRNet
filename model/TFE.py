# 纹理特征增强模块 TFE Texture Feature Enhancement
import torch
from torch import nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Texture_Encoder(nn.Module):
    def __init__(self, in_channel, out_channel,input_channel):
        super(Texture_Encoder, self).__init__()
        self.relu = nn.ReLU(True)
        self.down=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion_conv = nn.Conv2d(in_channel + input_channel, in_channel, kernel_size=1)
        self.branch0 = nn.Sequential(
            BasicConv(in_channel, out_channel, 1),
            nn.ReLU(True)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_channel, out_channel, 1),
            BasicConv(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(True)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channel, out_channel, 1),
            BasicConv(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(True)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channel, out_channel, 1),
            BasicConv(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(True)
        )
        self.conv_cat = nn.Sequential(
            BasicConv(4*out_channel, out_channel, 3,padding=1),
            nn.ReLU(True)
        )
        self.conv_res = nn.Sequential(
            BasicConv(in_channel, out_channel, 1),
            nn.ReLU(True)
        )
        self.conv_label = nn.Sequential(
            BasicConv(out_channel, 1,1),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(BasicConv(out_channel, out_channel, 3, padding=1))
    def get_patches_batch(self, x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in torch.split(column_x, split_size_or_sections=_size_h, dim=-2)]
            patch_sample = torch.cat(patches_x, dim=1)
            patches_batch.append(patch_sample)
        return torch.cat(patches_batch, dim=0)

    def forward(self, x, input):
        if input.size(2)>x.size(2):
            x_patches=self.get_patches_batch(input,x)
        elif input.size(2)<x.size(2):
            x_patches=F.interpolate(input,size=(x.size(2), x.size(3)),mode='bilinear',align_corners=True)
        else:
            x_patches=input
        x_=torch.cat((x,x_patches),dim=1)
        x_ = self.fusion_conv(x_)

        x0 = self.branch0(x_)
        x1 = self.branch1(x_)
        x2 = self.branch2(x_)
        x3 = self.branch3(x_)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x_grad = self.conv(x_cat + self.conv_res(x))
        x_glabel=self.conv_label(x_grad)
        return x_grad,x_glabel