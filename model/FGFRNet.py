import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.FAAF import FAAF
from model.TFE import Texture_Encoder
from model.sample_generator import SampleGenerator

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()

        self.dropout = nn.Dropout2d(0.2)  # 添加Dropout
        # Bottleneck structure: 1x1, 3x3, 1x1 convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride, padding=0)  # 1x1 conv
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)  # 3x3 conv
        self.bn2 = nn.BatchNorm2d(out_channels // 4)

        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, padding=0)  # 1x1 conv
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut path: If stride != 1 or in_channels != out_channels, apply a 1x1 convolution
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None


    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class FGFRNet(nn.Module):
    def __init__(self):
        super(FGFRNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(Res_CBAM_block, 3, 16)
        self.conv0_1 = self._make_layer(Res_CBAM_block, 16, 32, 3)
        self.conv1_2 = self._make_layer(Res_CBAM_block, 32,64, 4)
        self.conv2_3 = self._make_layer(Res_CBAM_block, 64,128, 6)
        self.conv3_4 = self._make_layer(Res_CBAM_block, 128,256, 3)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

        self.fuse1 = FAAF(128, 256,128)
        self.fuse2 = FAAF(64,64,64)
        self.fuse3 = FAAF(32,32,32)
        self.fuse4 = FAAF(16,16,16)

        self.tfe1 = Texture_Encoder(128,64,192)
        self.tfe2 = Texture_Encoder(64,32,48)
        self.tfe3 = Texture_Encoder(32,16,12)

        self.sample_generator = SampleGenerator()

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input,Y_mask):
        x0 = self.conv0_0(input) #16,256,256
        x1 = self.conv0_1(self.pool(x0)) #32,128,128
        x2 = self.conv1_2(self.pool(x1)) #64,64,64
        x3 = self.conv2_3(self.pool(x2))  # 16,128,32,32
        x4 = self.conv3_4(self.pool(x3))  # 16,256,16,16

        x3_,edge1 = self.fuse1(x3, self.up(x4)) #128,32,32
        x3_,x_glabel1=self.tfe1(x3_,input)

        x2_,edge2 = self.fuse2(x2,x3_) #64,64,64
        x2_,x_glabel2=self.tfe2(x2_,input)

        x1_,edge3= self.fuse3(x1,x2_) #32,128,128
        x1_,x_glabel3=self.tfe3(x1_,input)

        x00,edge4 = self.fuse4(x0, x1_) #16,256,256
        x_contras=x00
        F_feat_mlp,_,_,_,_,_,_, = self.sample_generator(x_contras,Y_mask, input)
        output=self.final_conv(F_feat_mlp)

        return (output,x_glabel1,x_glabel2,x_glabel3,x_contras,edge1,edge2,edge3,edge4)
