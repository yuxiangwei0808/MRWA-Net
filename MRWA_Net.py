import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from einops import rearrange, repeat

xfm_3 = DWTForward(J=3, mode='periodization', wave='sym2').cuda()
xfm_2 = DWTForward(J=2, mode='periodization', wave='sym2').cuda()
xfm_1 = DWTForward(J=1, mode='periodization', wave='sym2').cuda()


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3


class Vox_Att(nn.Module):

    def __init__(self, in_ch):
        super(Vox_Att, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, int(in_ch / 2), kernel_size=1, padding=0),
            nn.BatchNorm2d(int(in_ch / 2)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(in_ch / 2), int(in_ch / 2), kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(int(in_ch / 2)),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(in_ch / 2), in_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.output = Hsigmoid()

    def forward(self, in_x):
        x = self.conv1(in_x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.output(x)

        # output = in_x * x + in_x
        return x


class SEAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class Attn_Block(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.se_attn = SEAttention(channel)
        self.vox_attn = Vox_Att(channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        # residual = x
        channel_attn = self.se_attn(x)
        vox_attn = self.vox_attn(x)
        out = x * channel_attn.expand_as(x)
        out = out * vox_attn
        # out = out * self.spatial_attn(out)
        return out


class DWT(nn.Module):
    def __init__(self, xfm):
        super(DWT, self).__init__()
        self.xfm = xfm

    def forward(self, x):
        Yl, _ = self.xfm(x)
        return Yl


class DWT_3(nn.Module):
    def __init__(self, xfm, J=1):
        super(DWT_3, self).__init__()
        self.xfm = xfm
        self.j = J

    def forward(self, x):
        Yl, Yh = self.xfm(x)
        if self.j == 3:
            out = torch.cat((Yh[2][:, :, 0, :, :], Yh[2][:, :, 1, :, :], Yl), dim=1)
        elif self.j == 2:
            out = torch.cat((Yh[1][:, :, 0, :, :], Yh[1][:, :, 1, :, :], Yl), dim=1)
        else:
            out = torch.cat((Yh[0][:, :, 0, :, :], Yh[0][:, :, 1, :, :], Yl), dim=1)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Dropout(0.65),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Dropout(0.65),
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


class MrwaNet(nn.Module):
    def __init__(self):
        super(MrwaNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.xfm3 = DWT_3(xfm_3, J=3)
        self.layer_xfm3 = nn.Sequential(ResidualBlock(192, 256),
                                        ResidualBlock(256, 256),
                                        Attn_Block(256)
                                        )

        self.xfm2 = DWT_3(xfm_2, J=2)
        self.layer_xfm2 = nn.Sequential(ResidualBlock(192, 128),
                                        ResidualBlock(128, 128),
                                        Attn_Block(128),
                                        DWT(xfm_1),
                                        ResidualBlock(128, 256),
                                        ResidualBlock(256, 256),
                                        Attn_Block(256)
                                        )

        self.xfm1 = DWT_3(xfm_1)
        self.layer_xfm1 = nn.Sequential(ResidualBlock(192, 128),
                                        ResidualBlock(128, 128),
                                        Attn_Block(128),
                                        DWT(xfm_1),
                                        ResidualBlock(128, 256),
                                        ResidualBlock(256, 256),
                                        Attn_Block(256),
                                        DWT(xfm_1),
                                        ResidualBlock(256, 512),
                                        ResidualBlock(512, 512),
                                        Attn_Block(512)
                                        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.conv(x)

        y3 = self.xfm3(x)
        y3 = self.layer_xfm3(y3)

        y2 = self.xfm2(x)
        y2 = self.layer_xfm2(y2)

        y1 = self.xfm1(x)
        y1 = self.layer_xfm1(y1)

        y = torch.cat((y1, y2, y3), dim=1)
        out = self.avg_pool(y)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

