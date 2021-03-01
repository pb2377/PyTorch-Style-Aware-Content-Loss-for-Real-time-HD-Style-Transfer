import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        instn = nn.InstanceNorm2d(out_channels,  affine=True)

        if relu:
            self.layer = nn.Sequential(*(reflection_pad, conv, instn, nn.ReLU()))
        else:
            self.layer = nn.Sequential(*(reflection_pad, conv, instn))

    def forward(self, x):
        return self.layer(x)


class LeakyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leak=0.01):
        super(LeakyLayer, self).__init__()
        reflection_padding = kernel_size // 2
        reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        instn = nn.InstanceNorm2d(out_channels,  affine=True)
        relu = nn.LeakyReLU(leak)

        self.layer = nn.Sequential(*(reflection_pad, conv, instn, relu))

    def forward(self, x):
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        block = []
        relu = True
        for _ in range(2):
            block.append(ConvLayer(in_channels, out_channels, kernel_size, stride, relu=relu))
            relu = False
        self.block = nn.Sequential(*block)

    def forward(self, x):
        resi = x
        x = self.block(x)
        x = x + resi
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True):
        super(UpConvBlock, self).__init__()
        self.block = ConvLayer(in_channels, out_channels, kernel_size, stride, relu=relu)

    def forward(self, x):
        return self.block(F.interpolate(x, scale_factor=2, mode='nearest'))