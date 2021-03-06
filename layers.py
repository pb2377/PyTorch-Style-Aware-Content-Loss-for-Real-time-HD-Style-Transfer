import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, pad='SAME'):
        super(ConvLayer, self).__init__()
        assert pad in ['SAME', 'VALID', 'REFLECT']
        reflection_padding = kernel_size // 2
        reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        if pad == 'VALID' or pad == 'REFLECT':
            padding = 0
        elif pad == 'SAME':
            padding = kernel_size // 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        instn = nn.InstanceNorm2d(out_channels,  affine=True)  #, track_running_stats=True)

        if relu and pad == 'REFLECT':
            self.layer = nn.Sequential(*(reflection_pad, conv, instn, nn.ReLU()))
        elif pad == 'REFLECT' and not relu:
            self.layer = nn.Sequential(*(reflection_pad, conv, instn))
        elif relu:
            self.layer = nn.Sequential(*(conv, instn, nn.ReLU()))
        elif not relu:
            self.layer = nn.Sequential(*(conv, instn))
        else:
            print(relu, pad)
            raise NotImplementedError

    def forward(self, x):
        return self.layer(x)


class LeakyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leak=0.2):
        super(LeakyLayer, self).__init__()
        # reflection_padding = kernel_size // 2
        # reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                         padding=kernel_size // 2)
        instn = nn.InstanceNorm2d(out_channels,  affine=True)  #, track_running_stats=True)
        relu = nn.LeakyReLU(leak)

        # self.layer = nn.Sequential(*(reflection_pad, conv, instn, relu))
        self.layer = nn.Sequential(*(conv, instn, relu))

    def forward(self, x):
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        block = []
        relu = True
        for _ in range(2):
            block.append(ConvLayer(in_channels, out_channels, kernel_size, stride, relu=relu, pad='REFLECT'))
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
        self.block = ConvLayer(in_channels, out_channels, kernel_size, stride, relu=relu, pad='SAME')

    def forward(self, x):
        return self.block(F.interpolate(x, scale_factor=2, mode='nearest'))


# class UpConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, relu=None):
#         super(UpConvBlock, self).__init__()
#         # self.block = ConvLayer(in_channels, out_channels/, kernel_size, stride, relu=relu)
#         self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=False)
#         self.instn = nn.InstanceNorm2d(out_channels,  affine=True)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.relu(self.instn(self.conv(x)))

