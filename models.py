import torch
import torch.nn as nn
import layers as L
from torch.nn import init


def init_weights(net):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            stddev = 0.02
            mu = 0.0
            torch.nn.init.normal_(m.weight.data, mu, stddev)
            m.weight.data = torch.clamp(m.weight.data.clone(), mu - 2*stddev, mu+2*stddev)
            print(m, m.weight.data.max(), m.weight.data.min())
            # trunc = torch.clamp(trunc)
        elif isinstance(m, nn.InstanceNorm2d):
            stddev = 0.02
            mu = 1.0
            init.normal_(m.weight.data, mu, stddev)
            if hasattr(m, 'bias'):
                init.constant(m.bias.data, 0.0)
        # else:
            # print(m)
            # raise NotImplementedError

    net.apply(init_func)


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        layers = [nn.InstanceNorm2d(in_channels, affine=True)]
        layer_spec = [[3, 32, 3, 1], [32, 32, 3, 2], [32, 64, 3, 2], [64, 128, 3, 2], [128, 256, 3, 2]]
        for spec in layer_spec:
            in_channels, out_channels, kernel_size, stride = spec
            layers.append(L.ConvLayer(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        resblocks = []
        in_channels = 256
        out_channels = 256
        kernel_size = 3
        stride = 1
        for _ in range(9):
            resblocks.append(L.ResidualBlock(in_channels, out_channels, kernel_size, stride))
        self.resblocks = nn.Sequential(*resblocks)

        layers = []
        layer_spec = [[256, 256, 3, 1], [256, 128, 3, 1], [128, 64, 3, 1], [64, 32, 3, 1]]
        for spec in layer_spec:
            in_channels, out_channels, kernel_size, stride = spec
            layers.append(L.UpConvBlock(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride))
        self.uplayers = nn.Sequential(*layers)

        kernel_size = 7
        self.conv7x7 = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2),
                                     nn.Conv2d(in_channels=32, out_channels=3, kernel_size=kernel_size, stride=1))
        self.sigm = nn.Sigmoid()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.resblocks(x)
        x = self.uplayers(x)
        x = self.conv7x7(x)
        x = 2 * self.sigm(x) - 1
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernel_size = 5
        stride = 2
        leak = 0.2
        layers = []
        aux_layers = []
        self.aux_ids = [0, 1, 3, 5]
        aux_ks = [5, 10, 10, 6]
        layer_spec = [[3, 128], [128, 128], [128, 256], [256, 512], [512, 512], [512, 1024], [1024, 1024]]

        for layer_id, spec in enumerate(layer_spec):
            in_channels, out_channels = spec
            layers.append(L.LeakyLayer(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, leak=leak))
            if layer_id in self.aux_ids:
                aux_layers.append(nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=aux_ks[0], stride=1,
                                            padding=1, bias=True))
                aux_ks.pop(0)

        self.layers = nn.ModuleList(layers)
        self.aux_classifiers = nn.ModuleList(aux_layers)
        # self.classifier = L.ConvLayer(in_channels=in_channels, out_channels=1, kernel_size=10, stride=1, relu=False)
        self.classifier = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    bias=True)

    def forward(self, x):
        # x = self.instn(x.unsqueeze(1)).squeeze(1)
        aux_id = 0
        outputs = []
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id in self.aux_ids:
                outputs.append(self.aux_classifiers[aux_id](x))
                aux_id += 1
        outputs.append(self.classifier(x))
        # outputs = torch.cat(tuple([i.view(x.size(0), -1) for i in outputs]), dim=1)
        return outputs


class TransformerBlock(nn.Module):
    def __init__(self, kernel_size=10):
        super(TransformerBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1)

    def forward(self, x1, x2):
        return self.avg_pool(x1), self.avg_pool(x2)
