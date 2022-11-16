import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=0):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, k_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1):
        super().__init__()
        layers = [
            ConvBNReLU(in_ch, out_ch, k_size, 1, k_size // 2),
            ConvBNReLU(in_ch, out_ch, k_size, stride, k_size // 2)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Classifier(nn.Module):
    def __init__(self, k_size=5, channel=32, output=1):
        super().__init__()
        layers = [ConvBNReLU(1, channel, k_size, 1, k_size // 2)]
        for i in range(4):
            layers.append(ConvBlock(channel, channel, k_size, 1))
            layers.append(nn.AvgPool1d(4, 4))
        layers.append(nn.Conv1d(channel, output, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    net = Classifier()
    x = torch.randn(4, 1, 2048)
    y = net(x)
    print(net)
    print(x.shape, y.shape)
