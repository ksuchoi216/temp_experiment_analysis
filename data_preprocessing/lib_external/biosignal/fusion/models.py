import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(in_ch, out_ch, k_size, stride, padding, act=None):
    layers = [
        nn.Conv1d(in_ch, out_ch, k_size, stride, padding),
        nn.BatchNorm1d(out_ch),
    ]
    if act is not None:
        layers.append(act)
    return nn.Sequential(*layers)


def DeconvBlock(in_ch, out_ch, k_size, stride, padding, act=None):
    layers = [
        nn.ConvTranspose1d(in_ch, out_ch, k_size, stride, padding),
        nn.BatchNorm1d(out_ch),
    ]
    if act is not None:
        layers.append(act)
    return nn.Sequential(*layers)


# Encoder

class Encoder_V0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(1, 16, 4, 4, 0, nn.GELU()),
            ConvBlock(16, 32, 4, 4, 0, nn.GELU()),
            ConvBlock(32, 64, 4, 4, 0, nn.GELU()),
            ConvBlock(64, 128, 4, 4, 0, nn.GELU()),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(1, 8, 3, 1, 1, nn.GELU()),
            ConvBlock(8, 16, 4, 4, 0, nn.GELU()),
            ConvBlock(16, 16, 3, 1, 1, nn.GELU()),
            ConvBlock(16, 32, 4, 4, 0, nn.GELU()),
            ConvBlock(32, 32, 3, 1, 1, nn.GELU()),
            ConvBlock(32, 64, 4, 4, 0, nn.GELU()),
            ConvBlock(64, 64, 3, 1, 1, nn.GELU()),
            ConvBlock(64, 128, 4, 4, 0),
        )

    def forward(self, x):
        return self.layers(x)


# Decoder

class Decoder_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            DeconvBlock(128, 64, 4, 4, 0, nn.GELU()),
            ConvBlock(64, 64, 3, 1, 1, nn.GELU()),
            DeconvBlock(64, 32, 4, 4, 0, nn.GELU()),
            ConvBlock(32, 32, 3, 1, 1, nn.GELU()),
            DeconvBlock(32, 16, 4, 4, 0, nn.GELU()),
            ConvBlock(16, 16, 3, 1, 1, nn.GELU()),
            DeconvBlock(16, 8, 4, 4, 0, nn.GELU()),
            ConvBlock(8, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, 1)

    def forward(self, x):
        scores = self.lin(x)
        weights = F.softmax(scores, dim=-2)
        x = torch.sum(x * weights, dim=-2)
        return x


class ScaledDotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        dim_k = key.size(-1)
        scores = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(dim_k)
        weights = F.softmax(scores, dim=-1)
        outputs = torch.bmm(weights, value)
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, d_model, output_size):
        super().__init__()
        self.query = nn.Linear(d_model, output_size)
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)
        self.dot_att = ScaledDotAttention()

    def forward(self, q, k, v):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        outputs = self.dot_att(query, key, value)
        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_output_size = d_model // num_heads
        self.attentions = nn.ModuleList(
            [
                SelfAttention(d_model, self.attn_output_size)
                for _ in range(self.num_heads)
            ],
        )
        self.output = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        x = torch.cat(
            [
                layer(q, k, v) for layer in self.attentions
            ], dim=-1
        )
        x = self.output(x)
        return x


# Fusion Module

class Fuser_V0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(-2)


class Fuser_V1(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.att = Attention(d_model)

    def forward(self, x):
        return self.att(x)


class Fuser_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sda = ScaledDotAttention()

    def forward(self, x):
        return self.sda(x, x, x).mean(-2)


class Fuser_V3(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.sda = ScaledDotAttention()
        self.att = Attention(d_model)

    def forward(self, x):
        return self.att(self.sda(x, x, x))


class Fuser_V4(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        return self.mha(x, x, x).mean(-2)


class Fuser_V5(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.att = Attention(d_model)

    def forward(self, x):
        return self.att(self.mha(x, x, x))


class Fuser_V6(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.att = Attention(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(x, x, x)
        x = self.norm(x)
        x = self.att(x)
        return x


class Combo(nn.Module):
    """
    Encoder, Decoder, Fusion Module, Recurrent Module를 하나로 합한 모듈입니다.
    """

    def __init__(self, enc, fus, dec, rnn=None):
        super().__init__()
        self.enc = enc
        self.fus = fus
        self.rnn = rnn
        self.dec = dec

    def forward(self, x):
        """
        N : batch size
        M : number of sensors
        L : signal length
        C : number of features
        T : number of windows
        """

        # print(x.shape)
        N, M, L = x.shape
        x = x.reshape(N*M, 1, L)
        # print(x.shape)  # NM x 1 x L
        x = self.enc(x)
        # print(x.shape)  # NM x C x T
        x = x.permute(2, 0, 1)
        # print(x.shape)  # T x NM x C
        T, NM, C = x.shape
        x = x.reshape(T*N, M, C)
        # print(x.shape)  # TN x M x C
        x = self.fus(x)
        # print(x.shape)  # TN x C
        x = x.reshape(T, N, C)
        # print(x.shape)  # T x N x C
        if self.rnn is not None:
            x, h = self.rnn(x)
            # print(x.shape)  # T x N x C
        x = x.permute(1, 2, 0)
        # print(x.shape)  # N x C x T
        x = self.dec(x)
        # print(x.shape)  # N x 1 x L
        return x


def get_signal_reconstruction_model():
    net = Combo(
        enc=Encoder_V0(),
        dec=Decoder_V1(),
        fus=Fuser_V6(128, 8),
        rnn=nn.LSTM(128, 128, num_layers=1),
    )
    return net


def get_frequency_estimation_model():
    net = Combo(
        enc=Encoder_V0(),
        dec=nn.Conv1d(128, 1, 1),
        fus=Fuser_V6(128, 8),
        rnn=nn.LSTM(128, 128, num_layers=1),
    )
    return net


if __name__ == '__main__':
    # Signal Reconstruction Model
    net = get_signal_reconstruction_model()
    x = torch.randn(4, 16, 2048)
    y = net(x)
    print(x.shape, y.shape)

    # Frequency Estimation
    net = get_frequency_estimation_model()
    x = torch.randn(4, 16, 2048)
    y = net(x)
    print(x.shape, y.shape)
