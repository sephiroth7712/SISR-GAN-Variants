import torch
from torch.nn import Conv2d, ReLU, BatchNorm2d, Sequential, Module, PixelShuffle, PReLU
import math
from models.common import ResidualBlock


class EDSR(Module):
    def __init__(self, ngpu, scale):
        super(EDSR, self).__init__()

        n_resblocks = 32
        n_feats = 256
        kernel_size = 3
        self.scale = scale
        self.ngpu = ngpu
        act = ReLU(True)
        n_colors = 3

        # define head module
        m_head = [default_conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResidualBlock(n_feats, kernel_size, batch_norm=False)
            for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(default_conv, scale, n_feats, act=False),
            default_conv(n_feats, n_colors, kernel_size),
        ]

        self.head = Sequential(*m_head)
        self.body = Sequential(*m_body)
        self.tail = Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return torch.clamp_(x, 0.0, 1.0)


class Upsampler(Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(PixelShuffle(2))
                if bn:
                    m.append(BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(ReLU(True))
                elif act == "prelu":
                    m.append(PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(PixelShuffle(3))
            if bn:
                m.append(BatchNorm2d(n_feats))
            if act == "relu":
                m.append(ReLU(True))
            elif act == "prelu":
                m.append(PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )
