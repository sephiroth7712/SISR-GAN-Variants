from torch.nn import Module, Sequential, Conv2d, LeakyReLU, PixelShuffle, init
import torch
import math


class ESPCN(Module):
    def __init__(self, ngpu, scale):
        super(ESPCN, self).__init__()
        self.ngpu = ngpu
        self.scale = scale

        self.p1 = Sequential(
            Conv2d(3, 64, kernel_size=5, padding="same"),
            LeakyReLU(negative_slope=0.2),
            Conv2d(64, 32, kernel_size=3, padding="same"),
            LeakyReLU(negative_slope=0.2),
        )

        self.p2 = Sequential(
            Conv2d(32, 3 * scale**2, kernel_size=3, padding="same"), PixelShuffle(scale)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                if m.in_channels == 32:
                    init.normal_(m.weight.data, mean=0.0, std=0.001)
                    init.zeros_(m.bias.data)
                else:
                    init.normal_(
                        m.weight.data,
                        mean=0.0,
                        std=math.sqrt(
                            2 / (m.out_channels * m.weight.data[0][0].numel())
                        ),
                    )
                    init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        # x = torch.clamp_(x, 0.0, 1.0)
        return x
