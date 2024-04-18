from math import sqrt
import torch
from torch.nn import Sequential, Conv2d, PReLU, ConvTranspose2d, Module, init


class FSRCNN(Module):
    def __init__(self, ngpu, scale) -> None:
        super(FSRCNN, self).__init__()
        self.ngpu = ngpu
        self.scale = scale

        self.feature_extraction = Sequential(
            Conv2d(3, 56, (5, 5), (1, 1), (2, 2)), PReLU(56)
        )

        self.shrink = Sequential(Conv2d(56, 12, (1, 1), (1, 1), (0, 0)), PReLU(12))

        self.map = Sequential(
            Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            PReLU(12),
            Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            PReLU(12),
            Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            PReLU(12),
            Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            PReLU(12),
        )

        self.expand = Sequential(Conv2d(12, 56, (1, 1), (1, 1), (0, 0)), PReLU(56))

        self.deconv = ConvTranspose2d(
            56,
            3,
            (9, 9),
            (self.scale, self.scale),
            (4, 4),
            (self.scale - 1, self.scale - 1),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return torch.clamp_(out, 0.0, 1.0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.normal_(
                    m.weight.data,
                    mean=0.0,
                    std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())),
                )
                init.zeros_(m.bias.data)

        init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        init.zeros_(self.deconv.bias.data)
