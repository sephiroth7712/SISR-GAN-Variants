from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    PReLU,
    BatchNorm2d,
)
import torch
import math
from models.common import ResidualBlock, UpsampleBLock


class SRGAN(Module):
    def __init__(self, ngpu, scale):
        if scale < 2 or scale & (scale - 1):
            raise ValueError(
                "The SRGAN generator supports power-of-2 scales only, got %d" % scale
            )
        upsample_block_num = int(math.log(scale, 2))
        super(SRGAN, self).__init__()

        self.ngpu = ngpu
        self.scale = scale

        self.block1 = Sequential(Conv2d(3, 64, kernel_size=9, padding=4), PReLU())
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = Sequential(
            Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2
