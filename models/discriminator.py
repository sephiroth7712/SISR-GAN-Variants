from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    LeakyReLU,
    BatchNorm2d,
    AdaptiveAvgPool2d,
    init,
)
from torch import sigmoid


class Discriminator(Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.net = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),
            LeakyReLU(0.2),
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2),
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2),
            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2),
            AdaptiveAvgPool2d(1),
            Conv2d(512, 1024, kernel_size=1),
            LeakyReLU(0.2),
            Conv2d(1024, 1, kernel_size=1),
        )

    def _initialize_weights(self):
        for m in self.modules:
            if isinstance(m, Conv2d):
                init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.size(0)
        return sigmoid(self.net(x).view(batch_size))
