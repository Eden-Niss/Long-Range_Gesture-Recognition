from Hourglass.simple_UNET import UNET
import torch.nn as nn


class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()
        self.unet = UNET()
        # self.runet = RUNet()

    def forward(self, x, num_stack=3):
        for i in range(num_stack):
            x0 = x.clone()
            x = self.unet(x)
            x += x0
            # x = self.runet(x)
        del x0
        return x


