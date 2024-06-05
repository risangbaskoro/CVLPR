#!/usr/bin/env python
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from const import *


class SmallBasicBlock(nn.Module):
    """Small Basic Block for Residual
    Inspired from Squeeze Fire Blocks and Inception Blocks

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(SmallBasicBlock, self).__init__()
        out_div4 = out_channels // 4

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_div4, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(out_div4),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            # TODO: Check the stride h and pad h for Conv2d below (see LPRNet paper).
            nn.Conv2d(in_channels=out_div4, out_channels=out_div4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_div4),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            # TODO: Check the stride w and pad w for Conv2d below (see LPRNet paper).
            nn.Conv2d(in_channels=out_div4, out_channels=out_div4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_div4),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=out_div4, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class GlobalContext(nn.Module):
    def __init__(self, kernel_size: Union[int, tuple[int, int]], stride: Union[int, tuple[int, int], None] = None):
        super(GlobalContext, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        avg_pool = self.pool(x)
        sq = torch.square(avg_pool)
        sqm = torch.mean(sq)
        return torch.div(avg_pool, sqm)


class LPRNet(nn.Module):
    def __init__(self, num_classes: int, input_channel: int = 3, with_global_context: bool = False):
        super(LPRNet, self).__init__()
        self.use_global_context = with_global_context

        self.conv_1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.basic_block_1 = SmallBasicBlock(in_channels=64, out_channels=128)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1))
        self.basic_block_2 = SmallBasicBlock(in_channels=128, out_channels=256)
        self.basic_block_3 = SmallBasicBlock(in_channels=256, out_channels=256)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1))
        self.dropout_1 = nn.Dropout2d(p=0.5)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout2d(p=0.5)

        self.gc_1 = GlobalContext((1, 5), (3, 1))
        self.gc_2 = GlobalContext((1, 3), (1, 1))
        self.gc_3 = GlobalContext((3, 1), (3, 1))
        self.gc_4 = GlobalContext((3, 1), (3, 1))

        pre_decoder_in_channels = 960 if self.use_global_context else 256
        self.out_classes = nn.Sequential(
            nn.Conv2d(in_channels=pre_decoder_in_channels, out_channels=num_classes, kernel_size=(1, 13),
                      padding='same'),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def backbone_forward(self, x):
        out_1 = F.relu(self.max_pool_1(self.bn_1(self.conv_1(x))))
        out_2 = self.max_pool_2(self.basic_block_1(out_1))
        out_3 = self.basic_block_2(out_2)
        out_4 = self.max_pool_3(self.basic_block_3(out_3))
        x = self.dropout_1(out_4)
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = self.dropout_2(x)

        return x, out_1, out_2, out_3, out_4

    def global_context_forward(self, x, out_1, out_2, out_3, out_4):
        scale_1 = self.gc_1(out_1)
        scale_2 = self.gc_2(out_2)
        scale_3 = self.gc_3(out_3)
        scale_4 = self.gc_4(out_4)
        scale_5 = torch.div(x, torch.clone(x).square().mean())
        return torch.concat([scale_1, scale_2, scale_3, scale_4, scale_5], dim=1)

    def forward(self, x):
        x, out_1, out_2, out_3, out_4 = self.backbone_forward(x)
        if self.use_global_context:
            x = self.global_context_forward(x, out_1, out_2, out_3, out_4)
        x = self.out_classes(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    sample_img = torch.rand(size=(1, 3, 24, 94))

    n_classes = len(CHARS)
    model = LPRNet(n_classes, with_global_context=False)
    result = model(sample_img)
    print(result, result.shape)
    print()
    pred = F.log_softmax(torch.mean(result, dim=2), dim=-1)
    print(pred, pred.max(), pred.shape)
