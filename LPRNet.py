#!/usr/bin/env python
from typing import Union

import torch
from torch import nn

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


# TODO: Create Beam Search Module for decoding Plate


class LPRNet(nn.Module):
    def __init__(self, num_classes: int, input_channel: int = 3, use_global_context: bool = False):
        super(LPRNet, self).__init__()
        self.use_global_context = use_global_context

        self.layers = nn.ModuleDict({
            "conv_1": nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            "bn_1": nn.BatchNorm2d(64),
            "relu_1": nn.ReLU(),
            "max_pool_1": nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            "basic_block_1": SmallBasicBlock(in_channels=64, out_channels=128),
            "max_pool_2": nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            "basic_block_2": SmallBasicBlock(in_channels=128, out_channels=256),
            "basic_block_3": SmallBasicBlock(in_channels=256, out_channels=256),
            "max_pool_3": nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            "dropout_1": nn.Dropout2d(p=0.5),
            "conv_2": nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=1, padding=1),
            "bn_2": nn.BatchNorm2d(256),
            "relu_2": nn.ReLU(),
            "dropout_2": nn.Dropout2d(p=0.5),
        })

        self.gc = nn.ModuleList([
            GlobalContext((1, 5), (8, 1)),
            GlobalContext((2, 3), (7, 1)),
            GlobalContext((3, 1), (3, 1)),
            GlobalContext((3, 1), (3, 1)),
        ])

        pre_decoder_in_channels = 960 if use_global_context else 256
        self.pre_decoder = nn.Conv2d(in_channels=pre_decoder_in_channels, out_channels=num_classes, kernel_size=(1, 13), padding='same')

    def forward(self, x):
        outputs = []

        for layer in self.layers:
            x = self.layers[layer](x)
            outputs.append(x)

        # TODO: Fix global context concat dimension.
        if self.use_global_context:
            scale_1 = self.gc[0](outputs[0])
            scale_2 = self.gc[1](outputs[4])
            scale_3 = self.gc[2](outputs[6])
            scale_5 = self.gc[3](outputs[7])
            scale_4 = torch.div(x, x.square().mean())
            print(f"Original x shape: {x.shape}\n")
            print(f"Scale 1 shape: {scale_1.shape}")
            print(f"Scale 2 shape: {scale_2.shape}")
            print(f"Scale 3 shape: {scale_3.shape}")
            print(f"Scale 5 shape: {scale_5.shape}")
            print(f"\nScale 4 shape: {scale_4.shape}\n")
            x = torch.concat([scale_1, scale_2, scale_3, scale_5, scale_4], dim=1)

        x = self.pre_decoder(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    sample_img = torch.rand(size=(1, 3, 24, 94))

    n_classes = len(CHARS)
    model = LPRNet(n_classes, use_global_context=False)
    result = model(sample_img)
    print(result, result.shape)
    print()
    pred = torch.softmax(result, dim=3).argmax(dim=3)
    print(pred, pred.shape)
