import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(ParallelDilatedConv1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation_rates[0],
            padding=dilation_rates[0],
        )
        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation_rates[1],
            padding=dilation_rates[1],
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)  # 沿通道维度合并
        return x
