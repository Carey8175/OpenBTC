import torch
import torch.nn as nn
import torch.nn.functional as F

from mc_training.models.layers.column_normalization import PerFeatureNormalization


class FeatureExtractBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_num, window_size, stride=1, downsample=None):
        super(FeatureExtractBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = PerFeatureNormalization(method="min_max")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 使用 1x1 卷积调整残差的通道数
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(in_channels * window_size * feature_num, in_channels * feature_num),  # Flattened input: 64 channels, 30x5 feature map
            nn.Linear(in_channels * feature_num, in_channels * window_size * feature_num))

        # 每个通道一个可学习参数，此参数用于削弱残差连接
        self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))  # 初始化为 1

    def forward(self, x):
        residual = self.bn(self.downsample(x))

        out = self.fc(x.view(x.size(0), -1))  # Flatten input for fc
        out = out.view(x.size(0), x.shape[1], x.shape[2], x.shape[3])  # Reshape back to 4D
        out = self.conv1(out)
        out = self.bn(out)

        out = self.conv2(out)
        out = self.bn(out)

        # 削弱残差连接，同时保留一部分信息，防止梯度消失
        out += residual

        return out