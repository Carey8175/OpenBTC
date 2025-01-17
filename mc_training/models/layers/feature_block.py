import torch
import torch.nn as nn
import torch.nn.functional as F

from mc_training.models.layers.column_normalization import PerFeatureNormalization


# class FeatureExtractBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, feature_num, window_size, stride=1, downsample=None):
#         super(FeatureExtractBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn = PerFeatureNormalization(method="min_max")
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         # 使用 1x1 卷积调整残差的通道数
#         self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
#
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels * window_size * feature_num, in_channels * feature_num),  # Flattened input: 64 channels, 30x5 feature map
#             nn.Linear(in_channels * feature_num, in_channels * window_size * feature_num))
#
#         # 每个通道一个可学习参数，此参数用于削弱残差连接
#         self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))  # 初始化为 1
#
#     def forward(self, x):
#         residual = self.bn(self.downsample(x))
#
#         out = self.fc(x.view(x.size(0), -1))  # Flatten input for fc
#         out = out.view(x.size(0), x.shape[1], x.shape[2], x.shape[3])  # Reshape back to 4D
#         out = self.conv1(out)
#         out = self.bn(out)
#
#         out = self.conv2(out)
#         out = self.bn(out)
#
#         # 削弱残差连接，同时保留一部分信息，防止梯度消失
#         out += residual
#
#         return out

class FeatureExtractBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_num, window_size, stride=1, downsample=None):
        super(FeatureExtractBlock, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = PerFeatureNormalization(method="min_max")

        # 卷积层2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = PerFeatureNormalization(method="min_max")

        # 使用1x1卷积调整残差的通道数
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        # 可学习参数alpha，用于控制残差的强度
        self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        # 添加Dropout层
        self.dropout = nn.Dropout(0.1)

        # 非线性激活函数
        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(in_channels * window_size * feature_num, in_channels * feature_num),  # Flattened input: 64 channels, 30x5 feature map
            nn.Linear(in_channels * feature_num, in_channels * window_size * feature_num))

    def forward(self, x):
        residual = self.downsample(x)  # 对输入进行下采样

        out = self.fc(x.view(x.size(0), -1))  # Flatten input for fc
        out = out.view(x.size(0), x.shape[1], x.shape[2], x.shape[3])  # Reshape back to 4D

        # 卷积操作 + 非线性激活 + 批归一化
        out = self.conv1(out)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.relu(self.bn2(out))

        # 加入Dropout层
        out = self.dropout(out)

        # 调整残差连接的强度
        out += self.alpha * residual

        return out

