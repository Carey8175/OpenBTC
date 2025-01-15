import torch
import torch.nn as nn


class PositionalSortingLayer(nn.Module):
    def __init__(self, feature_dim):
        """
        使用位置编码进行特征排序
        :param feature_dim: 需要排序的特征维度
        """
        super(PositionalSortingLayer, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(feature_dim))  # 对最后一维进行位置编码

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征矩阵 (N, C, T, F)
        :return: 排列后的特征矩阵 (N, C, T, F)
        """
        # 获取最后一维的排序索引
        sorted_indices = torch.argsort(self.position_embedding, descending=True)  # (F,)

        # 对最后一维进行索引排序
        sorted_features = x[..., sorted_indices]  # 使用高级索引操作，对最后一维排序

        # Debug 信息
        # print("sorted_features[0][0][0][:]:", sorted_features[0, 0, 0, :].tolist())
        # print("x[0][0][0][:]:", x[0, 0, 0, :].tolist())

        return sorted_features, sorted_indices