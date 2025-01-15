import torch
import torch.nn as nn
import torch.nn.functional as F

from mc_training.models.layers.column_normalization import PerFeatureNormalization
from mc_training.models.layers.feature_block import FeatureExtractBlock


class MCModel(nn.Module):
    def __init__(self, class_num=3, feature_num=10, window_size=30):
        super(MCModel, self).__init__()
        # 特征提取层
        self.feb1 = FeatureExtractBlock(1, 16, feature_num, window_size)
        self.feb2 = FeatureExtractBlock(16, 32, feature_num, window_size)
        self.feb3 = FeatureExtractBlock(32, 64, feature_num, window_size)
        # 标准化
        self.pfn = PerFeatureNormalization(method="min_max")
        # dense
        self.fc = nn.Sequential(
            nn.Linear(64 * feature_num * window_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, class_num)
        )

    def forward(self, x):
        x = self.feb1(x)
        x = self.feb2(x)
        x = self.feb3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = MCModel(class_num=2, feature_num=66, window_size=30)
    x = torch.randn(32, 1, 30, 66)
    y = model(x)
    print(y.shape)
    print(y)