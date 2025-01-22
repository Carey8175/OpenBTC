import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x


class FeatureExtractorLSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers):
        super(FeatureExtractorLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 卷积层提取图像特征
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化

        # LSTM层，输入为卷积层输出的特征
        self.lstm = nn.LSTM(32 * 15 * 15, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # 通过卷积层提取特征
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # 展平卷积层输出
        x = x.view(x.size(0), -1)  # 展平为(batch_size, feature_dim)

        # 将展平后的特征送入LSTM，增加一个时间步维度
        x = x.unsqueeze(1)  # 增加一个时间步维度，使其适应LSTM输入(batch_size, sequence_length, input_size)

        # LSTM前向传播，不需要手动初始化h0和c0
        out, _ = self.lstm(x)
        return out


class FeatureExtractorTransformer(nn.Module):
    def __init__(self, input_channels, patch_size, hidden_size, num_heads, num_layers):
        super(FeatureExtractorTransformer, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 计算每个patch的大小
        self.patch_dim = patch_size * patch_size * input_channels

        # 使用卷积将图像划分为多个patch
        self.conv = nn.Conv2d(input_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x):
        # 通过卷积层将图像划分为多个patch
        x = self.conv(x)  # 输出形状 (batch_size, hidden_size, height/patch_size, width/patch_size)

        # 展平图像的空间维度，使其适应Transformer
        x = x.flatten(2)  # 输出形状 (batch_size, hidden_size, num_patches)
        x = x.permute(0, 2, 1)  # 转置为 (batch_size, num_patches, hidden_size)

        # 通过Transformer提取特征
        out = self.transformer(x)

        return out


if __name__ == '__main__':
    input_channels = 3  # 输入图像的通道数
    input_data = torch.randn(1, input_channels, 60, 60)

    model1 = FeatureExtractorCNN()
    output = model1(input_data)
    print("cnn features shape:", output.shape)

    hidden_size_LSTM = 50  # LSTM的隐藏状态维度
    num_layers = 2  # LSTM的层数
    model2 = FeatureExtractorLSTM(input_channels, hidden_size_LSTM, num_layers)
    # 前向传播
    output = model2(input_data)
    print("lstm output shape:", output.shape)

    patch_size = 15  # 每个patch的大小
    hidden_size_T = 128  # Transformer的隐藏状态维度
    num_heads = 8  # 多头注意力的头数
    num_layers = 6  # Transformer的层数
    # 创建模型
    model3 = FeatureExtractorTransformer(input_channels, patch_size, hidden_size_T, num_heads, num_layers)

    # 前向传播
    output = model3(input_data)
    print("transformer output shape:", output.shape)
