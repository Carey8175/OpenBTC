import torch
import torch.nn as nn
import torch.nn.functional as F


class MCModel(nn.Module):
    def __init__(self, class_num=3, feature_num=10, window_size=30):
        super(MCModel, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(input_size=window_size, hidden_size=50, num_layers=2, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc_input_size = 50 * feature_num * 2  # LSTM输出的维度
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, class_num)
        )

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(3))
        # x: (batch_size, window_size, feature_num) -> (batch_size, feature_num, window_size)
        x = x.permute(0, 2, 1)
        # LSTM层，获取LSTM输出
        lstm_out, (hn, cn) = self.lstm(x, )

        # (batch_size, hidden_size * window_size)
        x = lstm_out.contiguous().view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = MCModel(class_num=3, feature_num=10, window_size=30)
    x = torch.randn(32, 30, 10)  # 假设batch_size=32, window_size=30, feature_num=10
    y = model(x)
    print(y.shape)
    print(y)