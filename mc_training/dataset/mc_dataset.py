import torch
from torch.utils.data import Dataset
import numpy as np


class RollingDataset(Dataset):
    def __init__(self, data_loader, inst_id: str, window_size: int = 15, stride: int = 1, class_num=2):
        """
        PyTorch Dataset for rolling window training with configurable stride.
        Args:
            data_loader (DataLoader): DataLoader instance to load data.
            inst_id (str): Instrument ID to fetch data.
            window_size (int): Size of the rolling window (number of time steps).
            stride (int): Step size for moving the rolling window.
        """
        self.data_loader = data_loader
        self.inst_id = inst_id
        self.window_size = window_size
        self.stride = stride
        self.num_classes = class_num
        self.data = self.data_loader.get_data(inst_id)
        self.features_name = None

        if self.data.empty:
            raise ValueError(f"No data available for {inst_id}. Please check your DataLoader.")

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data for training: compute labels and features.
        """
        # Compute future 5-bar returns
        self.data['future_returns'] = (self.data['close'].shift(-5) - self.data['close']) / self.data['close']

        if self.num_classes == 2:
            # Assign labels based on future returns
            self.data['label'] = 0  # Default: price decreased or unchanged
            self.data.loc[self.data['future_returns'] > 0, 'label'] = 1  # Price increased

        elif self.num_classes == 3:
            # Assign labels based on future returns
            self.data['label'] = 2
            self.data.loc[self.data['future_returns'] >= 0.001, 'label'] = 1  # Price increased
            self.data.loc[self.data['future_returns'] < -0.001, 'label'] = 0  # Price decreased

        # Drop rows with NaN values (e.g., due to shift or pct_change)
        self.data.dropna(inplace=True)

        # Convert DataFrame to numpy arrays for faster indexing
        # 除去ts列，其余列作为特征
        try:
            self.features_name = self.data.drop(['ts', 'future_returns', 'label', 'datetime'], axis=1).columns
            self.features = self.data.drop(['ts', 'future_returns', 'label', 'datetime'], axis=1).values
        except:
            self.features_name = self.data.drop(['ts', 'future_returns', 'label'], axis=1).columns
            self.features = self.data.drop(['ts', 'future_returns', 'label'], axis=1).values
        self.labels = self.data['label'].values

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        # Adjust length based on stride
        return (len(self.features) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        """
        Returns a single sample (rolling window of features and corresponding label).
        Args:
            idx (int): Index of the sample.
        Returns:
            torch.Tensor: Rolling window of features (shape: [window_size, feature_dim]).
            torch.Tensor: Label for the last data point in the window.
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        x = self.features[start_idx:end_idx]
        y = self.labels[end_idx - 1]
        # 将 numpy.ndarray 转为 PyTorch 张量
        x = torch.tensor(x, dtype=torch.float32)
        # 添加单通道维度，确保形状为 [1, window_size, feature_dim]
        x = x.unsqueeze(0)

        return x, torch.tensor(y, dtype=torch.long)


# Example usage
if __name__ == '__main__':
    from data_loader import MCDataLoader
    from datetime import datetime

    # Initialize the DataLoader and load data
    dl = MCDataLoader()
    dl.load_data('BTC-USDT-SWAP', datetime(2024, 1, 2), datetime(2025, 1, 4))

    # Create a PyTorch Dataset with a stride of 5
    dataset = RollingDataset(data_loader=dl, inst_id='BTC-USDT-SWAP', window_size=30, stride=15)

    # Example: Fetch the first sample
    features, label = dataset[0]
    print("Features shape:", features.shape)  # Should be [window_size, feature_dim]
    print("Label:", label)

    # Check the total number of samples
    print("Total samples:", len(dataset))