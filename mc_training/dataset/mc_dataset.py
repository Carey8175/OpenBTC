import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mc_training.dataset.image_generator import ImageGenerator
from mc_training.core.config import Config


class RollingDataset(Dataset):
    def __init__(self, data_loader, inst_id: str, window_size: int = 15, stride: int = 1, class_num=2, load_img_local=False):
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
        self.img_data = []
        self.features_name = None
        self.load_img_local = load_img_local

        if self.data.empty:
            raise ValueError(f"No data available for {inst_id}. Please check your DataLoader.")

        self._prepare_data()
        self.get_images()

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

        self.labels = self.data['label'].values
        # Convert DataFrame to numpy arrays for faster indexing
        # 如果有'ts', 'future_returns', 'label', 'datetime' 列，除去这些列，其余列作为特征
        for key in ['ts', 'future_returns', 'label', 'datetime']:
            if key in self.data.columns:
                self.data.drop(key, axis=1, inplace=True)

        self.features_name = self.data.columns
        self.features = self.data.values

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        # Adjust length based on stride
        return (len(self.features) - self.window_size) // self.stride + 1

    def get_images(self):
        """
        Returns 4 channels feature images, macd, rsi, candles, vol  (rolling window of features and corresponding label).
        """
        if self.load_img_local and os.path.exists(os.path.join(Config.STATICS_PATH, 'img_data.pth')):
            self.img_data = torch.load(os.path.join(Config.STATICS_PATH, 'img_data.pth'))
            return

        for idx in tqdm(range(len(self)), desc='Generating Images, or maybe you can have a cup of coffee'):
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size
            window_data = self.features[start_idx:end_idx]
            window_data = pd.DataFrame(window_data, columns=self.features_name)
            self.img_data.append(ImageGenerator.gen(window_data))

        torch.save(self.img_data, os.path.join(Config.STATICS_PATH, 'img_data.pth'))

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

        return x, self.img_data[idx].to(torch.float32), torch.tensor(y, dtype=torch.long)


# Example usage
if __name__ == '__main__':
    from data_loader import MCDataLoader
    from datetime import datetime

    # Initialize the DataLoader and load data
    dl = MCDataLoader()
    dl.load_data('BTC-USDT-SWAP', datetime(2024, 12, 1), datetime(2025, 2, 18))

    # Create a PyTorch Dataset with a stride of 5
    dataset = RollingDataset(data_loader=dl, inst_id='BTC-USDT-SWAP', window_size=30, stride=1, load_img_local=True)

    # Example: Fetch the first sample
    import time
    time1 = time.time()
    features, img , label = dataset[0]
    print('Time cost of generating X' , time.time() - time1)
    print("Features shape:", features.shape)  # Should be [window_size, feature_dim]
    print("Label:", label)

    # Check the total number of samples
    print("Total samples:", len(dataset))
    df = pd.DataFrame(features.squeeze().cpu().numpy(), columns=dataset.features_name)
    # print(df.to_string())

    from PIL import Image
    print(f'Image shape: {img.shape}')
    # print(img.cpu().numpy())
    for i in range(img.shape[0]):
        img_ = img[i].cpu().numpy()
        img_ = Image.fromarray(img_)
        img_.show()