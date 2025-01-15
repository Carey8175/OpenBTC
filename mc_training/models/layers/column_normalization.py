import torch
import torch.nn as nn


class PerFeatureNormalization(nn.Module):
    def __init__(self, method: str = "z_score", eps: float = 1e-8):
        """
        A custom layer to normalize each sample's feature independently across the temporal dimension,
        supporting multi-channel input.

        Args:
            method (str): Normalization method, either "z_score" or "min_max".
            eps (float): A small value to prevent division by zero.
        """
        super(PerFeatureNormalization, self).__init__()
        self.method = method
        self.eps = eps

    def forward(self, x):
        """
        Normalize each feature independently for every sample and channel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps, features).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # Ensure input shape: batch_size x channels x time_steps x features
        assert x.dim() == 4, "Input tensor must have 4 dimensions (batch_size, channels, time_steps, features)"

        batch_size, channels, time_steps, num_features = x.shape

        if self.method == "z_score":
            # Compute mean and std for each channel-feature pair across the temporal dimension
            mean = x.mean(dim=2, keepdim=True)  # Shape: (batch_size, channels, 1, features)
            std = x.std(dim=2, keepdim=True, unbiased=False) + self.eps  # Shape: (batch_size, channels, 1, features)
            x_normalized = (x - mean) / std

        elif self.method == "min_max":
            # Compute min and max for each channel-feature pair across the temporal dimension
            min_val = x.min(dim=2, keepdim=True).values  # Shape: (batch_size, channels, 1, features)
            max_val = x.max(dim=2, keepdim=True).values  # Shape: (batch_size, channels, 1, features)
            x_normalized = (x - min_val) / (max_val - min_val + self.eps)

        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        return x_normalized