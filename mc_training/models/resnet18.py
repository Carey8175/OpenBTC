import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from mc_training.models.layers.column_normalization import PerFeatureNormalization
from mc_training.models.layers.feature_sort import PositionalSortingLayer


class ResNet18(nn.Module):
    def __init__(self, class_num=3, window_size=30, feature_num=7, intermediate_dim=128):
        """
        Modified ResNet16 model with a fully connected layer before ResNet.
        Args:
            num_classes: Number of output classes.
            input_height: Height of the input data.
            input_width: Width of the input data.
            intermediate_dim: Dimension of the output from the fully connected layer.
        """
        super(ResNet18, self).__init__()
        # Normalize each sample independently
        self.norm = PerFeatureNormalization(method="min_max")
        # Sort features based on importance
        self.sort = PositionalSortingLayer(feature_dim=feature_num)

        # Fully connected layer before ResNet
        input_dim = window_size * feature_num  # Flattened input dimension
        self.fc = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1 * window_size * feature_num),  # Output to match [batch_size, 1, 30, 30]
            nn.ReLU()
        )

        # Reshape dimensions for ResNet
        self.resnet_input_channels = 1  # Single channel input
        self.resnet_input_height = window_size
        self.resnet_input_width = feature_num

        # Modified ResNet18
        self.resnet = resnet50(num_classes=class_num)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=self.resnet_input_channels,  # Single channel
            out_channels=64,  # Matches original ResNet18
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

    def forward(self, x):
        # Normalize each sample independently
        x = self.norm(x)

        # Flatten input for the fully connected layer
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, input_dim]

        # Fully connected layer
        x = self.fc(x)

        # Reshape to match ResNet input dimensions
        x = x.view(batch_size, self.resnet_input_channels, self.resnet_input_height, self.resnet_input_width)

        # Sort features based on importance
        # x, sorted_index = self.sort(x)

        # Pass through ResNet
        return self.resnet(x)


# Example usage
if __name__ == '__main__':
    model = ResNet18(num_classes=3, input_height=30, input_width=5, intermediate_dim=128)

    # Example input: batch_size=32, single channel, 30x5
    x = torch.randn(32, 1, 30, 5)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [32, 3] for 3 classes