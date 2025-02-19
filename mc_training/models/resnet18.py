import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from mc_training.models.layers.column_normalization import PerFeatureNormalization


class ResNetDualInput(nn.Module):
    def __init__(self, class_num=3, window_size=30, feature_num=7, intermediate_dim=128):
        """
        Modified ResNet model that processes x and img separately through ResNet,
        then concatenates their feature vectors for final classification.
        """
        super(ResNetDualInput, self).__init__()

        # Feature extraction ResNets
        self.resnet_x = resnet18(weights=None)
        self.resnet_x.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_x.fc = nn.Identity()  # Remove classification head, keep feature extraction
        self.resnet_img = resnet18(weights=None)
        self.resnet_img.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_img.fc = nn.Identity()

        # Fully connected layer for final classification
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),  # Concatenate features from both ResNets (512 + 512)
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        self.pfn = PerFeatureNormalization(method="min_max")

    def forward(self, x, img):
        # Extract features separately
        x = self.pfn(x)

        feat_x = self.resnet_x(x)  # [batch_size, 512]
        feat_img = self.resnet_img(img)  # [batch_size, 512]

        # Concatenate feature vectors
        combined_feat = torch.cat([feat_x, feat_img], dim=1)  # [batch_size, 1024]

        # Classification
        output = self.fc(combined_feat)  # [batch_size, class_num]

        return output