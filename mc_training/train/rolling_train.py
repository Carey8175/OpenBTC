from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from loguru import logger


class RollingTrainer:
    def __init__(self, dataset, class_num, w_train=800, w_test=300, batch_size=32, device=None,
                 epoch_per_phase=50, learning_rate=5e-5):
        """
        Rolling training framework based on K-line count.
        """
        self.dataset = dataset
        self.w_train = w_train  # Number of K-lines for training
        self.w_test = w_test  # Number of K-lines for testing
        self.unit_size = w_train + w_test  # Total number of K-lines in each unit
        self.stride_model = w_test
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.stride_candles = self.w_test  # Sliding step in K-lines
        self.epoch_per_phase = epoch_per_phase
        self.class_num = class_num
        self.model = None
        self.learning_rate = learning_rate

    def split_dataset(self, start_idx, end_idx):
        """
        Split dataset into training and testing based on K-line index range.
        """
        train_size = self.w_train
        test_size = self.w_test

        train_indices = list(range(start_idx, start_idx + train_size))
        test_indices = list(range(start_idx + train_size, start_idx + train_size + test_size))

        train_subset = Subset(self.dataset, train_indices)
        test_subset = Subset(self.dataset, test_indices)

        train_distribution = self.calculate_label_distribution(train_subset)
        test_distribution = self.calculate_label_distribution(test_subset)
        logger.info(f"Train Label Distribution: {train_distribution}")
        logger.info(f"Test Label Distribution: {test_distribution}")

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, train_distribution

    def rolling_train(self):
        """
        Perform rolling training based on K-line count.
        """
        logger.info("Starting rolling training...")
        metrics_list = []
        # 自身的data添加列prediction
        self.dataset.data['prediction'] = np.nan

        total_samples = len(self.dataset)
        start_idx = 0

        while start_idx + self.unit_size <= total_samples:
            # Initialize model
            self.model = McModel(class_num=self.class_num, feature_num=self.dataset[0][0].shape[2],
                                 window_size=self.dataset[0][0].shape[1])
            self.model.to(self.device)

            end_idx = start_idx + self.unit_size
            train_loader, test_loader, train_labels_num = self.split_dataset(start_idx, end_idx)

            sub_samples = sum(train_labels_num.values())
            weights = [sub_samples / train_labels_num[label] for label in sorted(train_labels_num.keys())]
            class_weights = torch.tensor(weights, dtype=torch.float)

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

            # 会返回当前阶段最好的F1值模型所对应的指标
            train_loss, best_val = self.train_one_phase(train_loader, test_loader, optimizer, criterion, start_idx+self.w_train)
            accuracy, precision, recall, f1 = best_val

            metrics_list.append((train_loss[-1], accuracy, precision, recall, f1))
            logger.info(f"Train Loss: {train_loss[-1]}, Accuracy: {accuracy:.4f}, "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            start_idx += self.stride_candles  # Move window by stride size

        avg_metrics = np.mean(metrics_list, axis=0)
        logger.info(f"Average Metrics - Train Loss: {avg_metrics[0]:.4f}, "
                    f"Accuracy: {avg_metrics[1]:.4f}, Precision: {avg_metrics[2]:.4f}, "
                    f"Recall: {avg_metrics[3]:.4f}, F1: {avg_metrics[4]:.4f}")
        return avg_metrics

    @staticmethod
    def calculate_label_distribution(dataset):
        """ Calculate label distribution for a given dataset. """
        labels = [dataset[i][2].item() for i in range(len(dataset))]
        return dict(Counter(labels))

    def train_one_phase(self, train_loader, val_loader, optimizer, criterion, test_start_idx):
        """
        Train the model for multiple epochs in one phase with real-time tqdm updates.
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer for training.
            criterion: Loss function.
            test_start_idx: The starting index of the test data in the dataset.
        Returns:
            train_losses_per_epoch: List of average training losses for each epoch.
            val_losses_per_epoch: List of average validation losses for each epoch.
            val_accuracies_per_epoch: List of average validation accuracies for each epoch.
        """
        self.model.train()
        epochs = self.epoch_per_phase
        train_losses_per_epoch = []
        val_losses_per_epoch = []
        val_accuracies_per_epoch = []
        best_f1 = (0, 0, 0, 0)  # accuracy, avg_precision, avg_recall, f1

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            all_train_labels = []
            all_train_predictions = []

            train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False)
            for x, img, y in train_progress:
                x, img, y = x.to(self.device), img.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x, img)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Collect predictions and true labels for metrics calculation
                predictions = torch.argmax(outputs, dim=1)
                all_train_labels.extend(y.cpu().numpy())
                all_train_predictions.extend(predictions.cpu().numpy())

                # Calculate real-time accuracy and F1 score
                train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
                train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro')
                train_progress.set_postfix(loss=loss.item(), acc=train_accuracy, f1=train_f1)

            avg_train_loss = total_loss / len(train_loader)
            train_losses_per_epoch.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            all_labels = []
            all_predictions = []

            val_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}", leave=False)
            with torch.no_grad():
                for x, img, y in val_progress:
                    x, img, y = x.to(self.device), img.to(self.device), y.to(self.device)
                    outputs = self.model(x, img)
                    loss = criterion(outputs, y)
                    total_val_loss += loss.item()

                    # Collect predictions and true labels
                    predictions = torch.argmax(outputs, dim=1)
                    all_labels.extend(y.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())

                    # Calculate real-time accuracy
                    overall_accuracy = accuracy_score(all_labels, all_predictions)
                    val_progress.set_postfix(accuracy=overall_accuracy)

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses_per_epoch.append(avg_val_loss)

            # Calculate final validation accuracy for the epoch
            overall_accuracy = accuracy_score(all_labels, all_predictions)
            val_accuracies_per_epoch.append(overall_accuracy)

            # Calculate precision, recall, and F1 score
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')
            classification_report_str = classification_report(all_labels, all_predictions)

            # Calculate per-label accuracy
            label_counter = Counter(all_labels)
            label_correct = Counter([label for label, pred in zip(all_labels, all_predictions) if label == pred])
            label_accuracy = {label: label_correct[label] / label_counter[label] for label in label_counter}

            # Log metrics for the epoch
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Overall Validation Accuracy: {overall_accuracy:.4f}")
            logger.info(f"Per-label Validation Accuracy: {label_accuracy}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            logger.info(f"Classification Report:\n{classification_report_str}")

            # Track best F1 score
            if f1 > best_f1[3]:
                best_f1 = (overall_accuracy, precision, recall, f1)
                # 更新自身data的prediction
                indices = [idx * self.dataset.stride + self.dataset.window_size - 1 for idx in val_loader.dataset.indices]
                self.dataset.data.loc[indices, 'prediction'] = all_predictions


        return train_losses_per_epoch, best_f1

    @staticmethod
    def set_random_seed(seed: int):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value to use for random number generation.
        """
        # Python's built-in random module
        random.seed(seed)

        # NumPy random module
        np.random.seed(seed)

        # PyTorch random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Make cuDNN deterministic (optional, may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"Random seed set to: {seed}")


# Example usage
if __name__ == '__main__':
    # from mc_training.models.resnet18 import ResNetDualInput as McModel
    from mc_training.models.transmodel import TransformerEncoderDualInput as McModel
    from mc_training.dataset.data_loader import MCDataLoader
    from mc_training.dataset.mc_dataset import RollingDataset

    train_params = {
        "class_num": 2,
        "Wx": 30,  # 滑动窗口的蜡烛根数
        "S_sample": 1,
        "batch_size": 32,
        "epoch_per_phase": 20,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "start_date": datetime(2024, 12, 1),
        "end_date": datetime(2025, 2, 18),
        "total_days": 360,
        "inst_id": 'BTC-USDT-SWAP',
        "add_indicators": True,
        "add_delta": False,
        "W_train": 800,  # 单次训练的蜡烛根数
        "W_test": 300,  # 单次测试的蜡烛根数
        "learning_rate": 5e-5,
        "img_load_local": True
    }

    # Load data
    dl = MCDataLoader()
    dl.load_data(train_params['inst_id'], train_params['start_date'], train_params['end_date'],
                 add_indicators=train_params['add_indicators'],
                 add_delta=train_params['add_delta'])

    data_df = dl.data['BTC-USDT-SWAP'].copy()
    data_df = data_df[['open', 'high', 'low', 'close', 'ts']]
    data_df.reset_index(drop=True, inplace=True)

    dataset = RollingDataset(data_loader=dl,
                             inst_id=train_params['inst_id'],
                             window_size=train_params['Wx'],
                             stride=train_params['S_sample'],
                             class_num=train_params['class_num'],
                             load_img_local=train_params['img_load_local'])

    logger.info(f"shape of features: {dataset[0][0].shape}")
    logger.info(f"shape of images: {dataset[0][1].shape}")

    # Initialize model and trainer
    trainer = RollingTrainer(dataset,
                             train_params['class_num'],
                             w_train=train_params['W_train'],
                             w_test=train_params['W_test'],
                             batch_size=train_params['batch_size'],
                             epoch_per_phase=train_params['epoch_per_phase'],
                             device=train_params['device'],
                             learning_rate=train_params['learning_rate'])

    trainer.set_random_seed(42)

    # Perform rolling training
    avg_metrics = trainer.rolling_train()

    # for backtest
    from mc_training.core.backtest import Backtest

    labeled_data = trainer.dataset.data.dropna(subset=['prediction'])['prediction']
    # merge labeled_data with original data according to index
    labeled_data = data_df.merge(labeled_data, left_index=True, right_index=True)
    labeled_data.dropna(subset=['prediction'], inplace=True)

    bt = Backtest()
    bt.backtest(labeled_data)