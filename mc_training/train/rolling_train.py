import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
from loguru import logger


class RollingTrainer:
    def __init__(self, dataset, class_num, unit_size=100, stride_day=50,  train_ratio=0.8, batch_size=32, device=None, epoch_per_phase=50, learning_rate=5e-5):
        """
        Rolling training framework.
        Args:
            dataset (Dataset): Instance of RollingDataset.
            model (nn.Module): PyTorch model, e.g., ResNet.
            unit_size (int): Number of samples in one training phase.
            train_ratio (float): Ratio of training samples in each phase (0.0-1.0).
            batch_size (int): Batch size for training and validation.
            device (str): Device to run the training ('cuda' or 'cpu').
        """
        self.dataset = dataset
        self.unit_size = unit_size  # days
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.stride_days = stride_day
        self.epoch_per_phase = epoch_per_phase
        self.class_num = class_num
        self.model = None
        self.learning_rate = learning_rate

    def split_dataset(self, start_idx, end_idx):
        """
        Split the dataset into training and testing based on start and end indices.
        """
        train_size = int(self.unit_size * self.train_ratio * self.day_data_num)
        test_size = self.unit_size * self.day_data_num - train_size

        train_indices = list(range(start_idx, start_idx + train_size))
        test_indices = list(range(start_idx + train_size, start_idx + train_size + test_size))

        train_subset = Subset(self.dataset, train_indices)
        test_subset = Subset(self.dataset, test_indices)

        # Log label distributions for train and test sets
        train_distribution = self.calculate_label_distribution(train_subset)
        test_distribution = self.calculate_label_distribution(test_subset)
        logger.info(f"Train Label Distribution: {train_distribution}")
        logger.info(f"Test Label Distribution: {test_distribution}")

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, train_distribution

    def train_one_phase(self, train_loader, val_loader, optimizer, criterion):
        """
        Train the model for multiple epochs in one phase with real-time tqdm updates.
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer for training.
            criterion: Loss function.
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
            for x, y in train_progress:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
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
                for x, y in val_progress:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
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

        return train_losses_per_epoch, best_f1

    def evaluate(self, test_loader):
        """
        Evaluate the model and calculate metrics.
        """
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                predictions = torch.argmax(outputs, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    @staticmethod
    def calculate_label_distribution(dataset):
        """
        Calculate label distribution for a given dataset.
        """
        labels = [dataset[i][1].item() for i in range(len(dataset))]
        return dict(Counter(labels))

    def rolling_train(self, total_days, start_date):
        """
        Perform rolling training and validation.
        """
        logger.info("Starting rolling training...")
        metrics_list = []

        total_samples = len(self.dataset)
        day_to_index = total_samples // total_days  # Samples per day
        self.day_data_num = day_to_index

        start_day = 0
        while start_day + self.unit_size <= total_days:
            # 重新初始化模型
            self.model = McModel(class_num=self.class_num, feature_num=self.dataset[0][0].shape[2], window_size=self.dataset[0][0].shape[1])
            self.model.to(self.device)

            start_idx = start_day * day_to_index
            end_idx = (start_day + self.unit_size) * day_to_index

            # Create DataLoader for training and testing
            train_loader, test_loader, train_labels_num = self.split_dataset(start_idx, end_idx)
            # define weight according to label distribution
            total_samples = sum(train_labels_num.values())
            weights = [total_samples / train_labels_num[label] for label in sorted(train_labels_num.keys())]
            # 转换为 PyTorch 张量
            class_weights = torch.tensor(weights, dtype=torch.float)

            # Define optimizer and loss function
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

            # Train and evaluate
            train_loss, best_val = self.train_one_phase(train_loader, test_loader, optimizer, criterion)
            # accuracy, precision, recall, f1 = self.evaluate(test_loader)
            accuracy, precision, recall, f1 = best_val

            metrics_list.append((train_loss[-1], accuracy, precision, recall, f1))
            logger.info(f"Train Loss: {train_loss[-1]}, Accuracy: {accuracy:.4f}, "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Move to the next rolling phase
            start_day += self.stride_days

        # Calculate average metrics
        avg_metrics = np.mean(metrics_list, axis=0)
        logger.info(f"Average Metrics - Train Loss: {avg_metrics[0]:.4f}, "
                    f"Accuracy: {avg_metrics[1]:.4f}, Precision: {avg_metrics[2]:.4f}, "
                    f"Recall: {avg_metrics[3]:.4f}, F1: {avg_metrics[4]:.4f}")
        logger.info("==================================================")
        # 选出最好的 f1 值进行评估
        best_metrics = max(metrics_list, key=lambda x: x[4])
        logger.info(f"Best Metrics - Accuracy: {best_metrics[1]:.4f}, avgPrecision: {best_metrics[2]:.4f}, "
                    f"avgRecall: {best_metrics[3]:.4f}, F1: {best_metrics[4]:.4f}")
        logger.info("==================================================")
        # 最差的 f1 值进行评估
        worst_metrics = min(metrics_list, key=lambda x: x[4])
        logger.info(f"Worst Metrics - Accuracy: {worst_metrics[1]:.4f}, avgPrecision: {worst_metrics[2]:.4f}, "
                    f"avgRecall: {worst_metrics[3]:.4f}, F1: {worst_metrics[4]:.4f}")

        return avg_metrics

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
    from mc_training.models.mc import MCModel as McModel
    from mc_training.dataset.data_loader import MCDataLoader
    from mc_training.dataset.mc_dataset import RollingDataset

    train_params = {
        "class_num": 2,
        "window_size": 30,
        "stride": 10,
        "train_ratio": 0.8,
        "batch_size": 32,
        "epoch_per_phase": 20,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "start_date": datetime(2024, 1, 2),
        "end_date": datetime(2025, 1, 4),
        "total_days": 360,
        "inst_id": 'BTC-USDT-SWAP',
        "add_indicators": True,
        "add_delta": False,
        "unit_size": 100,
        "day_stride": 50,
        "learning_rate": 5e-5
    }

    # Load data
    dl = MCDataLoader()
    dl.load_data(train_params['inst_id'], train_params['start_date'], train_params['end_date'], add_indicators=train_params['add_indicators'],
                 add_delta=train_params['add_delta'])
    # normalize data
    # dl.normalize_data('BTC-USDT-SWAP', method='min-max')
    dataset = RollingDataset(data_loader=dl,
                             inst_id=train_params['inst_id'],
                             window_size=train_params['window_size'],
                             stride=train_params['stride'],
                             class_num=train_params['class_num'])

    logger.info(f"shape of features: {dataset[0][0].shape}")

    # Initialize model and trainer
    trainer = RollingTrainer(dataset,
                             train_params['class_num'],
                             unit_size=train_params['unit_size'],
                             train_ratio=train_params['train_ratio'],
                             batch_size=train_params['batch_size'],
                             epoch_per_phase=train_params['epoch_per_phase'],
                             device=train_params['device'],
                             stride_day=train_params['day_stride'],
                             learning_rate=train_params['learning_rate'])

    trainer.set_random_seed(42)

    # Perform rolling training
    avg_metrics = trainer.rolling_train(total_days=train_params['total_days'],
                                        start_date=train_params['start_date'])

