import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from collections import Counter
from datetime import datetime
from loguru import logger
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


class RollingRFTrainer:
    def __init__(self, dataset, unit_size=100, stride_day=50, train_ratio=0.8, random_state=42):
        """
        滚动训练框架，适用于随机森林模型。
        Args:
            dataset: 数据集，格式为 (features, labels)。
            unit_size (int): 每个训练阶段包含的样本数。
            stride_day (int): 滑动步长。
            train_ratio (float): 每个阶段中训练集的比例。
            random_state (int): 随机种子。
        """
        self.dataset = dataset
        self.unit_size = unit_size
        self.train_ratio = train_ratio
        self.stride_days = stride_day
        self.random_state = random_state
        self.day_data_num = None
        self.batch_size = 32

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

        train_features, train_labels = self.extract_features_and_labels(train_subset)
        test_features, test_labels = self.extract_features_and_labels(test_subset)

        # Log label distributions for train and test sets
        train_distribution = dict(Counter(train_labels))
        test_distribution = dict(Counter(test_labels))
        logger.info(f"Train Label Distribution: {train_distribution}")
        logger.info(f"Test Label Distribution: {test_distribution}")

        return train_features, train_labels, test_features, test_labels

    @staticmethod
    def extract_features_and_labels(dataset):
        """
        Extract features and labels from a dataset.
        """
        features, labels = [], []
        for i in range(len(dataset)):
            x, img, y = dataset[i]
            # features.append(x.numpy().flatten())  # Flatten window * feature into a single vector
            features.append(x[0][-1].squeeze(0).numpy())  # Only keep the last candle
            labels.append(y.item())
        return np.array(features), np.array(labels)

    def train_one_phase(self, train_features, train_labels, val_features, val_labels):
        """
        单个训练阶段。
        """
        # 初始化随机森林模型
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=self.random_state,
                                       class_weight='balanced',
                                       max_depth=6)

        # 训练模型
        model.fit(train_features, train_labels)

        # 验证训练集指标
        train_predictions = model.predict(train_features)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_predictions, average='macro')
        train_classification_report_str = classification_report(train_labels, train_predictions)
        logger.info(f"======================= train ===========================")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
        logger.info(f"Train Classification Report:\n{train_classification_report_str}")


        # 预测验证集
        val_predictions = model.predict(val_features)

        # 计算验证集指标
        accuracy = accuracy_score(val_labels, val_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='macro')
        classification_report_str = classification_report(val_labels, val_predictions)

        # Log metrics
        logger.info(f"======================= validation ===========================")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        logger.info(f"Classification Report:\n{classification_report_str}")


        feature_importances = model.feature_importances_
        feature_names = dataset.features_name  # Use original feature names

        # sort the features by importance
        sorted_idx = np.argsort(feature_importances)
        feature_importances = feature_importances[sorted_idx]
        feature_names = feature_names[sorted_idx]

        plt.figure(figsize=(12, 48))
        plt.barh(range(len(feature_importances)), feature_importances, tick_label=feature_names)
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.savefig("feature_importances.png")

        return model, accuracy, precision, recall, f1

    def rolling_train(self, total_days, start_date):
        """
        执行滚动训练。
        """
        logger.info("Starting rolling training...")
        metrics_list = []

        total_samples = len(self.dataset)
        day_to_index = total_samples // total_days  # 每天对应的样本数
        self.day_data_num = day_to_index

        start_day = 0
        best_model = None
        best_f1 = 0

        while start_day + self.unit_size <= total_days:
            start_idx = start_day * day_to_index
            end_idx = (start_day + self.unit_size) * day_to_index

            # 获取当前阶段的数据
            train_features, train_labels, test_features, test_labels = self.split_dataset(start_idx, end_idx)

            # 训练单个阶段
            model, accuracy, precision, recall, f1 = self.train_one_phase(train_features, train_labels, test_features, test_labels)

            metrics_list.append((accuracy, precision, recall, f1))

            if f1 > best_f1:
                best_f1 = f1
                best_model = model

            logger.info(f"Phase Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # 滑动到下一个阶段
            start_day += self.stride_days

        # 计算平均指标
        avg_metrics = np.mean(metrics_list, axis=0)
        logger.info(f"Average Metrics - Accuracy: {avg_metrics[0]:.4f}, Precision: {avg_metrics[1]:.4f}, Recall: {avg_metrics[2]:.4f}, F1: {avg_metrics[3]:.4f}")
        # best info
        best_one = max(metrics_list, key=lambda x: x[3])
        logger.info(f"Best - Accuracy: {best_one[0]:.4f}, Precision: {best_one[1]:.4f}, Recall: {best_one[2]:.4f}, F1: {best_one[3]:.4f}")
        # worst one
        worst_one = min(metrics_list, key=lambda x: x[3])
        logger.info(f"Worst - Accuracy: {worst_one[0]:.4f}, Precision: {worst_one[1]:.4f}, Recall: {worst_one[2]:.4f}, F1: {worst_one[3]:.4f}")


        return best_model, avg_metrics

    @staticmethod
    def set_random_seed(seed: int):
        """
        设置随机种子以保证结果可复现。
        """
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")


# Example usage
if __name__ == '__main__':
    from mc_training.dataset.mc_dataset import RollingDataset
    from mc_training.dataset.data_loader import MCDataLoader

    dl = MCDataLoader()
    dl.load_data('BTC-USDT-SWAP', datetime(2024, 12, 1), datetime(2025, 2, 18),
                 add_delta=True,
                 add_indicators=True)
    dataset = RollingDataset(dl, inst_id="BTC-USDT-SWAP", window_size=30, stride=1, class_num=2, load_img_local=True)

    trainer = RollingRFTrainer(dataset, unit_size=int(len(dataset) / 24), stride_day=50, train_ratio=0.6)
    trainer.set_random_seed(42)

    best_model, avg_metrics = trainer.rolling_train(total_days=int(len(dataset) / 24), start_date=datetime(2024, 11, 24))
    # logger.info(f" Model Metrics: {avg_metrics}")