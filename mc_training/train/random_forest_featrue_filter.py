import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime

from mc_training.dataset.mc_dataset import RollingDataset
from mc_training.dataset.data_loader import MCDataLoader

# Load data
data_loader = MCDataLoader()
data_loader.load_data("BTC-USDT-SWAP",
                      datetime(2024, 1, 2),
                      datetime(2025, 1, 4),
                      add_indicators=True,
                      add_delta=False)
dataset = RollingDataset(data_loader=data_loader, inst_id="BTC-USDT-SWAP", window_size=30, stride=15)

# Extract features and labels from the dataset
features = []
labels = []

# Prepare rolling window features
for i in range(len(dataset) - 1):
    window_features, label = dataset[i]
    features.append(window_features[0][-1].squeeze(0).numpy())  # Remove channel dimension
    labels.append(label.item())

features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)  # Flatten window features

# Predict on the test set
y_pred = rf.predict(X_train)

# Print classification report
print("Classification Report:\n", classification_report(y_train, y_pred))

# Count the occurrences of each label in predictions
unique_labels, counts = np.unique(y_pred, return_counts=True)
label_distribution = dict(zip(unique_labels, counts))

# Find the label used the most
most_used_label = max(label_distribution, key=label_distribution.get)
print(f"Most used label: {most_used_label} ({label_distribution[most_used_label]} occurrences)")

# Display feature importances
feature_importances = rf.feature_importances_
feature_names = dataset.features_name  # Use original feature names

# sort the features by importance
sorted_idx = np.argsort(feature_importances)
feature_importances = feature_importances[sorted_idx]
feature_names = feature_names[sorted_idx]

plt.figure(figsize=(12, 12))
plt.barh(range(len(feature_importances)), feature_importances, tick_label=feature_names)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.savefig("feature_importances.png")
print(feature_names[60:])