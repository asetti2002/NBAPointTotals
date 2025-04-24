import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

games_df = pd.read_csv('data/games_df.csv')

features_to_use = [
    'team1', 'team2',
    'total_close', 'total_open'
]

df_model = games_df[features_to_use + ['label']].copy()
df_model = pd.get_dummies(df_model, columns=['team1', 'team2'], drop_first=True)

X = df_model.drop(columns=['label']).values
y = df_model['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

train_ds = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_ds = TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)

# For epoch‐level accuracy, also keep full‐tensor versions:
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)


class DNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

model = DNNClassifier(X_train_scaled.shape[1])

# 6. Loss Function and Optimizer (with weight decay)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


num_epochs = 500

loss_history = []
train_acc_history = []
test_acc_history  = []

for epoch in range(1, num_epochs + 1):
    # ---- train ----
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(train_ds)
    loss_history.append(epoch_loss)

    # ---- train accuracy ----
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_tensor)
        train_preds  = (torch.sigmoid(train_logits) > 0.5).int().cpu().numpy()
    train_acc = accuracy_score(y_train, train_preds)
    train_acc_history.append(train_acc)

    # ---- test accuracy ----
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds  = (torch.sigmoid(test_logits) > 0.5).int().cpu().numpy()
    test_acc = accuracy_score(y_test, test_preds)
    test_acc_history.append(test_acc)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs} — Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")


print("\nFinal Classification Report:")
print(classification_report(y_test, test_preds, target_names=['Under','Over']))


os.makedirs('classification_plots/DNN', exist_ok=True)

# Loss
plt.figure()
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('classification_plots/DNN/loss.png')
plt.close()

# Training Accuracy
plt.figure()
plt.plot(train_acc_history)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('classification_plots/DNN/train_accuracy.png')
plt.close()

# Test Accuracy
plt.figure()
plt.plot(test_acc_history)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('classification_plots/DNN/test_accuracy.png')
plt.close()

print("Plots saved under classification_plots/DNN/")
