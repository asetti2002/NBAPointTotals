import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

games_df = pd.read_csv('data/games_df.csv')

features_to_use = [
    'team1', 'team2',
    'total_close', 'total_2H', 'total_open', 'total_final'
]
df_model = games_df[features_to_use].copy()
df_model = pd.get_dummies(df_model, columns=['team1', 'team2'], drop_first=True)

X = df_model.drop(columns=['total_final']).values
y = df_model['total_final'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# locate the index of 'total_open' in the encoded feature set
open_idx = list(df_model.drop(columns=['total_final']).columns).index('total_open')
open_lines_test = X_test[:, open_idx]

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled  = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1,1))
y_test_scaled  = target_scaler.transform(y_test.reshape(-1,1))

Xtr = torch.tensor(X_train_scaled, dtype=torch.float32)
Xte = torch.tensor(X_test_scaled,  dtype=torch.float32)
ytr = torch.tensor(y_train_scaled, dtype=torch.float32)

class DNNRegressor(nn.Module):
    def __init__(self, in_dim, h1=64, h2=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = DNNRegressor(Xtr.shape[1])


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10_000

loss_history = []
rmse_history = []
ou_acc_history = []

for epoch in range(1, num_epochs+1):
    model.train()
    optimizer.zero_grad()
    out_scaled = model(Xtr)
    loss = criterion(out_scaled, ytr)
    loss.backward()
    optimizer.step()

    # record training loss
    loss_history.append(loss.item())

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        preds_scaled = model(Xte).cpu().numpy().reshape(-1,1)
    preds_orig = target_scaler.inverse_transform(preds_scaled).flatten()

    # compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, preds_orig))
    rmse_history.append(rmse)

    # compute Over/Under accuracy
    ou_preds = (preds_orig > open_lines_test).astype(int)
    true_labels = (y_test > open_lines_test).astype(int)
    ou_acc = accuracy_score(true_labels, ou_preds)
    ou_acc_history.append(ou_acc)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{num_epochs} — Loss: {loss.item():.4f}, "
              f"RMSE: {rmse:.2f}, OU Acc: {ou_acc:.2%}")
        
print("\nFinal RMSE:", rmse_history[-1])
print("Final Over/Under Accuracy:", ou_acc_history[-1])


os.makedirs('regression_plots/DNN', exist_ok=True)

# 2. Plot training loss
plt.figure()
plt.plot(loss_history)
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('regression_plots/DNN/loss.png')
plt.close()

# 3. Plot test RMSE
plt.figure()
plt.plot(rmse_history)
plt.title('Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.savefig('regression_plots/DNN/rmse.png')
plt.close()

# 4. Plot test Over/Under accuracy
plt.figure()
plt.plot(ou_acc_history)
plt.title('Test Over/Under Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('regression_plots/DNN/overunder_accuracy.png')
plt.close()

print("Saved regression plots under regression_plots/DNN/")

