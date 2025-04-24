import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

games_df = pd.read_csv('data/games_df.csv')

features_to_use = [
    'team1', 'team2',
    'total_close', 'total_2H', 'total_open'
]

df_model = games_df[features_to_use + ['total_final']].copy()

# One-hot encode the team names (if you haven't already)
df_model = pd.get_dummies(df_model, columns=['team1', 'team2'], drop_first=True)

X = df_model.drop(columns=['total_final'])
y = df_model['total_final']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X = df_model.drop(columns=['total_final'])
y = df_model['total_final']

# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# Convert features and targets to tensors.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

X_train_tensor = X_train_tensor.unsqueeze(1)  # Shape: (n_train, 1, n_features)
X_test_tensor = X_test_tensor.unsqueeze(1)    # Shape: (n_test, 1, n_features)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


input_size = X_train_scaled.shape[1]  # Number of features.
hidden_size = 50
num_layers = 1
output_size = 1

model = LSTMRegressor(input_size, hidden_size, num_layers, output_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_tensor)
    test_loss = criterion(preds_scaled, y_test_tensor)
    print("Test Loss (scaled):", test_loss.item())
    
    # Convert predictions back to the original scale.
    preds_scaled_np = preds_scaled.detach().numpy()  # Shape: (n_test, 1)
    preds_original = target_scaler.inverse_transform(preds_scaled_np)
    
    # Compute RMSE on the original scale.
    rmse_original = np.sqrt(mean_squared_error(y_test, preds_original))
    print("RMSE on Original Scale:", rmse_original)

open_lines_test = X_test['total_open'].values
# flatten
preds_flat = preds_original.flatten()

y_test_flat = y_test.to_numpy().flatten()

# Create binary predictions: 1 for "Over" if predicted total > open line, else 0 for "Under".
pred_labels = (preds_flat > open_lines_test).astype(int)

# Create ground truth binary labels in the same way.
true_labels = (y_test_flat > open_lines_test).astype(int)

# Compute the accuracy of the over/under decisions.
over_under_accuracy = accuracy_score(true_labels, pred_labels)
print("Over/Under Accuracy: {:.2%}".format(over_under_accuracy))

os.makedirs('regression_plots/LSTM', exist_ok=True)

plt.figure()
plt.plot(loss_history)
plt.title('Training Loss (Scaled)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('regression_plots/LSTM/train_loss.png')
plt.close()
