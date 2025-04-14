import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score


games_df = pd.read_csv('data/games_df.csv')

features_to_use = [
    'team1', 'team2',
    'total_close', 'total_2H', 'total_open'
]

# Assume games_df is already created and cleaned. For regression, we are predicting total_final.
df_model = games_df[features_to_use + ['total_final']].copy()

# One-hot encode the team names (if you haven't already)
df_model = pd.get_dummies(df_model, columns=['team1', 'team2'], drop_first=True)

# Define features and target
X = df_model.drop(columns=['total_final'])
y = df_model['total_final']

# Split the data into training and test sets (80/20 split, for example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features (scaling helps most models, especially neural nets and SVM/RF sometimes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =============================================================================
# 1. Assume Your DataFrame Already Exists as df_model
# =============================================================================
# df_model should contain at least the following columns:
#  - Features: 'total_1st', 'total_2nd', 'total_3rd', 'total_4th', 'total_close', 'total_2H', 'total_open', etc.
#  - Target: 'total_final'
# For example, you might have loaded it as:
# df_model = pd.read_csv("your_dataframe.csv")
#
# Define features (X) and target (y):
X = df_model.drop(columns=['total_final'])
y = df_model['total_final']

# =============================================================================
# 2. Split the Data (Train/Test)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Extract the original open betting line values from the test set for evaluating Over/Under.
open_lines_test = X_test['total_open'].values

# =============================================================================
# 3. Scale the Features and the Target
# =============================================================================
# Scale the feature variables.
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale the target (total points). This is done separately.
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# =============================================================================
# 4. Convert the Data to PyTorch Tensors
# =============================================================================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)  # shape: (n_train, n_features)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)    # shape: (n_test, n_features)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)    # shape: (n_train, 1)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)      # shape: (n_test, 1)

# =============================================================================
# 5. Define the DNN Model in PyTorch
# =============================================================================
class DNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, output_dim=1):
        super(DNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X_train_tensor.shape[1]
model = DNNRegressor(input_dim)

# =============================================================================
# 6. Define Loss Function and Optimizer
# =============================================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# 7. Train the DNN Model
# =============================================================================
num_epochs = 10_000  # Adjust as necessary
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)  # Predictions in scaled space.
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# =============================================================================
# 8. Evaluate the Model and Convert Predictions Back to Original Scale
# =============================================================================
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_tensor)  # shape: (n_test, 1)

# Convert scaled predictions back to the original total points scale.
preds_scaled_np = preds_scaled.detach().numpy()
preds_original = target_scaler.inverse_transform(preds_scaled_np)

# Compute RMSE on the original scale.
rmse_dnn = np.sqrt(mean_squared_error(y_test, preds_original))
print("DNN RMSE on Original Scale:", rmse_dnn)

# =============================================================================
# 9. Compute Over/Under Decisions and Accuracy
# =============================================================================
# Flatten predictions and true values.
preds_flat = preds_original.flatten()
y_test_flat = y_test.to_numpy().flatten()

# Create binary decisions:
# - Prediction is "Over" (1) if predicted total > open line, else "Under" (0).
pred_labels = (preds_flat > open_lines_test).astype(int)
true_labels = (y_test_flat > open_lines_test).astype(int)

dnn_overunder_accuracy = accuracy_score(true_labels, pred_labels)
print("DNN Over/Under Accuracy: {:.2%}".format(dnn_overunder_accuracy))

# =============================================================================
# 10. Print Sample Predictions with Over/Under Decisions
# =============================================================================
# print("\nSample DNN Predictions (Total Points in Original Scale) with Over/Under Decision:")
# for i in range(10):
#     decision = "Over" if preds_flat[i] > open_lines_test[i] else "Under"
#     print(f"Game {i+1}: Predicted Total = {preds_flat[i]:.2f}, Open = {open_lines_test[i]:.2f}, Decision = {decision}")
