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
# 1. Assume You Have a DataFrame Already (df_model)
# =============================================================================
# For example, df_model might have the following columns:
# ['total_1st', 'total_2nd', 'total_3rd', 'total_4th', 'total_close', 'total_2H', 'total_open', 'total_final']
# where 'total_final' is our target variable.

# If you haven't already, load or define df_model.
# For this example, I'll assume df_model is already defined.

# =============================================================================
# 2. Split the Data (Same as in the classification pipeline)
# =============================================================================
# Define features and target.
X = df_model.drop(columns=['total_final'])
y = df_model['total_final']

# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# 3. Scale the Features and Target
# =============================================================================
# Scale features.
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale the target (total points) separately.
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# =============================================================================
# 4. Convert to PyTorch Tensors and Reshape for LSTM
# =============================================================================
# Convert features and targets to tensors.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# LSTM expects 3D input: (batch_size, seq_length, n_features).
# Since each game is independent, we use a sequence length of 1.
X_train_tensor = X_train_tensor.unsqueeze(1)  # Shape: (n_train, 1, n_features)
X_test_tensor = X_test_tensor.unsqueeze(1)    # Shape: (n_test, 1, n_features)

# =============================================================================
# 5. Define the LSTM Regressor Model in PyTorch
# =============================================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take output from the last time step (shape: batch_size x hidden_size)
        out = self.fc(out)   # Final output (shape: batch_size x output_size)
        return out

# Model hyperparameters.
input_size = X_train_scaled.shape[1]  # Number of features.
hidden_size = 50
num_layers = 1
output_size = 1

model = LSTMRegressor(input_size, hidden_size, num_layers, output_size)

# =============================================================================
# 6. Define Loss Function and Optimizer
# =============================================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# 7. Training Loop
# =============================================================================
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)           # Forward pass: predictions (scaled).
    loss = criterion(outputs, y_train_tensor)   # Loss computed on scaled targets.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# =============================================================================
# 8. Evaluation and Inverse Transform Predictions
# =============================================================================
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
    
    # Print a few sample predictions.
#     print("Sample LSTM Predictions (Total Points, original scale):")
#     print(preds_original)



# import numpy as np
# from sklearn.metrics import accuracy_score

# Assume that after training your LSTM model you have:
# - preds_original: a NumPy array of shape (n_test, 1) holding the LSTM's predicted total points (in the original scale)
# - y_test: a pandas Series of the true total points (unscaled) for the test set.
# - open_lines_test: the original open betting line values for the test set, e.g., extracted as:
#        open_lines_test = X_test['total_open'].values
#
# Ensure that these are aligned. For instance, if you split your DataFrame earlier,
# make sure that open_lines_test was extracted from the test set (X_test) rather than the full DataFrame.
open_lines_test = X_test['total_open'].values
# Flatten predictions (preds_original is shape (n_test, 1)) into a 1D array.
preds_flat = preds_original.flatten()
# Convert y_test to a NumPy 1D array, if it is a pandas Series.
y_test_flat = y_test.to_numpy().flatten()

# Create binary predictions: 1 for "Over" if predicted total > open line, else 0 for "Under".
pred_labels = (preds_flat > open_lines_test).astype(int)

# Create ground truth binary labels in the same way.
true_labels = (y_test_flat > open_lines_test).astype(int)

# Compute the accuracy of the over/under decisions.
over_under_accuracy = accuracy_score(true_labels, pred_labels)
print("Over/Under Accuracy: {:.2%}".format(over_under_accuracy))