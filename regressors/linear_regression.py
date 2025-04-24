import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():

    games_df = pd.read_csv('data/games_df.csv')
    features_to_use = [
        'team1', 'team2',
        'total_close', 'total_2H', 'total_open'
    ]

    df_model = games_df[features_to_use + ['total_final']].copy()

    df_model = pd.get_dummies(df_model, columns=['team1', 'team2'], drop_first=True)

    X = df_model.drop(columns=['total_final'])
    y = df_model['total_final']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    lr_preds = lr_model.predict(X_test_scaled)

    mse_lr = mean_squared_error(y_test, lr_preds)
    r2_lr = r2_score(y_test, lr_preds)

    print("Linear Regression MSE:", mse_lr)
    print("Linear Regression R2 Score:", r2_lr)


    predicted_over_under_lr = ["Over" if pred > open_line else "Under" 
                            for pred, open_line in zip(lr_preds, X_test.iloc[:, X.columns.get_loc('total_open')])]



if __name__ == '__main__':
    main()