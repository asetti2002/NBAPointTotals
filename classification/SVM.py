import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def main():
    games_df = pd.read_csv('data/games_df.csv')

    features_to_use = ['team1', 'team2', 'total_close', 'total_2H', 'total_open']
    df_model = games_df[features_to_use + ['label']].copy()

    # One-hot encode the categorical team columns
    df_model = pd.get_dummies(df_model, columns=['team1', 'team2'], drop_first=True)

    # Define features X and target y
    X = df_model.drop(columns=['label'])
    y = df_model['label']

    # Split the dataset into training and test sets (e.g., 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features. (We scale everything for models like SVM and MLP.)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_preds = svm_model.predict(X_test_scaled)
    print("\nSVM Accuracy:", accuracy_score(y_test, svm_preds))
    print("Classification Report (SVM):")
    print(classification_report(y_test, svm_preds))


if __name__ == '__main__':
    main()