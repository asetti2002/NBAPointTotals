import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report



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



    os.makedirs('classification_plots/SVM', exist_ok=True)

    # 1. Get predicted probabilities and labels
    probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    preds = svm_model.predict(X_test_scaled)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve – SVM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('classification_plots/SVM/roc_curve.png')
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve – SVM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('classification_plots/SVM/precision_recall_curve.png')
    plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Under','Over'], yticklabels=['Under','Over'])
    plt.title('Confusion Matrix – SVM')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('classification_plots/SVM/confusion_matrix.png')
    plt.close()

    print("Saved SVM plots under classification_plots/SVM/")

if __name__ == '__main__':
    main()