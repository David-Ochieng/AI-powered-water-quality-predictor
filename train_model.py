import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_model():
    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv('water_quality.csv')
    except FileNotFoundError:
        print("Error: water_quality.csv not found. Please run generate_data.py first.")
        return

    # 2. Data Cleaning & Preprocessing
    print("Preprocessing data...")
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 3. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Train Model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save Model and Scaler
    print("Saving model and artifacts...")
    joblib.dump(model, 'water_quality_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    print("Model saved as water_quality_model.pkl")
    print("Scaler saved as scaler.pkl")
    print("Imputer saved as imputer.pkl")

if __name__ == "__main__":
    train_model()
