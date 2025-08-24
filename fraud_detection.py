"""
Credit Card Fraud Detection
Student: John Doe
Date: December 2024

This script trains ML models to detect credit card fraud
"""

# Import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load dataset and show basic information"""
    print("Loading dataset...")
    df = pd.read_csv('data/creditcard.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    print(f"\nFraud percentage: {df['Class'].value_counts()[1] / len(df) * 100:.2f}%")
    
    return df

def preprocess_data(df):
    """Preprocess the data - scaling and splitting"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Amount and Time columns
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def balance_data(X_train, y_train):
    """Balance the training data using SMOTE"""
    print("\nBalancing data with SMOTE...")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original shape: {X_train.shape}")
    print(f"Balanced shape: {X_train_balanced.shape}")
    
    return X_train_balanced, y_train_balanced

def train_models(X_train, y_train, X_test, y_test):
    """Train different ML models"""
    print("\nTraining models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return results

def evaluate_models(results, X_test, y_test):
    """Create evaluation plots"""
    print("\nCreating evaluation plots...")
    
    # Create figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    best_predictions = results[best_model_name]['predictions']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('figures/confusion_matrix.png')
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    
    for model_name, model_results in results.items():
        model = model_results['model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/roc_curves.png')
    plt.close()
    
    print("Plots saved in figures/ folder")
    
    return best_model_name, best_model

def save_model(model, scaler):
    """Save the trained model and scaler"""
    print("\nSaving model...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model saved to models/best_model.pkl")
    print("Scaler saved to models/scaler.pkl")

def main():
    """Main function to run the entire pipeline"""
    print("=" * 50)
    print("CREDIT CARD FRAUD DETECTION")
    print("=" * 50)
    
    # Load data
    df = load_and_explore_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Balance data
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    
    # Train models
    results = train_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Evaluate
    best_model_name, best_model = evaluate_models(results, X_test, y_test)
    
    # Save model
    save_model(best_model, scaler)
    
    print("\n" + "=" * 50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    # Print final summary
    print("\nFINAL RESULTS SUMMARY:")
    print("-" * 30)
    for name in results:
        print(f"\n{name}:")
        print(f"  F1-Score: {results[name]['f1_score']:.4f}")
    
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ All results saved in folders")
    print(f"✓ Ready for deployment!")

if __name__ == "__main__":
    main()