"""
Simple test script to train and evaluate a model on the shipping fraud data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

try:
    import xgboost as xgb
    has_xgboost = True
except ImportError:
    has_xgboost = False
    print("XGBoost not installed. Skipping XGBoost model.")

# Load a sample of the data
print("Loading data...")
train_transaction = pd.read_csv('data/raw/train_transaction.csv', nrows=20000)

# Map transaction data to shipping data
print("Mapping features...")
feature_mapping = {
    'TransactionID': 'ShipmentID',
    'TransactionDT': 'ShipmentTimestamp',
    'TransactionAmt': 'ShipmentValue',
    'ProductCD': 'ShipmentType',
    'card1': 'SenderID',
    'card2': 'SenderAccountAge',
    'card3': 'SenderAccountType',
    'card4': 'PaymentMethod',
    'card5': 'PaymentAccountType',
    'card6': 'PaymentCardType',
    'addr1': 'SenderAddressID',
    'addr2': 'RecipientAddressID',
    'dist1': 'ShippingDistance',
    'dist2': 'BillingDistance',
    'P_emaildomain': 'SenderEmailDomain',
    'R_emaildomain': 'RecipientEmailDomain',
    'DeviceType': 'DeviceType',
    'DeviceInfo': 'DeviceInfo',
    'isFraud': 'isFraud'
}

# Select and rename columns
selected_columns = list(feature_mapping.keys())
available_columns = [col for col in selected_columns if col in train_transaction.columns]
shipping_data = train_transaction[available_columns].copy()
shipping_data.rename(columns=feature_mapping, inplace=True)

# Handle missing values
for col in shipping_data.columns:
    if shipping_data[col].dtype == 'object':
        shipping_data[col].fillna('unknown', inplace=True)
    else:
        shipping_data[col].fillna(-999, inplace=True)

# Split data
X = shipping_data.drop('isFraud', axis=1, errors='ignore')
y = shipping_data['isFraud'] if 'isFraud' in shipping_data.columns else None

# Convert categorical features to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.factorize(X[col])[0]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

if has_xgboost:
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42)

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    training_time = time.time() - start_time
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'training_time': training_time
    }
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# Create ensemble predictions
if len(results) > 1:
    print("\nCreating ensemble predictions...")
    ensemble_probs = np.zeros(len(y_test))
    
    for name, model in models.items():
        ensemble_probs += model.predict_proba(X_test)[:, 1] / len(models)
    
    # Convert ensemble probabilities to binary predictions
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    # Calculate ensemble metrics
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    ensemble_precision = precision_score(y_test, ensemble_preds)
    ensemble_recall = recall_score(y_test, ensemble_preds)
    ensemble_f1 = f1_score(y_test, ensemble_preds)
    ensemble_roc_auc = roc_auc_score(y_test, ensemble_probs)
    
    # Store ensemble results
    results['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'f1': ensemble_f1,
        'roc_auc': ensemble_roc_auc,
        'training_time': sum(results[model]['training_time'] for model in models)
    }
    
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Ensemble Precision: {ensemble_precision:.4f}")
    print(f"Ensemble Recall: {ensemble_recall:.4f}")
    print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
    print(f"Ensemble ROC AUC: {ensemble_roc_auc:.4f}")

# Print results table for README
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS FOR README")
print("="*80)
print("\n| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |")
print("|-------|----------|-----------|--------|----------|---------|")

for model_name, metrics in results.items():
    print(f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |")

print("\n" + "="*80) 