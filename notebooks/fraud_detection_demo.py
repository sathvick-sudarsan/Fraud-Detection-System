"""
Shipping Fraud Detection System Demo

This script demonstrates the usage of the Shipping Fraud Detection System, including:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Graph-based fraud ring detection
5. Making predictions on new data
"""

# Add the project root directory to the Python path
import sys
import os
sys.path.append(os.path.abspath('..'))

# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# Import project modules
from src.data.data_loader import ShippingFraudDataLoader
from src.features.feature_engineering import ShippingFraudFeatureEngineer
from src.models.fraud_detector import ShippingFraudDetector
from src.models.graph_analysis import ShippingFraudGraphAnalyzer

# Set up plotting
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

def main():
    print("Shipping Fraud Detection System Demo")
    print("====================================")
    
    # 1. Data Loading and Preprocessing
    print("\n1. Data Loading and Preprocessing")
    print("--------------------------------")
    
    # Initialize the data loader
    data_loader = ShippingFraudDataLoader(data_dir='../data/raw')
    
    # Load and preprocess the data
    train_data, test_data = data_loader.preprocess_data()
    
    # Display sample data
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print("\nSample data:")
    print(train_data.head())
    
    # Get feature information
    feature_info = data_loader.get_feature_names()
    
    print(f"\nTotal features: {len(feature_info['all'])}")
    print(f"Categorical features: {len(feature_info['categorical'])}")
    print(f"Numerical features: {len(feature_info['numerical'])}")
    
    # Display categorical features
    print("\nCategorical features:")
    print(feature_info['categorical'])
    
    # Check fraud distribution
    if 'isFraud' in train_data.columns:
        fraud_count = train_data['isFraud'].sum()
        total_count = len(train_data)
        fraud_percentage = (fraud_count / total_count) * 100
        
        print("\nFraud Distribution:")
        print(f"Fraud transactions: {fraud_count} ({fraud_percentage:.2f}%)")
        print(f"Non-fraud transactions: {total_count - fraud_count} ({100 - fraud_percentage:.2f}%)")
        
        # Plot fraud distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='isFraud', data=train_data)
        plt.title('Fraud Distribution')
        plt.xlabel('Is Fraud')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
        plt.savefig('../results/fraud_distribution.png')
        plt.close()
        
        print("Fraud distribution plot saved to '../results/fraud_distribution.png'")
    
    # 2. Feature Engineering
    print("\n2. Feature Engineering")
    print("---------------------")
    
    # Initialize the feature engineer
    feature_engineer = ShippingFraudFeatureEngineer()
    
    # Extract additional features
    train_data_enriched = feature_engineer.extract_all_features(train_data)
    test_data_enriched = feature_engineer.extract_all_features(test_data)
    
    # Display new features
    new_features = set(train_data_enriched.columns) - set(train_data.columns)
    print(f"Added {len(new_features)} new features: {sorted(new_features)}")
    
    # Display sample data with new features
    print("\nSample data with new features:")
    print(train_data_enriched[list(new_features)].head())
    
    # Analyze time-based patterns
    if 'ShipmentHour' in train_data_enriched.columns and 'ShipmentDayOfWeek' in train_data_enriched.columns:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        fraud_by_hour = train_data_enriched.groupby('ShipmentHour')['isFraud'].mean()
        sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values)
        plt.title('Fraud Rate by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Rate')
        
        plt.subplot(1, 2, 2)
        fraud_by_day = train_data_enriched.groupby('ShipmentDayOfWeek')['isFraud'].mean()
        sns.barplot(x=fraud_by_day.index, y=fraud_by_day.values)
        plt.title('Fraud Rate by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Fraud Rate')
        
        plt.tight_layout()
        plt.savefig('../results/time_patterns.png')
        plt.close()
        
        print("Time-based patterns plot saved to '../results/time_patterns.png'")
    
    # 3. Model Training and Evaluation
    print("\n3. Model Training and Evaluation")
    print("-------------------------------")
    
    # Split data into features and target
    X_train = train_data_enriched.drop('isFraud', axis=1) if 'isFraud' in train_data_enriched.columns else train_data_enriched
    y_train = train_data_enriched['isFraud'] if 'isFraud' in train_data_enriched.columns else None
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = data_loader.get_train_test_split()
    
    # Update with engineered features
    X_train = feature_engineer.extract_all_features(X_train)
    X_val = feature_engineer.extract_all_features(X_val)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_detector = ShippingFraudDetector(model_type='xgboost')
    xgb_results = xgb_detector.train(
        X_train, 
        y_train,
        categorical_features=[col for col in X_train.columns if X_train[col].dtype == 'object' or X_train[col].dtype == 'category'],
        numerical_features=[col for col in X_train.columns if col not in ['ShipmentID'] and X_train[col].dtype != 'object' and X_train[col].dtype != 'category'],
        hyperparameter_tuning=True,
        n_iter=5
    )
    
    print(f"XGBoost training completed in {xgb_results['training_time']:.2f} seconds")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_detector = ShippingFraudDetector(model_type='random_forest')
    rf_results = rf_detector.train(
        X_train, 
        y_train,
        categorical_features=[col for col in X_train.columns if X_train[col].dtype == 'object' or X_train[col].dtype == 'category'],
        numerical_features=[col for col in X_train.columns if col not in ['ShipmentID'] and X_train[col].dtype != 'object' and X_train[col].dtype != 'category'],
        hyperparameter_tuning=True,
        n_iter=5
    )
    
    print(f"Random Forest training completed in {rf_results['training_time']:.2f} seconds")
    
    # Evaluate XGBoost model
    print("\nEvaluating XGBoost model...")
    xgb_evaluation = xgb_detector.evaluate(X_val, y_val)
    
    print("XGBoost Model Evaluation:")
    print(f"Accuracy: {xgb_evaluation['accuracy']:.4f}")
    print(f"Precision: {xgb_evaluation['precision']:.4f}")
    print(f"Recall: {xgb_evaluation['recall']:.4f}")
    print(f"F1 Score: {xgb_evaluation['f1']:.4f}")
    print(f"ROC AUC: {xgb_evaluation['roc_auc']:.4f}")
    print(f"Average Precision: {xgb_evaluation['avg_precision']:.4f}")
    
    # Plot evaluation results
    xgb_detector.plot_evaluation(xgb_evaluation, output_dir='../results', prefix='XGBoost_')
    print("XGBoost evaluation plots saved to '../results/'")
    
    # Evaluate Random Forest model
    print("\nEvaluating Random Forest model...")
    rf_evaluation = rf_detector.evaluate(X_val, y_val)
    
    print("Random Forest Model Evaluation:")
    print(f"Accuracy: {rf_evaluation['accuracy']:.4f}")
    print(f"Precision: {rf_evaluation['precision']:.4f}")
    print(f"Recall: {rf_evaluation['recall']:.4f}")
    print(f"F1 Score: {rf_evaluation['f1']:.4f}")
    print(f"ROC AUC: {rf_evaluation['roc_auc']:.4f}")
    print(f"Average Precision: {rf_evaluation['avg_precision']:.4f}")
    
    # Plot evaluation results
    rf_detector.plot_evaluation(rf_evaluation, output_dir='../results', prefix='RandomForest_')
    print("Random Forest evaluation plots saved to '../results/'")
    
    # Compare model performance
    print("\nComparing model performance...")
    models = ['XGBoost', 'Random Forest']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
    xgb_scores = [xgb_evaluation[metric] for metric in metrics]
    rf_scores = [rf_evaluation[metric] for metric in metrics]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, xgb_scores, width, label='XGBoost')
    plt.bar(x + width/2, rf_scores, width, label='Random Forest')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    for i, v in enumerate(xgb_scores):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
        
    for i, v in enumerate(rf_scores):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png')
    plt.close()
    
    print("Model comparison plot saved to '../results/model_comparison.png'")
    
    # 4. Graph-based Fraud Ring Detection
    print("\n4. Graph-based Fraud Ring Detection")
    print("---------------------------------")
    
    # Initialize graph analyzer
    graph_analyzer = ShippingFraudGraphAnalyzer()
    
    # Build graph from training data
    print("Building graph from training data...")
    graph = graph_analyzer.build_graph(train_data_enriched)
    
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Detect fraud rings
    print("\nDetecting fraud rings...")
    fraud_rings = graph_analyzer.detect_fraud_rings(min_ring_size=3, max_ring_size=10)
    
    print(f"Detected {len(fraud_rings)} potential fraud rings")
    
    # Calculate node risk scores
    print("\nCalculating node risk scores...")
    risk_scores = graph_analyzer.calculate_node_risk_scores()
    
    # Identify suspicious nodes
    print("\nIdentifying suspicious nodes...")
    suspicious_nodes = graph_analyzer.identify_suspicious_nodes(risk_threshold=0.7)
    
    print(f"Identified {len(suspicious_nodes)} suspicious nodes")
    
    # Generate fraud ring report
    print("\nGenerating fraud ring report...")
    report = graph_analyzer.get_fraud_ring_report()
    
    print("Fraud Ring Report:")
    print(report.head(10))
    
    # Save report to CSV
    report.to_csv('../results/fraud_ring_report.csv', index=False)
    print("Fraud ring report saved to '../results/fraud_ring_report.csv'")
    
    # Visualize graph
    print("\nVisualizing graph...")
    graph_analyzer.visualize_graph(output_file='../results/fraud_graph.png', highlight_rings=True, highlight_suspicious=True)
    print("Graph visualization saved to '../results/fraud_graph.png'")
    
    # 5. Making Predictions on New Data
    print("\n5. Making Predictions on New Data")
    print("-------------------------------")
    
    # Create a sample shipment for prediction
    sample_shipment = {
        'ShipmentID': 1000001,
        'ShipmentTimestamp': 86400,  # 1 day after reference date
        'ShipmentValue': 500.0,
        'ShipmentType': 'W',
        'SenderID': 12345,
        'SenderAccountAge': 365,
        'SenderAccountType': 1,
        'PaymentMethod': 'visa',
        'PaymentAccountType': 1,
        'PaymentCardType': 'credit',
        'SenderAddressID': 54321,
        'RecipientAddressID': 67890,
        'ShippingDistance': 1000.0,
        'BillingDistance': 950.0,
        'SenderEmailDomain': 'gmail.com',
        'RecipientEmailDomain': 'company.com',
        'DeviceType': 'mobile',
        'DeviceInfo': 'iOS 15.0'
    }
    
    print("Sample shipment for prediction:")
    for key, value in sample_shipment.items():
        print(f"  {key}: {value}")
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_shipment])
    
    # Extract additional features
    sample_df_enriched = feature_engineer.extract_all_features(sample_df)
    
    # Make prediction
    fraud_probability = xgb_detector.predict_proba(sample_df_enriched)[0]
    fraud_prediction = int(fraud_probability >= 0.5)
    
    print(f"\nFraud Prediction: {fraud_prediction} (Probability: {fraud_probability:.4f})")
    
    # Determine risk level
    if fraud_probability < 0.2:
        risk_level = 'Low'
    elif fraud_probability < 0.5:
        risk_level = 'Medium'
    elif fraud_probability < 0.8:
        risk_level = 'High'
    else:
        risk_level = 'Very High'
    
    print(f"Risk Level: {risk_level}")
    
    # 6. Saving the Model
    print("\n6. Saving the Model")
    print("------------------")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save the best model (XGBoost in this case)
    model_path = xgb_detector.save_model(output_dir='../models')
    print(f"Model saved to {model_path}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Run the demo
    main() 