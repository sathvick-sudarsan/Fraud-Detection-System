"""
Main Script for Shipping Fraud Detection

This script runs the entire shipping fraud detection pipeline:
1. Load and preprocess data
2. Extract additional features
3. Train and evaluate models
4. Analyze fraud patterns using graph analysis
5. Save the best model
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import project modules
from data.data_loader import ShippingFraudDataLoader
from features.feature_engineering import ShippingFraudFeatureEngineer
from models.fraud_detector import ShippingFraudDetector
from models.graph_analysis import ShippingFraudGraphAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(output_dir: str = 'results', 
                model_types: list = ['xgboost', 'random_forest'],
                hyperparameter_tuning: bool = True,
                graph_analysis: bool = True,
                save_model: bool = True):
    """
    Run the entire shipping fraud detection pipeline.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save results
    model_types : list
        List of model types to train and evaluate
    hyperparameter_tuning : bool
        Whether to perform hyperparameter tuning
    graph_analysis : bool
        Whether to perform graph analysis
    save_model : bool
        Whether to save the best model
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories
    models_dir = os.path.join(output_dir, 'models')
    plots_dir = os.path.join(output_dir, 'plots')
    reports_dir = os.path.join(output_dir, 'reports')
    
    for directory in [models_dir, plots_dir, reports_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data...")
    data_loader = ShippingFraudDataLoader()
    train_data, test_data = data_loader.preprocess_data()
    
    # Get feature information
    feature_info = data_loader.get_feature_names()
    categorical_features = feature_info['categorical']
    numerical_features = feature_info['numerical']
    
    logger.info(f"Loaded data: Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    logger.info(f"Features: {len(feature_info['all'])} total, {len(categorical_features)} categorical, {len(numerical_features)} numerical")
    
    # Step 2: Extract additional features
    logger.info("Step 2: Extracting additional features...")
    feature_engineer = ShippingFraudFeatureEngineer()
    
    train_data_enriched = feature_engineer.extract_all_features(train_data)
    test_data_enriched = feature_engineer.extract_all_features(test_data)
    
    # Get new feature information
    new_features = set(train_data_enriched.columns) - set(train_data.columns)
    logger.info(f"Added {len(new_features)} new features")
    
    # Split into features and target
    X_train = train_data_enriched.drop('isFraud', axis=1) if 'isFraud' in train_data_enriched.columns else train_data_enriched
    y_train = train_data_enriched['isFraud'] if 'isFraud' in train_data_enriched.columns else None
    
    # Update categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = data_loader.get_train_test_split()
    
    # Step 3: Train and evaluate models
    logger.info("Step 3: Training and evaluating models...")
    
    # Dictionary to store model results
    model_results = {}
    best_model = None
    best_score = 0
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        
        # Create and train model
        detector = ShippingFraudDetector(model_type=model_type)
        training_results = detector.train(
            X_train, 
            y_train,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            hyperparameter_tuning=hyperparameter_tuning,
            n_iter=10 if hyperparameter_tuning else 0
        )
        
        # Evaluate model
        evaluation_results = detector.evaluate(X_val, y_val)
        
        # Plot evaluation results
        detector.plot_evaluation(
            evaluation_results, 
            output_dir=plots_dir, 
            prefix=f"{model_type}_"
        )
        
        # Store results
        model_results[model_type] = {
            'training_time': training_results['training_time'],
            'evaluation': {
                'accuracy': evaluation_results['accuracy'],
                'precision': evaluation_results['precision'],
                'recall': evaluation_results['recall'],
                'f1': evaluation_results['f1'],
                'roc_auc': evaluation_results['roc_auc'],
                'avg_precision': evaluation_results['avg_precision']
            }
        }
        
        # Check if this is the best model
        if evaluation_results['roc_auc'] > best_score:
            best_score = evaluation_results['roc_auc']
            best_model = detector
    
    # Save model comparison report
    with open(os.path.join(reports_dir, 'model_comparison.json'), 'w') as f:
        json.dump(model_results, f, indent=4)
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_types)
    
    for i, model_type in enumerate(model_types):
        values = [model_results[model_type]['evaluation'][metric] for metric in metrics]
        plt.bar(x + i * width, values, width, label=model_type)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width * (len(model_types) - 1) / 2, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Step 4: Perform graph analysis if requested
    if graph_analysis and best_model is not None:
        logger.info("Step 4: Performing graph analysis...")
        
        # Build and analyze graph
        graph_analyzer = ShippingFraudGraphAnalyzer()
        graph_analyzer.build_graph(train_data_enriched)
        
        # Detect fraud rings
        fraud_rings = graph_analyzer.detect_fraud_rings()
        
        # Calculate risk scores
        risk_scores = graph_analyzer.calculate_node_risk_scores()
        
        # Identify suspicious nodes
        suspicious_nodes = graph_analyzer.identify_suspicious_nodes()
        
        # Generate fraud ring report
        report = graph_analyzer.get_fraud_ring_report()
        
        # Save report
        report.to_csv(os.path.join(reports_dir, 'fraud_ring_report.csv'), index=False)
        
        # Visualize graph
        graph_analyzer.visualize_graph(output_file=os.path.join(plots_dir, 'fraud_graph.png'))
        
        logger.info(f"Graph analysis complete. Detected {len(fraud_rings)} potential fraud rings.")
    
    # Step 5: Save the best model if requested
    if save_model and best_model is not None:
        logger.info("Step 5: Saving the best model...")
        
        model_path = best_model.save_model(output_dir=models_dir)
        logger.info(f"Best model ({best_model.model_type}) saved to {model_path}")
        
        # Save model info
        model_info = {
            'model_type': best_model.model_type,
            'model_path': model_path,
            'performance': {
                'accuracy': model_results[best_model.model_type]['evaluation']['accuracy'],
                'precision': model_results[best_model.model_type]['evaluation']['precision'],
                'recall': model_results[best_model.model_type]['evaluation']['recall'],
                'f1': model_results[best_model.model_type]['evaluation']['f1'],
                'roc_auc': model_results[best_model.model_type]['evaluation']['roc_auc'],
                'avg_precision': model_results[best_model.model_type]['evaluation']['avg_precision']
            },
            'training_time': model_results[best_model.model_type]['training_time'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(reports_dir, 'best_model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4)
    
    logger.info("Pipeline completed successfully!")
    
    return {
        'model_results': model_results,
        'best_model': best_model,
        'best_model_type': best_model.model_type if best_model else None,
        'best_score': best_score
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Shipping Fraud Detection pipeline')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--models', type=str, default='xgboost,random_forest', help='Comma-separated list of models to train')
    parser.add_argument('--no-tuning', action='store_true', help='Disable hyperparameter tuning')
    parser.add_argument('--no-graph', action='store_true', help='Disable graph analysis')
    parser.add_argument('--no-save', action='store_true', help='Disable model saving')
    
    args = parser.parse_args()
    
    # Parse model types
    model_types = args.models.split(',')
    
    # Run pipeline
    results = run_pipeline(
        output_dir=args.output_dir,
        model_types=model_types,
        hyperparameter_tuning=not args.no_tuning,
        graph_analysis=not args.no_graph,
        save_model=not args.no_save
    )
    
    # Print summary
    print("\nPipeline Results Summary:")
    print(f"Best model: {results['best_model_type']}")
    print(f"Best score (ROC AUC): {results['best_score']:.4f}")
    
    for model_type, result in results['model_results'].items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  Training time: {result['training_time']:.2f} seconds")
        print(f"  Accuracy: {result['evaluation']['accuracy']:.4f}")
        print(f"  Precision: {result['evaluation']['precision']:.4f}")
        print(f"  Recall: {result['evaluation']['recall']:.4f}")
        print(f"  F1 Score: {result['evaluation']['f1']:.4f}")
        print(f"  ROC AUC: {result['evaluation']['roc_auc']:.4f}")
        print(f"  Average Precision: {result['evaluation']['avg_precision']:.4f}") 