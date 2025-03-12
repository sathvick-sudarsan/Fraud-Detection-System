#!/usr/bin/env python
"""
Shipping Fraud Detection System

This script runs the entire shipping fraud detection pipeline by calling the main module.
It serves as the entry point for the application as mentioned in the README.md.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Import the main pipeline function
from src.main import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fraud_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Parse command line arguments and run the fraud detection pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Shipping Fraud Detection System')
    parser.add_argument('--output-dir', type=str, default='results', 
                        help='Directory to save results (default: results)')
    parser.add_argument('--models', type=str, default='xgboost,random_forest', 
                        help='Comma-separated list of models to train (default: xgboost,random_forest)')
    parser.add_argument('--no-tuning', action='store_true', 
                        help='Disable hyperparameter tuning')
    parser.add_argument('--no-graph', action='store_true', 
                        help='Disable graph analysis')
    parser.add_argument('--no-save', action='store_true', 
                        help='Disable model saving')
    parser.add_argument('--data-dir', type=str, default='data/raw', 
                        help='Directory containing raw data (default: data/raw)')
    
    args = parser.parse_args()
    
    # Parse model types
    model_types = args.models.split(',')
    
    logger.info("Starting Shipping Fraud Detection System")
    logger.info(f"Models to train: {model_types}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Hyperparameter tuning: {not args.no_tuning}")
    logger.info(f"Graph analysis: {not args.no_graph}")
    logger.info(f"Model saving: {not args.no_save}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    # Run the pipeline
    try:
        results = run_pipeline(
            output_dir=args.output_dir,
            model_types=model_types,
            hyperparameter_tuning=not args.no_tuning,
            graph_analysis=not args.no_graph,
            save_model=not args.no_save
        )
        
        # Print summary
        print("\n" + "="*50)
        print("SHIPPING FRAUD DETECTION SYSTEM - RESULTS SUMMARY")
        print("="*50)
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
        
        print("\n" + "="*50)
        print(f"Results saved to: {args.output_dir}")
        print("="*50)
        
        logger.info("Shipping Fraud Detection System completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
