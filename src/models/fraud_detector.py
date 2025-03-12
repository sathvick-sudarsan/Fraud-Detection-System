"""
Machine Learning Models for Shipping Fraud Detection

This module provides classes for training and evaluating machine learning models
to detect fraudulent shipping activities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
import os
from datetime import datetime
import time

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShippingFraudDetector:
    """
    Machine learning model for detecting fraudulent shipping activities.
    """
    
    def __init__(self, 
                model_type: str = 'xgboost', 
                model_params: Optional[Dict[str, Any]] = None,
                class_weight: Union[str, Dict[int, float]] = 'balanced',
                random_state: int = 42):
        """
        Initialize the fraud detector.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression')
        model_params : Optional[Dict[str, Any]]
            Parameters for the model
        class_weight : Union[str, Dict[int, float]]
            Class weights for handling imbalanced data
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model = None
        self.feature_importances = None
        self.preprocessor = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.training_time = None
        self.model_path = None
    
    def _create_model(self) -> BaseEstimator:
        """
        Create a model based on the specified type.
        
        Returns:
        --------
        BaseEstimator
            Scikit-learn compatible model
        """
        if self.model_type == 'xgboost':
            # Default parameters for XGBoost
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': self.random_state,
                'scale_pos_weight': 1  # Will be set based on class distribution
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            return XGBClassifier(**params)
        
        elif self.model_type == 'random_forest':
            # Default parameters for Random Forest
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': self.class_weight,
                'random_state': self.random_state
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            return RandomForestClassifier(**params)
        
        elif self.model_type == 'gradient_boosting':
            # Default parameters for Gradient Boosting
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 0.8,
                'max_features': 'sqrt',
                'random_state': self.random_state
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            return GradientBoostingClassifier(**params)
        
        elif self.model_type == 'logistic_regression':
            # Default parameters for Logistic Regression
            params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'liblinear',
                'class_weight': self.class_weight,
                'random_state': self.random_state,
                'max_iter': 1000
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            return LogisticRegression(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_preprocessor(self, categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
        """
        Create a preprocessor for the data.
        
        Parameters:
        -----------
        categorical_features : List[str]
            List of categorical feature names
        numerical_features : List[str]
            List of numerical feature names
            
        Returns:
        --------
        ColumnTransformer
            Scikit-learn preprocessor
        """
        # Create transformers for categorical and numerical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             categorical_features: Optional[List[str]] = None,
             numerical_features: Optional[List[str]] = None,
             hyperparameter_tuning: bool = False,
             n_iter: int = 10,
             cv: int = 5) -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        categorical_features : Optional[List[str]]
            List of categorical feature names
        numerical_features : Optional[List[str]]
            List of numerical feature names
        hyperparameter_tuning : bool
            Whether to perform hyperparameter tuning
        n_iter : int
            Number of iterations for randomized search
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        logger.info(f"Training {self.model_type} model...")
        start_time = time.time()
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Identify categorical and numerical features if not provided
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numerical_features is None:
            numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        logger.info(f"Categorical features: {len(categorical_features)}, Numerical features: {len(numerical_features)}")
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(categorical_features, numerical_features)
        
        # Create model
        base_model = self._create_model()
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', base_model)
        ])
        
        # Perform hyperparameter tuning if requested
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            
            # Define parameter grid based on model type
            if self.model_type == 'xgboost':
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__min_child_weight': [1, 3, 5],
                    'classifier__subsample': [0.6, 0.8, 1.0],
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [5, 10, 15, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['sqrt', 'log2', None]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__subsample': [0.6, 0.8, 1.0],
                    'classifier__max_features': ['sqrt', 'log2', None]
                }
            elif self.model_type == 'logistic_regression':
                param_grid = {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            else:
                param_grid = {}
            
            # Create cross-validation strategy
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            
            # Create randomized search
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='roc_auc',
                cv=cv_strategy,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Get best model
            self.model = search.best_estimator_
            
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            # Fit the pipeline
            self.model = pipeline
            self.model.fit(X_train, y_train)
        
        # Calculate feature importances if available
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            
            # Get feature names from one-hot encoding
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = ohe.get_feature_names_out(categorical_features).tolist()
            
            # Combine with numerical features
            all_features = numerical_features + cat_features
            
            # Get feature importances
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Create a dictionary of feature importances
            self.feature_importances = dict(zip(all_features, importances))
            
            # Sort by importance
            self.feature_importances = {k: v for k, v in sorted(
                self.feature_importances.items(), 
                key=lambda item: item[1], 
                reverse=True
            )}
        
        # Record training time
        self.training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {self.training_time:.2f} seconds")
        
        return {
            'model_type': self.model_type,
            'training_time': self.training_time,
            'feature_importances': self.feature_importances
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
            
        Returns:
        --------
        np.ndarray
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Making predictions...")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions using the trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
            
        Returns:
        --------
        np.ndarray
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Making probability predictions...")
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, 
                X_test: pd.DataFrame, 
                y_test: pd.Series,
                threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        threshold : float
            Probability threshold for binary classification
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model...")
        
        # Get probability predictions
        y_prob = self.predict_proba(X_test)
        
        # Convert to binary predictions based on threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Log results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'threshold': threshold
        }
    
    def plot_evaluation(self, 
                       evaluation_results: Dict[str, Any],
                       output_dir: str = 'results',
                       prefix: str = '') -> None:
        """
        Plot evaluation results.
        
        Parameters:
        -----------
        evaluation_results : Dict[str, Any]
            Results from evaluate()
        output_dir : str
            Directory to save plots
        prefix : str
            Prefix for plot filenames
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Plotting evaluation results...")
        
        # Extract data
        y_test = evaluation_results.get('y_test')
        y_prob = evaluation_results.get('y_prob')
        y_pred = evaluation_results.get('y_pred')
        cm = evaluation_results.get('confusion_matrix')
        
        if y_test is None or y_prob is None:
            logger.warning("Missing data for plotting. Make sure to run evaluate() first.")
            return
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = evaluation_results.get('roc_auc')
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}roc_curve.png'), dpi=300)
        plt.close()
        
        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = evaluation_results.get('avg_precision')
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=sum(y_test) / len(y_test), color='red', linestyle='--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}precision_recall_curve.png'), dpi=300)
        plt.close()
        
        # Plot feature importances if available
        if self.feature_importances:
            # Get top 20 features
            top_features = dict(list(self.feature_importances.items())[:20])
            
            plt.figure(figsize=(10, 8))
            plt.barh(list(top_features.keys())[::-1], list(top_features.values())[::-1])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}feature_importances.png'), dpi=300)
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}")
    
    def save_model(self, output_dir: str = 'models') -> str:
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the model
            
        Returns:
        --------
        str
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.model_type}_model_{timestamp}.joblib"
        filepath = os.path.join(output_dir, filename)
        
        # Save the model
        joblib.dump(self.model, filepath)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'training_time': self.training_time,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_importances': self.feature_importances
        }
        
        metadata_filepath = os.path.join(output_dir, f"{self.model_type}_metadata_{timestamp}.joblib")
        joblib.dump(metadata, metadata_filepath)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Metadata saved to {metadata_filepath}")
        
        self.model_path = filepath
        return filepath
    
    @classmethod
    def load_model(cls, model_path: str) -> 'ShippingFraudDetector':
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        ShippingFraudDetector
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load the model
        model = joblib.load(model_path)
        
        # Create a new instance
        detector = cls()
        detector.model = model
        detector.model_path = model_path
        
        # Try to load metadata
        metadata_path = model_path.replace('model_', 'metadata_')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            detector.model_type = metadata.get('model_type')
            detector.training_time = metadata.get('training_time')
            detector.feature_names = metadata.get('feature_names')
            detector.categorical_features = metadata.get('categorical_features')
            detector.numerical_features = metadata.get('numerical_features')
            detector.feature_importances = metadata.get('feature_importances')
        
        logger.info("Model loaded successfully")
        
        return detector


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import ShippingFraudDataLoader
    
    # Load and preprocess data
    data_loader = ShippingFraudDataLoader()
    X_train, X_val, y_train, y_val = data_loader.get_train_test_split()
    
    # Get feature names
    feature_info = data_loader.get_feature_names()
    categorical_features = feature_info['categorical']
    numerical_features = feature_info['numerical']
    
    # Create and train model
    detector = ShippingFraudDetector(model_type='xgboost')
    training_results = detector.train(
        X_train, 
        y_train,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        hyperparameter_tuning=True,
        n_iter=5
    )
    
    # Evaluate model
    evaluation_results = detector.evaluate(X_val, y_val)
    
    # Plot evaluation results
    detector.plot_evaluation(evaluation_results, output_dir='results', prefix='xgboost_')
    
    # Save model
    model_path = detector.save_model(output_dir='models')
    
    print(f"\nModel saved to {model_path}")
    print(f"Top 10 feature importances:")
    for feature, importance in list(detector.feature_importances.items())[:10]:
        print(f"  {feature}: {importance:.4f}") 