"""
Visualization Tools for Shipping Fraud Detection

This module provides visualization tools for analyzing and presenting
results from the Shipping Fraud Detection System.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudVisualization:
    """
    Visualization tools for shipping fraud detection results.
    """
    
    def __init__(self, output_dir: str = 'results/plots'):
        """
        Initialize the visualization module.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    def plot_fraud_distribution(self, 
                               df: pd.DataFrame, 
                               target_col: str = 'isFraud',
                               title: str = 'Fraud Distribution',
                               filename: str = 'fraud_distribution.png') -> None:
        """
        Plot the distribution of fraud vs. non-fraud transactions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Name of the target column
        title : str
            Plot title
        filename : str
            Output filename
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in DataFrame")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Count plot
        ax = sns.countplot(x=target_col, data=df)
        
        # Add percentages
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            percentage = height / total * 100
            ax.annotate(f'{height}\n({percentage:.1f}%)', 
                       (p.get_x() + p.get_width() / 2., height), 
                       ha='center', va='bottom')
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Is Fraud')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"Fraud distribution plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_feature_importance(self, 
                               feature_importances: Dict[str, float], 
                               top_n: int = 20,
                               title: str = 'Feature Importance',
                               filename: str = 'feature_importance.png') -> None:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        feature_importances : Dict[str, float]
            Dictionary mapping feature names to importance scores
        top_n : int
            Number of top features to display
        title : str
            Plot title
        filename : str
            Output filename
        """
        if not feature_importances:
            logger.warning("No feature importances provided")
            return
        
        # Sort features by importance
        sorted_features = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        plt.barh(list(sorted_features.keys())[::-1], list(sorted_features.values())[::-1])
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray,
                             labels: List[str] = ['Non-Fraud', 'Fraud'],
                             title: str = 'Confusion Matrix',
                             filename: str = 'confusion_matrix.png') -> None:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        labels : List[str]
            Class labels
        title : str
            Plot title
        filename : str
            Output filename
        """
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_roc_curve(self, 
                      fpr: np.ndarray, 
                      tpr: np.ndarray, 
                      roc_auc: float,
                      title: str = 'ROC Curve',
                      filename: str = 'roc_curve.png') -> None:
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        fpr : np.ndarray
            False positive rate
        tpr : np.ndarray
            True positive rate
        roc_auc : float
            Area under the ROC curve
        title : str
            Plot title
        filename : str
            Output filename
        """
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"ROC curve plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_precision_recall_curve(self, 
                                  precision: np.ndarray, 
                                  recall: np.ndarray, 
                                  avg_precision: float,
                                  baseline: float,
                                  title: str = 'Precision-Recall Curve',
                                  filename: str = 'precision_recall_curve.png') -> None:
        """
        Plot precision-recall curve.
        
        Parameters:
        -----------
        precision : np.ndarray
            Precision values
        recall : np.ndarray
            Recall values
        avg_precision : float
            Average precision score
        baseline : float
            Baseline precision (proportion of positive samples)
        title : str
            Plot title
        filename : str
            Output filename
        """
        plt.figure(figsize=(8, 6))
        
        # Plot precision-recall curve
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"Precision-recall curve plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_model_comparison(self, 
                             model_results: Dict[str, Dict[str, Any]],
                             metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision'],
                             title: str = 'Model Comparison',
                             filename: str = 'model_comparison.png') -> None:
        """
        Plot model comparison.
        
        Parameters:
        -----------
        model_results : Dict[str, Dict[str, Any]]
            Dictionary mapping model names to evaluation results
        metrics : List[str]
            List of metrics to compare
        title : str
            Plot title
        filename : str
            Output filename
        """
        if not model_results:
            logger.warning("No model results provided")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        model_names = list(model_results.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        # Plot bars for each model
        for i, model_name in enumerate(model_names):
            values = [model_results[model_name]['evaluation'][metric] for metric in metrics]
            plt.bar(x + i * width, values, width, label=model_name)
            
            # Add value labels
            for j, v in enumerate(values):
                plt.text(x[j] + i * width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(x + width * (len(model_names) - 1) / 2, metrics)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"Model comparison plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_time_patterns(self, 
                          df: pd.DataFrame,
                          hour_col: str = 'ShipmentHour',
                          day_col: str = 'ShipmentDayOfWeek',
                          target_col: str = 'isFraud',
                          title: str = 'Fraud Rate by Time',
                          filename: str = 'time_patterns.png') -> None:
        """
        Plot fraud rate by hour of day and day of week.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        hour_col : str
            Name of the hour column
        day_col : str
            Name of the day of week column
        target_col : str
            Name of the target column
        title : str
            Plot title
        filename : str
            Output filename
        """
        if hour_col not in df.columns or day_col not in df.columns or target_col not in df.columns:
            logger.warning(f"Required columns not found in DataFrame")
            return
        
        plt.figure(figsize=(14, 6))
        
        # Plot fraud rate by hour
        plt.subplot(1, 2, 1)
        fraud_by_hour = df.groupby(hour_col)[target_col].mean()
        sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values)
        plt.title('Fraud Rate by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Rate')
        plt.xticks(range(0, 24, 2))
        plt.grid(True, alpha=0.3)
        
        # Plot fraud rate by day of week
        plt.subplot(1, 2, 2)
        fraud_by_day = df.groupby(day_col)[target_col].mean()
        sns.barplot(x=fraud_by_day.index, y=fraud_by_day.values)
        plt.title('Fraud Rate by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Fraud Rate')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Save plot
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        logger.info(f"Time patterns plot saved to {os.path.join(self.output_dir, filename)}")
    
    def create_interactive_dashboard(self, 
                                    model_results: Dict[str, Dict[str, Any]],
                                    feature_importances: Dict[str, float],
                                    fraud_rings_report: pd.DataFrame,
                                    filename: str = 'fraud_dashboard.html') -> None:
        """
        Create an interactive dashboard with Plotly.
        
        Parameters:
        -----------
        model_results : Dict[str, Dict[str, Any]]
            Dictionary mapping model names to evaluation results
        feature_importances : Dict[str, float]
            Dictionary mapping feature names to importance scores
        fraud_rings_report : pd.DataFrame
            DataFrame containing information about detected fraud rings
        filename : str
            Output filename
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Performance Comparison',
                'Top 10 Feature Importances',
                'Fraud Ring Sizes',
                'Fraud Ring Risk Scores'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Model Performance Comparison
        if model_results:
            model_names = list(model_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
            
            for i, model_name in enumerate(model_names):
                values = [model_results[model_name]['evaluation'][metric] for metric in metrics]
                fig.add_trace(
                    go.Bar(
                        x=metrics,
                        y=values,
                        name=model_name,
                        text=[f'{v:.3f}' for v in values],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
        
        # 2. Feature Importances
        if feature_importances:
            # Sort and get top 10 features
            sorted_features = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig.add_trace(
                go.Bar(
                    x=list(sorted_features.values()),
                    y=list(sorted_features.keys()),
                    orientation='h',
                    marker=dict(color='rgba(58, 71, 80, 0.6)'),
                    name='Feature Importance'
                ),
                row=1, col=2
            )
        
        # 3. Fraud Ring Sizes
        if not fraud_rings_report.empty and 'Size' in fraud_rings_report.columns:
            size_counts = fraud_rings_report['Size'].value_counts().sort_index()
            
            fig.add_trace(
                go.Pie(
                    labels=size_counts.index,
                    values=size_counts.values,
                    name='Ring Sizes',
                    hole=.3,
                    textinfo='percent+label'
                ),
                row=2, col=1
            )
        
        # 4. Fraud Ring Risk Scores
        if not fraud_rings_report.empty and 'AvgRiskScore' in fraud_rings_report.columns and 'Size' in fraud_rings_report.columns:
            fig.add_trace(
                go.Scatter(
                    x=fraud_rings_report['Size'],
                    y=fraud_rings_report['AvgRiskScore'],
                    mode='markers',
                    marker=dict(
                        size=fraud_rings_report['Size'] * 3,
                        color=fraud_rings_report['AvgRiskScore'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Risk Score')
                    ),
                    name='Fraud Rings',
                    text=fraud_rings_report['RingID'].astype(str)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text='Shipping Fraud Detection Dashboard',
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes
        fig.update_yaxes(title_text='Score', row=1, col=1)
        fig.update_xaxes(title_text='Metric', row=1, col=1)
        
        fig.update_yaxes(title_text='Feature', row=1, col=2)
        fig.update_xaxes(title_text='Importance', row=1, col=2)
        
        fig.update_yaxes(title_text='Risk Score', row=2, col=2)
        fig.update_xaxes(title_text='Ring Size', row=2, col=2)
        
        # Save to HTML
        fig.write_html(os.path.join(self.output_dir, filename))
        
        logger.info(f"Interactive dashboard saved to {os.path.join(self.output_dir, filename)}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'isFraud': y_true,
        'ShipmentHour': np.random.randint(0, 24, 1000),
        'ShipmentDayOfWeek': np.random.randint(0, 7, 1000)
    })
    
    # Create sample feature importances
    feature_importances = {
        f'feature_{i}': np.random.random() for i in range(20)
    }
    
    # Create sample model results
    model_results = {
        'xgboost': {
            'evaluation': {
                'accuracy': 0.92,
                'precision': 0.89,
                'recall': 0.85,
                'f1': 0.87,
                'roc_auc': 0.95,
                'avg_precision': 0.90
            }
        },
        'random_forest': {
            'evaluation': {
                'accuracy': 0.90,
                'precision': 0.87,
                'recall': 0.83,
                'f1': 0.85,
                'roc_auc': 0.93,
                'avg_precision': 0.88
            }
        }
    }
    
    # Create sample fraud rings report
    fraud_rings_report = pd.DataFrame({
        'RingID': range(1, 11),
        'Size': np.random.randint(3, 10, 10),
        'AvgRiskScore': np.random.random(10),
        'SharedAddresses': np.random.randint(1, 5, 10),
        'SharedDevices': np.random.randint(1, 5, 10),
        'SharedIPs': np.random.randint(1, 5, 10),
        'SharedEmails': np.random.randint(1, 5, 10)
    })
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = 0.85
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = 0.80
    baseline = sum(y_true) / len(y_true)
    
    # Create visualizer
    visualizer = FraudVisualization(output_dir='example_plots')
    
    # Generate plots
    visualizer.plot_fraud_distribution(df)
    visualizer.plot_feature_importance(feature_importances)
    visualizer.plot_confusion_matrix(cm)
    visualizer.plot_roc_curve(fpr, tpr, roc_auc)
    visualizer.plot_precision_recall_curve(precision, recall, avg_precision, baseline)
    visualizer.plot_model_comparison(model_results)
    visualizer.plot_time_patterns(df)
    visualizer.create_interactive_dashboard(model_results, feature_importances, fraud_rings_report) 