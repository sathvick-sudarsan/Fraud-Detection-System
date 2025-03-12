"""
Data Loader for Shipping Fraud Detection

This module loads and preprocesses the IEEE-CIS Fraud Detection dataset,
adapting it for shipping fraud detection by mapping transaction features
to shipping-related features.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShippingFraudDataLoader:
    """
    Loads and preprocesses the IEEE-CIS Fraud Detection dataset for shipping fraud detection.
    
    This class adapts the transaction and identity data from the IEEE-CIS Fraud Detection
    dataset to fit a shipping fraud context, mapping transaction features to shipping-related
    features.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the raw data files
        """
        self.data_dir = data_dir
        self.train_transaction = None
        self.train_identity = None
        self.test_transaction = None
        self.test_identity = None
        self.train_data = None
        self.test_data = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
        # Feature mapping from transaction to shipping context
        self.feature_mapping = {
            'TransactionID': 'ShipmentID',
            'isFraud': 'isFraud',
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
            'R_emaildomain': 'RecipientEmailDomain'
        }
        
        # Identity features mapping
        self.identity_mapping = {
            'DeviceType': 'DeviceType',
            'DeviceInfo': 'DeviceInfo'
        }
        
    def load_data(self) -> None:
        """
        Load the raw transaction and identity data.
        """
        logger.info("Loading raw data files...")
        
        # Load transaction data
        train_transaction_path = os.path.join(self.data_dir, 'train_transaction.csv')
        test_transaction_path = os.path.join(self.data_dir, 'test_transaction.csv')
        
        self.train_transaction = pd.read_csv(train_transaction_path)
        self.test_transaction = pd.read_csv(test_transaction_path)
        
        # Load identity data
        train_identity_path = os.path.join(self.data_dir, 'train_identity.csv')
        test_identity_path = os.path.join(self.data_dir, 'test_identity.csv')
        
        self.train_identity = pd.read_csv(train_identity_path)
        self.test_identity = pd.read_csv(test_identity_path)
        
        logger.info(f"Loaded {len(self.train_transaction)} training transactions and {len(self.test_transaction)} test transactions")
        logger.info(f"Loaded {len(self.train_identity)} training identity records and {len(self.test_identity)} test identity records")
    
    def merge_data(self) -> None:
        """
        Merge transaction and identity data.
        """
        if self.train_transaction is None or self.test_transaction is None:
            self.load_data()
            
        logger.info("Merging transaction and identity data...")
        
        # Merge training data
        self.train_data = pd.merge(
            self.train_transaction, 
            self.train_identity, 
            on='TransactionID', 
            how='left'
        )
        
        # Merge test data
        self.test_data = pd.merge(
            self.test_transaction, 
            self.test_identity, 
            on='TransactionID', 
            how='left'
        )
        
        logger.info(f"Merged data shapes: Train: {self.train_data.shape}, Test: {self.test_data.shape}")
    
    def preprocess_data(self, 
                        selected_features: Optional[List[str]] = None,
                        rename_features: bool = True,
                        handle_missing: bool = True,
                        scale_numerical: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data for modeling.
        
        Parameters:
        -----------
        selected_features : Optional[List[str]]
            List of features to select from the dataset. If None, use a predefined set of features.
        rename_features : bool
            Whether to rename features to shipping context
        handle_missing : bool
            Whether to handle missing values
        scale_numerical : bool
            Whether to scale numerical features
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Preprocessed training and test data
        """
        if self.train_data is None or self.test_data is None:
            self.merge_data()
            
        logger.info("Preprocessing data...")
        
        # Select features
        if selected_features is None:
            # Use a predefined set of features that are most relevant
            # These are based on feature importance from previous analyses of the dataset
            selected_features = [
                'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD',
                'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                'addr1', 'addr2', 'dist1', 'dist2',
                'P_emaildomain', 'R_emaildomain',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
                'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
                'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45',
                'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62',
                'DeviceType', 'DeviceInfo'
            ]
            
            # Add target variable for training data
            if 'isFraud' in self.train_data.columns:
                selected_features.append('isFraud')
        
        # Select features that exist in both datasets
        existing_features = [f for f in selected_features if f in self.train_data.columns]
        
        train_data = self.train_data[existing_features].copy()
        
        # For test data, exclude 'isFraud' if it doesn't exist
        test_features = [f for f in existing_features if f != 'isFraud' or f in self.test_data.columns]
        test_data = self.test_data[test_features].copy()
        
        # Identify categorical and numerical features
        categorical_features = [
            'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo'
        ]
        
        # Keep only categorical features that exist in the selected features
        self.categorical_features = [f for f in categorical_features if f in existing_features]
        
        # Numerical features are all non-categorical features except ID and target
        self.numerical_features = [
            f for f in existing_features 
            if f not in self.categorical_features 
            and f not in ['TransactionID', 'isFraud']
        ]
        
        # Handle missing values
        if handle_missing:
            logger.info("Handling missing values...")
            
            # Fill missing numerical values with median
            for feature in self.numerical_features:
                median_value = train_data[feature].median()
                train_data[feature] = train_data[feature].fillna(median_value)
                test_data[feature] = test_data[feature].fillna(median_value)
            
            # Fill missing categorical values with 'Unknown'
            for feature in self.categorical_features:
                train_data[feature] = train_data[feature].fillna('Unknown')
                test_data[feature] = test_data[feature].fillna('Unknown')
        
        # Scale numerical features
        if scale_numerical:
            logger.info("Scaling numerical features...")
            scaler = StandardScaler()
            
            # Fit on training data, transform both training and test
            train_data[self.numerical_features] = scaler.fit_transform(
                train_data[self.numerical_features].values
            )
            
            test_data[self.numerical_features] = scaler.transform(
                test_data[self.numerical_features].values
            )
        
        # Rename features to shipping context
        if rename_features:
            logger.info("Renaming features to shipping context...")
            
            # Create mapping dictionary for selected features
            mapping = {}
            for old_name, new_name in self.feature_mapping.items():
                if old_name in train_data.columns:
                    mapping[old_name] = new_name
            
            # Add identity mapping
            for old_name, new_name in self.identity_mapping.items():
                if old_name in train_data.columns:
                    mapping[old_name] = new_name
            
            # Rename columns
            train_data = train_data.rename(columns=mapping)
            test_data = test_data.rename(columns=mapping)
            
            # Update feature lists with new names
            self.numerical_features = [mapping.get(f, f) for f in self.numerical_features]
            self.categorical_features = [mapping.get(f, f) for f in self.categorical_features]
        
        # Store feature names for later use
        self.feature_names = list(train_data.columns)
        
        logger.info(f"Preprocessing complete. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        
        return train_data, test_data
    
    def get_train_test_split(self, 
                            test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the preprocessed training data into train and validation sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of the data to include in the validation set
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_val, y_train, y_val
        """
        train_data, _ = self.preprocess_data()
        
        if 'isFraud' not in train_data.columns:
            raise ValueError("Target variable 'isFraud' not found in training data")
        
        # Split features and target
        X = train_data.drop('isFraud', axis=1)
        y = train_data['isFraud']
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train-validation split: X_train: {X_train.shape}, X_val: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def get_test_data(self) -> pd.DataFrame:
        """
        Get the preprocessed test data.
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed test data
        """
        _, test_data = self.preprocess_data()
        return test_data
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get the feature names categorized by type.
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary with feature names by type
        """
        if self.feature_names is None:
            self.preprocess_data()
            
        return {
            'all': self.feature_names,
            'categorical': self.categorical_features,
            'numerical': self.numerical_features
        }


if __name__ == "__main__":
    # Example usage
    data_loader = ShippingFraudDataLoader()
    train_data, test_data = data_loader.preprocess_data()
    
    print("\nData Sample:")
    print(train_data.head())
    
    print("\nFeature Types:")
    feature_names = data_loader.get_feature_names()
    print(f"Categorical Features ({len(feature_names['categorical'])}): {feature_names['categorical']}")
    print(f"Numerical Features ({len(feature_names['numerical'])}): {feature_names['numerical'][:10]}...")
    
    # Get fraud distribution
    if 'isFraud' in train_data.columns:
        fraud_count = train_data['isFraud'].sum()
        total_count = len(train_data)
        fraud_percentage = (fraud_count / total_count) * 100
        
        print(f"\nFraud Distribution:")
        print(f"Fraud transactions: {fraud_count} ({fraud_percentage:.2f}%)")
        print(f"Non-fraud transactions: {total_count - fraud_count} ({100 - fraud_percentage:.2f}%)") 