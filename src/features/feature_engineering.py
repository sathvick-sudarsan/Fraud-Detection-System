"""
Feature Engineering for Shipping Fraud Detection

This module provides functions to extract additional features from the shipping data
that might be useful for detecting fraudulent activities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShippingFraudFeatureEngineer:
    """
    Extracts additional features from shipping data for fraud detection.
    """
    
    def __init__(self):
        """
        Initialize the feature engineer.
        """
        pass
    
    def extract_time_features(self, df: pd.DataFrame, timestamp_col: str = 'ShipmentTimestamp') -> pd.DataFrame:
        """
        Extract time-based features from the timestamp column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        timestamp_col : str
            Name of the timestamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional time features
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            return df
        
        logger.info("Extracting time-based features...")
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # The timestamp in the dataset is a timedelta from a reference date
        # We'll use a reference date of 2017-12-01 (based on the competition description)
        reference_date = datetime(2017, 12, 1)
        
        # Convert timestamp to datetime
        result_df['ShipmentDateTime'] = result_df[timestamp_col].apply(
            lambda x: reference_date + timedelta(seconds=x)
        )
        
        # Extract time-based features
        result_df['ShipmentHour'] = result_df['ShipmentDateTime'].dt.hour
        result_df['ShipmentDayOfWeek'] = result_df['ShipmentDateTime'].dt.dayofweek
        result_df['ShipmentDayOfMonth'] = result_df['ShipmentDateTime'].dt.day
        result_df['ShipmentMonth'] = result_df['ShipmentDateTime'].dt.month
        result_df['ShipmentWeekend'] = result_df['ShipmentDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Time of day categories
        result_df['ShipmentTimeOfDay'] = result_df['ShipmentHour'].apply(
            lambda hour: 'Night' if 0 <= hour < 6 else 
                        'Morning' if 6 <= hour < 12 else
                        'Afternoon' if 12 <= hour < 18 else 'Evening'
        )
        
        # Drop the intermediate datetime column
        result_df.drop('ShipmentDateTime', axis=1, inplace=True)
        
        logger.info(f"Added time-based features: {['ShipmentHour', 'ShipmentDayOfWeek', 'ShipmentDayOfMonth', 'ShipmentMonth', 'ShipmentWeekend', 'ShipmentTimeOfDay']}")
        
        return result_df
    
    def extract_email_features(self, 
                              df: pd.DataFrame, 
                              sender_email_col: str = 'SenderEmailDomain', 
                              recipient_email_col: str = 'RecipientEmailDomain') -> pd.DataFrame:
        """
        Extract features from email domains.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        sender_email_col : str
            Name of the sender email domain column
        recipient_email_col : str
            Name of the recipient email domain column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional email features
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Check if email columns exist
        email_cols_exist = True
        if sender_email_col not in result_df.columns:
            logger.warning(f"Sender email column '{sender_email_col}' not found in DataFrame")
            email_cols_exist = False
        
        if recipient_email_col not in result_df.columns:
            logger.warning(f"Recipient email column '{recipient_email_col}' not found in DataFrame")
            email_cols_exist = False
        
        if not email_cols_exist:
            return result_df
        
        logger.info("Extracting email-based features...")
        
        # Email provider categories
        email_providers = {
            'HighRisk': ['protonmail.com', 'tutanota.com', 'guerrillamail.com', 'temp-mail.org', 'mailinator.com'],
            'Free': ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'mail.com', 'icloud.com'],
            'Corporate': ['company.com', 'business.com', 'corp.com', 'enterprise.com']
        }
        
        # Function to categorize email domains
        def categorize_email(domain):
            if pd.isna(domain):
                return 'Unknown'
            
            for category, providers in email_providers.items():
                if any(provider in str(domain).lower() for provider in providers):
                    return category
            
            # Check for common TLDs
            if str(domain).lower().endswith(('.edu', '.gov', '.mil')):
                return 'Institutional'
            elif str(domain).lower().endswith('.org'):
                return 'Organization'
            else:
                return 'Other'
        
        # Categorize sender and recipient email domains
        result_df['SenderEmailCategory'] = result_df[sender_email_col].apply(categorize_email)
        result_df['RecipientEmailCategory'] = result_df[recipient_email_col].apply(categorize_email)
        
        # Check if sender and recipient email domains match
        result_df['EmailDomainMatch'] = (result_df[sender_email_col] == result_df[recipient_email_col]).astype(int)
        
        logger.info(f"Added email-based features: {['SenderEmailCategory', 'RecipientEmailCategory', 'EmailDomainMatch']}")
        
        return result_df
    
    def extract_payment_features(self, 
                               df: pd.DataFrame, 
                               payment_method_col: str = 'PaymentMethod',
                               payment_account_type_col: str = 'PaymentAccountType',
                               payment_card_type_col: str = 'PaymentCardType',
                               shipment_value_col: str = 'ShipmentValue') -> pd.DataFrame:
        """
        Extract features related to payment methods and amounts.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        payment_method_col : str
            Name of the payment method column
        payment_account_type_col : str
            Name of the payment account type column
        payment_card_type_col : str
            Name of the payment card type column
        shipment_value_col : str
            Name of the shipment value column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional payment features
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_cols = [payment_method_col, payment_account_type_col, payment_card_type_col, shipment_value_col]
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for payment features: {missing_cols}")
            return result_df
        
        logger.info("Extracting payment-related features...")
        
        # Create a risk score for payment methods (higher score = higher risk)
        payment_risk = {
            'discover': 2,
            'visa': 1,
            'mastercard': 1,
            'american express': 3,
            'Unknown': 4
        }
        
        # Create a risk score for payment card types
        card_type_risk = {
            'credit': 2,
            'debit': 1,
            'Unknown': 3
        }
        
        # Apply risk scores
        result_df['PaymentMethodRisk'] = result_df[payment_method_col].apply(
            lambda x: payment_risk.get(str(x).lower(), 4)
        )
        
        result_df['PaymentCardTypeRisk'] = result_df[payment_card_type_col].apply(
            lambda x: card_type_risk.get(str(x).lower(), 3)
        )
        
        # Calculate combined payment risk
        result_df['CombinedPaymentRisk'] = result_df['PaymentMethodRisk'] * result_df['PaymentCardTypeRisk']
        
        # Value-to-risk ratio (higher values with risky payment methods are more suspicious)
        result_df['ValueToRiskRatio'] = result_df[shipment_value_col] * result_df['CombinedPaymentRisk']
        
        # Bin shipment values into categories
        result_df['ShipmentValueCategory'] = pd.qcut(
            result_df[shipment_value_col], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info(f"Added payment-related features: {['PaymentMethodRisk', 'PaymentCardTypeRisk', 'CombinedPaymentRisk', 'ValueToRiskRatio', 'ShipmentValueCategory']}")
        
        return result_df
    
    def extract_address_features(self, 
                               df: pd.DataFrame, 
                               sender_address_col: str = 'SenderAddressID',
                               recipient_address_col: str = 'RecipientAddressID',
                               shipping_distance_col: str = 'ShippingDistance',
                               billing_distance_col: str = 'BillingDistance') -> pd.DataFrame:
        """
        Extract features related to addresses and distances.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        sender_address_col : str
            Name of the sender address ID column
        recipient_address_col : str
            Name of the recipient address ID column
        shipping_distance_col : str
            Name of the shipping distance column
        billing_distance_col : str
            Name of the billing distance column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional address features
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_cols = [sender_address_col, recipient_address_col]
        distance_cols = [shipping_distance_col, billing_distance_col]
        
        missing_required = [col for col in required_cols if col not in result_df.columns]
        if missing_required:
            logger.warning(f"Missing required columns for address features: {missing_required}")
            return result_df
        
        logger.info("Extracting address-related features...")
        
        # Check if addresses match (same pickup and delivery address is suspicious)
        result_df['AddressMatch'] = (result_df[sender_address_col] == result_df[recipient_address_col]).astype(int)
        
        # Process distance features if available
        available_distance_cols = [col for col in distance_cols if col in result_df.columns]
        
        if available_distance_cols:
            # Fill missing distance values with median
            for col in available_distance_cols:
                median_distance = result_df[col].median()
                result_df[col] = result_df[col].fillna(median_distance)
            
            # Calculate distance ratio if both distances are available
            if len(available_distance_cols) == 2:
                result_df['DistanceRatio'] = result_df[shipping_distance_col] / result_df[billing_distance_col].replace(0, 0.1)
                
                # Flag cases where shipping distance is much larger than billing distance
                result_df['LargeDistanceGap'] = (result_df['DistanceRatio'] > 5).astype(int)
                
                logger.info(f"Added distance-related features: {['DistanceRatio', 'LargeDistanceGap']}")
        
        logger.info(f"Added address-related features: {['AddressMatch']}")
        
        return result_df
    
    def extract_device_features(self, 
                              df: pd.DataFrame, 
                              device_type_col: str = 'DeviceType',
                              device_info_col: str = 'DeviceInfo') -> pd.DataFrame:
        """
        Extract features related to devices used for shipping.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        device_type_col : str
            Name of the device type column
        device_info_col : str
            Name of the device info column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional device features
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_cols = [device_type_col, device_info_col]
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for device features: {missing_cols}")
            return result_df
        
        logger.info("Extracting device-related features...")
        
        # Extract operating system from device info
        def extract_os(device_info):
            if pd.isna(device_info):
                return 'Unknown'
            
            device_info = str(device_info).lower()
            
            if 'android' in device_info:
                return 'Android'
            elif 'ios' in device_info or 'iphone' in device_info or 'ipad' in device_info:
                return 'iOS'
            elif 'windows' in device_info:
                return 'Windows'
            elif 'mac' in device_info or 'macintosh' in device_info:
                return 'MacOS'
            elif 'linux' in device_info:
                return 'Linux'
            else:
                return 'Other'
        
        # Extract browser from device info
        def extract_browser(device_info):
            if pd.isna(device_info):
                return 'Unknown'
            
            device_info = str(device_info).lower()
            
            if 'chrome' in device_info:
                return 'Chrome'
            elif 'firefox' in device_info:
                return 'Firefox'
            elif 'safari' in device_info:
                return 'Safari'
            elif 'edge' in device_info:
                return 'Edge'
            elif 'opera' in device_info:
                return 'Opera'
            elif 'ie' in device_info or 'internet explorer' in device_info:
                return 'Internet Explorer'
            else:
                return 'Other'
        
        # Apply extraction functions
        result_df['OperatingSystem'] = result_df[device_info_col].apply(extract_os)
        result_df['Browser'] = result_df[device_info_col].apply(extract_browser)
        
        # Create device risk score
        # Mobile devices are generally considered lower risk than desktops for fraud
        device_risk = {
            'mobile': 1,
            'desktop': 2,
            'Unknown': 3
        }
        
        result_df['DeviceRisk'] = result_df[device_type_col].apply(
            lambda x: device_risk.get(str(x).lower(), 3)
        )
        
        logger.info(f"Added device-related features: {['OperatingSystem', 'Browser', 'DeviceRisk']}")
        
        return result_df
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all additional features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all additional features
        """
        logger.info("Extracting all additional features...")
        
        result_df = df.copy()
        
        # Apply all feature extraction methods
        result_df = self.extract_time_features(result_df)
        result_df = self.extract_email_features(result_df)
        result_df = self.extract_payment_features(result_df)
        result_df = self.extract_address_features(result_df)
        result_df = self.extract_device_features(result_df)
        
        logger.info(f"Feature extraction complete. New shape: {result_df.shape}")
        
        return result_df


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import ShippingFraudDataLoader
    
    # Load and preprocess data
    data_loader = ShippingFraudDataLoader()
    train_data, test_data = data_loader.preprocess_data()
    
    # Extract additional features
    feature_engineer = ShippingFraudFeatureEngineer()
    train_data_enriched = feature_engineer.extract_all_features(train_data)
    
    # Display new features
    new_features = set(train_data_enriched.columns) - set(train_data.columns)
    print(f"Added {len(new_features)} new features: {sorted(new_features)}")
    
    # Display sample data with new features
    print("\nSample data with new features:")
    print(train_data_enriched[list(new_features)].head()) 