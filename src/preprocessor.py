"""
Data Preprocessing Module for CUSTO CLARITY Project

This module contains functions for data cleaning, feature engineering, and preprocessing
for customer segmentation analysis.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerDataPreprocessor:
    """Comprehensive data preprocessing class for customer segmentation analysis."""
    
    def __init__(self):
        """Initialize the preprocessor with default scalers and encoders."""
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.preprocessing_steps = []
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'mean',
                            categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            strategy (str): Strategy for numerical columns ('mean', 'median', 'constant')
            categorical_strategy (str): Strategy for categorical columns ('most_frequent', 'constant')
        
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        df_processed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        if len(numerical_cols) > 0 and df_processed[numerical_cols].isnull().sum().sum() > 0:
            num_imputer = SimpleImputer(strategy=strategy)
            df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
            self.imputers['numerical'] = num_imputer
            logger.info(f"Handled missing values in numerical columns using {strategy} strategy")
        
        # Handle categorical missing values
        if len(categorical_cols) > 0 and df_processed[categorical_cols].isnull().sum().sum() > 0:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
            self.imputers['categorical'] = cat_imputer
            logger.info(f"Handled missing values in categorical columns using {categorical_strategy} strategy")
        
        self.preprocessing_steps.append(f"Missing values handled: num({strategy}), cat({categorical_strategy})")
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None,
                                   method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (List[str], optional): Columns to encode. If None, all object columns.
            method (str): Encoding method ('label', 'onehot')
        
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        if method == 'label':
            for col in columns:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Label encoded column: {col}")
        
        elif method == 'onehot':
            df_processed = pd.get_dummies(df_processed, columns=columns, prefix=columns)
            logger.info(f"One-hot encoded columns: {columns}")
        
        self.preprocessing_steps.append(f"Categorical encoding: {method} for {columns}")
        return df_processed
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """
        Detect outliers in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (List[str], optional): Columns to check. If None, all numerical columns.
            method (str): Method to detect outliers ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with outlier indices for each column
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > threshold
                
                outliers[col] = df[outlier_mask].index.values
                logger.info(f"Found {len(outliers[col])} outliers in {col} using {method} method")
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, 
                       outliers: Dict[str, np.ndarray],
                       method: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            outliers (Dict[str, np.ndarray]): Outlier indices from detect_outliers
            method (str): Method to handle outliers ('remove', 'cap', 'log_transform')
        
        Returns:
            pd.DataFrame: Dataset with handled outliers
        """
        df_processed = df.copy()
        
        if method == 'remove':
            # Remove rows with outliers
            all_outlier_indices = set()
            for indices in outliers.values():
                all_outlier_indices.update(indices)
            df_processed = df_processed.drop(list(all_outlier_indices))
            logger.info(f"Removed {len(all_outlier_indices)} outlier rows")
        
        elif method == 'cap':
            # Cap outliers to 5th and 95th percentiles
            for col in outliers.keys():
                if col in df_processed.columns:
                    lower_cap = df_processed[col].quantile(0.05)
                    upper_cap = df_processed[col].quantile(0.95)
                    df_processed[col] = df_processed[col].clip(lower=lower_cap, upper=upper_cap)
                    logger.info(f"Capped outliers in {col} to [{lower_cap:.2f}, {upper_cap:.2f}]")
        
        elif method == 'log_transform':
            # Log transform to reduce impact of outliers
            for col in outliers.keys():
                if col in df_processed.columns and (df_processed[col] > 0).all():
                    df_processed[col] = np.log1p(df_processed[col])
                    logger.info(f"Applied log transformation to {col}")
        
        self.preprocessing_steps.append(f"Outliers handled using {method} method")
        return df_processed
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for customer segmentation analysis.
        
        Args:
            df (pd.DataFrame): Input dataset
        
        Returns:
            pd.DataFrame: Dataset with additional features
        """
        df_processed = df.copy()
        
        # Assuming standard column names for Mall Customer dataset
        if 'Age' in df_processed.columns and 'Annual Income (k$)' in df_processed.columns:
            # Age groups
            df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                             bins=[0, 25, 35, 50, 100], 
                                             labels=['Young', 'Adult', 'Middle_Age', 'Senior'])
            
            # Income groups
            df_processed['Income_Group'] = pd.cut(df_processed['Annual Income (k$)'], 
                                                bins=[0, 40, 70, 200], 
                                                labels=['Low', 'Medium', 'High'])
            
            logger.info("Created Age_Group and Income_Group features")
        
        if 'Spending Score (1-100)' in df_processed.columns:
            # Spending categories
            df_processed['Spending_Category'] = pd.cut(df_processed['Spending Score (1-100)'], 
                                                     bins=[0, 35, 65, 100], 
                                                     labels=['Low_Spender', 'Medium_Spender', 'High_Spender'])
            
            logger.info("Created Spending_Category feature")
        
        # Income to spending ratio (if both columns exist)
        if 'Annual Income (k$)' in df_processed.columns and 'Spending Score (1-100)' in df_processed.columns:
            df_processed['Income_Spending_Ratio'] = (df_processed['Annual Income (k$)'] / 
                                                   df_processed['Spending Score (1-100)'])
            logger.info("Created Income_Spending_Ratio feature")
        
        self.preprocessing_steps.append("Feature engineering completed")
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (List[str], optional): Columns to scale. If None, all numerical columns.
            method (str): Scaling method ('standard', 'minmax', 'robust')
        
        Returns:
            Tuple[pd.DataFrame, Any]: Scaled dataset and fitted scaler
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            # Remove categorical encoded columns if they exist
            columns = [col for col in columns if 'CustomerID' not in col]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        df_processed[columns] = scaler.fit_transform(df_processed[columns])
        
        logger.info(f"Applied {method} scaling to columns: {columns}")
        self.preprocessing_steps.append(f"Feature scaling: {method} for {columns}")
        
        return df_processed, scaler
    
    def prepare_for_clustering(self, df: pd.DataFrame,
                             target_columns: Optional[List[str]] = None,
                             encoding_method: str = 'label',
                             scaling_method: str = 'standard',
                             handle_outliers_method: str = 'cap') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Complete preprocessing pipeline for clustering analysis.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_columns (List[str], optional): Columns to use for clustering
            encoding_method (str): Method for categorical encoding
            scaling_method (str): Method for feature scaling
            handle_outliers_method (str): Method for outlier handling
        
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Processed dataset and feature matrix for clustering
        """
        logger.info("Starting complete preprocessing pipeline for clustering...")
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Create additional features
        df_processed = self.create_features(df_processed)
        
        # Step 3: Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            df_processed = self.encode_categorical_variables(df_processed, 
                                                           categorical_cols, 
                                                           encoding_method)
        
        # Step 4: Detect and handle outliers
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'CustomerID' in numerical_cols:
            numerical_cols.remove('CustomerID')
        
        outliers = self.detect_outliers(df_processed, numerical_cols)
        if any(len(indices) > 0 for indices in outliers.values()):
            df_processed = self.handle_outliers(df_processed, outliers, handle_outliers_method)
        
        # Step 5: Select features for clustering
        if target_columns is None:
            # Use key features for clustering
            feature_candidates = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            target_columns = [col for col in feature_candidates if col in df_processed.columns]
            
            # Add engineered features if available
            if 'Income_Spending_Ratio' in df_processed.columns:
                target_columns.append('Income_Spending_Ratio')
        
        # Step 6: Scale features
        df_scaled, scaler = self.scale_features(df_processed, target_columns, scaling_method)
        
        # Step 7: Create feature matrix for clustering
        X = df_scaled[target_columns].values
        self.feature_names = target_columns
        
        logger.info(f"Preprocessing completed. Feature matrix shape: {X.shape}")
        logger.info(f"Features used for clustering: {target_columns}")
        
        return df_scaled, X
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all preprocessing steps performed.
        
        Returns:
            Dict[str, Any]: Summary of preprocessing steps
        """
        summary = {
            'steps_performed': self.preprocessing_steps,
            'features_for_clustering': self.feature_names,
            'scalers_used': list(self.imputers.keys()) if self.imputers else [],
            'label_encoders': list(self.label_encoders.keys()),
        }
        
        return summary


def quick_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Quick preprocessing function for immediate use.
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Processed dataset and feature matrix
    """
    preprocessor = CustomerDataPreprocessor()
    return preprocessor.prepare_for_clustering(df)


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import load_sample_data
    
    df = load_sample_data()
    preprocessor = CustomerDataPreprocessor()
    
    df_processed, X = preprocessor.prepare_for_clustering(df)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"\nPreprocessing summary:")
    print(preprocessor.get_preprocessing_summary())
