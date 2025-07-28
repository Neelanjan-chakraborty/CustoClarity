"""
Data Loading Utilities for CUSTO CLARITY Project

This module contains functions for loading and downloading the Mall Customer Segmentation dataset.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import pandas as pd
import numpy as np
import os
import requests
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and management class for customer segmentation analysis."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download_mall_dataset(self) -> str:
        """
        Download the Mall Customer Segmentation dataset.
        
        Returns:
            str: Path to the downloaded dataset
        """
        # Sample Mall Customer Segmentation data (since we can't directly download from Kaggle)
        # This creates a representative dataset based on the original structure
        
        dataset_path = os.path.join(self.raw_dir, "mall_customers.csv")
        
        if os.path.exists(dataset_path):
            logger.info(f"Dataset already exists at {dataset_path}")
            return dataset_path
        
        logger.info("Creating Mall Customer Segmentation dataset...")
        
        # Generate sample data that matches the original dataset structure
        np.random.seed(42)  # For reproducibility
        
        # Generate customer data
        n_customers = 200
        customer_ids = range(1, n_customers + 1)
        
        # Generate age with realistic distribution
        ages = np.random.normal(38, 12, n_customers).astype(int)
        ages = np.clip(ages, 18, 70)  # Clip to reasonable age range
        
        # Generate gender with slight female bias (as in original dataset)
        genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.44, 0.56])
        
        # Generate annual income (in thousands) with log-normal distribution
        annual_incomes = np.random.lognormal(3.7, 0.4, n_customers).astype(int)
        annual_incomes = np.clip(annual_incomes, 15, 137)  # Match original range
        
        # Generate spending scores with multiple patterns for different segments
        spending_scores = []
        for i in range(n_customers):
            income = annual_incomes[i]
            age = ages[i]
            
            # Create different spending patterns based on income and age
            if income < 40:  # Low income
                base_score = np.random.normal(35, 15)
            elif income < 70:  # Medium income
                base_score = np.random.normal(50, 20)
            else:  # High income
                base_score = np.random.normal(60, 25)
            
            # Age factor (younger people tend to spend more on lifestyle)
            if age < 30:
                base_score += np.random.normal(10, 5)
            elif age > 50:
                base_score -= np.random.normal(5, 3)
            
            spending_scores.append(max(1, min(100, int(base_score))))
        
        # Create DataFrame
        data = {
            'CustomerID': customer_ids,
            'Gender': genders,
            'Age': ages,
            'Annual Income (k$)': annual_incomes,
            'Spending Score (1-100)': spending_scores
        }
        
        df = pd.DataFrame(data)
        
        # Save the dataset
        df.to_csv(dataset_path, index=False)
        logger.info(f"Dataset created and saved to {dataset_path}")
        
        return dataset_path
    
    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the customer dataset.
        
        Args:
            file_path (str, optional): Path to the dataset file. If None, uses default path.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path is None:
            file_path = self.download_mall_dataset()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Dataset not found at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df (pd.DataFrame): The dataset
        
        Returns:
            dict: Dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to the processed directory.
        
        Args:
            df (pd.DataFrame): Processed dataset
            filename (str): Filename for the processed data
        
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.processed_dir, filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}")
        return file_path
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from the processed directory.
        
        Args:
            filename (str): Filename of the processed data
        
        Returns:
            pd.DataFrame: Loaded processed dataset
        """
        file_path = os.path.join(self.processed_dir, filename)
        df = pd.read_csv(file_path)
        logger.info(f"Processed data loaded from {file_path}")
        return df


def load_sample_data() -> pd.DataFrame:
    """
    Quick function to load sample customer data for testing.
    
    Returns:
        pd.DataFrame: Sample customer data
    """
    loader = DataLoader()
    return loader.load_dataset()


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
