"""
Data Loader Module

This module contains functions for loading and performing initial checks on the marketing campaign data.
"""

import pandas as pd
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the marketing campaign dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path, delimiter=';')
        logger.info(f"Successfully loaded {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def get_data_info(data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Get basic information about the dataset including statistics and null values.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        
    Returns:
        Tuple containing:
            - Dictionary with dataset statistics
            - Dictionary with null value counts per column
    """
    logger.info("Generating dataset information")
    
    # Get basic statistics
    stats = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "numeric_stats": data.describe().to_dict()
    }
    
    # Count null values
    null_counts = data.isnull().sum().to_dict()
    
    return stats, null_counts

def identify_column_types(data: pd.DataFrame) -> Dict[str, list]:
    """
    Identify and categorize columns by data type.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        
    Returns:
        Dict: Dictionary with column names categorized by type
    """
    logger.info("Identifying column types")
    
    column_types = {
        "numeric": list(data.select_dtypes(include=['float64', 'int64']).columns),
        "categorical": list(data.select_dtypes(include=['object']).columns),
        "datetime": [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
    }
    
    logger.info(f"Found {len(column_types['numeric'])} numeric columns, "
                f"{len(column_types['categorical'])} categorical columns, and "
                f"{len(column_types['datetime'])} datetime columns")
    
    return column_types

def display_data_summary(data: pd.DataFrame) -> None:
    """
    Display a comprehensive summary of the dataset.
    
    Args:
        data (pd.DataFrame): The dataset to summarize
    """
    stats, null_counts = get_data_info(data)
    column_types = identify_column_types(data)
    
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total records: {stats['shape'][0]}")
    print(f"Total features: {stats['shape'][1]}")
    print("\nFeature Types:")
    print(f"- Numeric features: {len(column_types['numeric'])}")
    print(f"- Categorical features: {len(column_types['categorical'])}")
    print(f"- Datetime features: {len(column_types['datetime'])}")
    
    print("\nMissing Values:")
    for col, count in null_counts.items():
        if count > 0:
            percentage = (count / stats['shape'][0]) * 100
            print(f"- {col}: {count} ({percentage:.2f}%)")
    
    print("\nSample Records:")
    print(data.head(3))
    print("=" * 50)

if __name__ == "__main__":
    # Example usage
    data = load_data("marketing_campaign.csv")
    display_data_summary(data)