"""
Exploratory Data Analysis Module

This module contains functions for exploring and visualizing the marketing campaign data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_numeric_distributions(data: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Plot histograms for numeric variables to visualize their distributions.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        columns (List[str], optional): List of columns to plot. If None, all numeric columns are used.
    """
    if columns is None:
        columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    logger.info(f"Plotting distributions for {len(columns)} numeric variables")
    
    # Set plot style
    sns.set(style="whitegrid")
    
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True, color='steelblue')
        plt.title(f'Distribution of {col}', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_categorical_distributions(data: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Plot bar charts for categorical variables to visualize their distributions.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        columns (List[str], optional): List of columns to plot. If None, all object columns are used.
    """
    if columns is None:
        columns = data.select_dtypes(include=['object']).columns
    
    logger.info(f"Plotting distributions for {len(columns)} categorical variables")
    
    # Set plot style
    sns.set(style="whitegrid")
    
    for col in columns:
        plt.figure(figsize=(10, 6))
        value_counts = data[col].value_counts().sort_values(ascending=False)
        
        # If there are too many categories, limit to top 10
        if len(value_counts) > 10:
            value_counts = value_counts.head(10)
            logger.info(f"Limiting {col} to top 10 categories (out of {len(data[col].unique())})")
        
        sns.barplot(x=value_counts.values, y=value_counts.index, palette='viridis')
        plt.title(f'Distribution of {col}', fontsize=15)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Print category counts for reference
        print(f"\nCategory Counts for '{col}':")
        print(value_counts)

def plot_correlation_matrix(data: pd.DataFrame, method: str = 'pearson', threshold: float = 0.0) -> pd.DataFrame:
    """
    Plot a correlation matrix for numeric variables and identify high correlations.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        method (str): Correlation method ('pearson', 'spearman', or 'kendall')
        threshold (float): Correlation threshold to highlight (absolute value)
        
    Returns:
        pd.DataFrame: The correlation matrix
    """
    logger.info(f"Calculating {method} correlation matrix")
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr(method=method)
    
    # Plot the matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)
    
    plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Identify high correlations
    if threshold > 0:
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_correlations:
            print(f"\nHigh Correlations (|r| > {threshold}):")
            for var1, var2, corr in high_correlations:
                print(f"- {var1} and {var2}: {corr:.2f}")
        else:
            print(f"\nNo variable pairs have correlation above {threshold}.")
    
    return correlation_matrix

def plot_relationship(data: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> None:
    """
    Plot the relationship between two variables, with an optional categorical variable for coloring.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str, optional): Column name for coloring the points
    """
    plt.figure(figsize=(10, 6))
    
    if hue is None:
        sns.scatterplot(x=x, y=y, data=data, alpha=0.6)
    else:
        sns.scatterplot(x=x, y=y, hue=hue, data=data, alpha=0.6)
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Relationship between {x} and {y}', fontsize=15)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_comparative_boxplots(data: pd.DataFrame, numeric_col: str, categorical_col: str) -> None:
    """
    Create box plots to compare a numeric variable across categories.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        numeric_col (str): Name of the numeric column
        categorical_col (str): Name of the categorical column
    """
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x=categorical_col, y=numeric_col, data=data, palette='viridis')
    plt.title(f'{numeric_col} by {categorical_col}', fontsize=15)
    plt.xlabel(categorical_col, fontsize=12)
    plt.ylabel(numeric_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary statistics for {numeric_col} by {categorical_col}:")
    print(data.groupby(categorical_col)[numeric_col].describe())

def detect_outliers(data: pd.DataFrame, columns: Optional[List[str]] = None, 
                    method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect outliers in numeric columns using various methods.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        columns (List[str], optional): List of columns to check. If None, all numeric columns are used.
        method (str): Method to use ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        
    Returns:
        Dict[str, List[int]]: Dictionary mapping column names to lists of outlier indices
    """
    if columns is None:
        columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    logger.info(f"Detecting outliers in {len(columns)} columns using {method} method")
    
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
            
        elif method == 'zscore':
            z_scores = (data[col] - data[col].mean()) / data[col].std()
            outlier_indices = data[abs(z_scores) > threshold].index.tolist()
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
        
        if outlier_indices:
            outliers[col] = outlier_indices
            percentage = (len(outlier_indices) / len(data)) * 100
            logger.info(f"Found {len(outlier_indices)} outliers ({percentage:.2f}%) in column {col}")
    
    return outliers

def plot_pairplot(data: pd.DataFrame, columns: List[str], hue: Optional[str] = None) -> None:
    """
    Create a pairplot to visualize relationships between multiple variables.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        columns (List[str]): List of columns to include in the pairplot
        hue (str, optional): Column name for coloring the points
    """
    logger.info(f"Creating pairplot with {len(columns)} variables")
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(data[columns + ([hue] if hue else [])], 
                 hue=hue, 
                 corner=True, 
                 plot_kws={'alpha': 0.6})
    plt.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    
    data = load_data("marketing_campaign.csv")
    
    # Example exploratory analysis
    plot_numeric_distributions(data, ['Income', 'Age', 'Recency'])
    plot_categorical_distributions(data, ['Education', 'Marital_Status'])
    plot_correlation_matrix(data, threshold=0.7)
    plot_relationship(data, 'Income', 'MntWines', 'Education')