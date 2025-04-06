"""
Data Preprocessing Module

This module contains functions for cleaning data, handling missing values,
transforming variables, and performing feature engineering on the marketing campaign data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_missing_values(data: pd.DataFrame, strategy: Dict[str, str], 
                          visualize: bool = False) -> pd.DataFrame:
    """
    Handle missing values in the dataset using the specified strategies.
    
    Args:
        data (pd.DataFrame): The dataset to process
        strategy (Dict[str, str]): Dictionary mapping column names to imputation strategies
                                   ('mean', 'median', 'mode', 'drop', 'regression')
        visualize (bool): Whether to visualize distributions before and after imputation
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    # Create a copy of the dataset
    df = data.copy()
    
    # Get missing value counts
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    if not columns_with_missing:
        logger.info("No missing values found in the dataset")
        return df
    
    logger.info(f"Handling missing values in {len(columns_with_missing)} columns")
    
    for column in columns_with_missing:
        column_strategy = strategy.get(column, 'drop')
        missing_count = missing_counts[column]
        missing_percentage = (missing_count / len(df)) * 100
        
        logger.info(f"Column '{column}' has {missing_count} missing values "
                   f"({missing_percentage:.2f}%), using '{column_strategy}' strategy")
        
        # Visualize distribution before imputation if requested
        if visualize:
            plt.figure(figsize=(12, 5))
            
            # Before imputation
            plt.subplot(1, 2, 1)
            if df[column].dtype in ['float64', 'int64']:
                sns.histplot(df[column].dropna(), kde=True)
                plt.title(f'{column} Distribution (Before Imputation)')
            else:
                sns.countplot(x=df[column].dropna())
                plt.title(f'{column} Counts (Before Imputation)')
                plt.xticks(rotation=45)
        
        # Apply the imputation strategy
        if column_strategy == 'mean':
            # Only applicable for numeric data
            if df[column].dtype not in ['float64', 'int64']:
                logger.warning(f"Cannot apply 'mean' strategy to non-numeric column '{column}'. Using mode instead.")
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)
                
        elif column_strategy == 'median':
            # Only applicable for numeric data
            if df[column].dtype not in ['float64', 'int64']:
                logger.warning(f"Cannot apply 'median' strategy to non-numeric column '{column}'. Using mode instead.")
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)
                
        elif column_strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
            
        elif column_strategy == 'drop':
            df.dropna(subset=[column], inplace=True)
            logger.info(f"Dropped {missing_count} rows with missing values in '{column}'")
            
        elif column_strategy == 'regression':
            # Only applicable for numeric data
            if df[column].dtype not in ['float64', 'int64']:
                logger.warning(f"Cannot apply 'regression' strategy to non-numeric column '{column}'. Using mode instead.")
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df = impute_by_regression(df, column)
                
        else:
            raise ValueError(f"Unknown strategy: {column_strategy}. "
                            "Use 'mean', 'median', 'mode', 'drop', or 'regression'.")
        
        # Visualize distribution after imputation if requested
        if visualize and column_strategy != 'drop':
            # After imputation
            plt.subplot(1, 2, 2)
            if df[column].dtype in ['float64', 'int64']:
                sns.histplot(df[column], kde=True)
                plt.title(f'{column} Distribution (After {column_strategy.capitalize()} Imputation)')
            else:
                sns.countplot(x=df[column])
                plt.title(f'{column} Counts (After {column_strategy.capitalize()} Imputation)')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    return df

def impute_by_regression(data: pd.DataFrame, target_column: str, 
                        predictor_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Impute missing values in a column using linear regression on other columns.
    
    Args:
        data (pd.DataFrame): The dataset to process
        target_column (str): The column containing missing values to impute
        predictor_columns (List[str], optional): List of columns to use as predictors.
                                               If None, use all numeric columns except the target.
        
    Returns:
        pd.DataFrame: Dataset with missing values imputed
    """
    df = data.copy()
    
    # Identify rows with missing and non-missing values
    missing_mask = df[target_column].isnull()
    
    if missing_mask.sum() == 0:
        logger.info(f"No missing values in column '{target_column}'")
        return df
    
    # If predictor columns are not specified, use all numeric columns except the target
    if predictor_columns is None:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        predictor_columns = [col for col in numeric_columns if col != target_column]
    
    # Ensure all predictor columns are present and contain no missing values in the training data
    valid_predictor_columns = []
    for col in predictor_columns:
        if col in df.columns:
            # Check if the column has any missing values in the training data
            if df.loc[~missing_mask, col].isnull().sum() > 0:
                logger.warning(f"Predictor column '{col}' contains missing values. Skipping.")
            else:
                valid_predictor_columns.append(col)
        else:
            logger.warning(f"Predictor column '{col}' not found in the dataset. Skipping.")
    
    if not valid_predictor_columns:
        logger.error("No valid predictor columns available for regression imputation")
        # Fall back to mean imputation
        df[target_column].fillna(df[target_column].mean(), inplace=True)
        return df
    
    # Split data into training (non-missing target) and prediction (missing target) sets
    X_train = df.loc[~missing_mask, valid_predictor_columns]
    y_train = df.loc[~missing_mask, target_column]
    X_predict = df.loc[missing_mask, valid_predictor_columns]
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict missing values
    y_predict = model.predict(X_predict)
    
    # Impute the predicted values
    df.loc[missing_mask, target_column] = y_predict
    
    logger.info(f"Imputed {missing_mask.sum()} missing values in '{target_column}' using regression")
    logger.info(f"R² score of regression model: {model.score(X_train, y_train):.4f}")
    
    return df

def create_date_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extract features from a date column.
    
    Args:
        data (pd.DataFrame): The dataset to process
        date_column (str): The column containing dates
        
    Returns:
        pd.DataFrame: Dataset with new date-based features
    """
    df = data.copy()
    
    # Ensure the column is in datetime format
    if df[date_column].dtype != 'datetime64[ns]':
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            logger.error(f"Error converting '{date_column}' to datetime: {str(e)}")
            return df
    
    logger.info(f"Creating date features from column '{date_column}'")
    
    # Extract the most recent date in the dataset
    most_recent_date = df[date_column].max()
    oldest_date = df[date_column].min()
    
    # Calculate days since the most recent date
    df[f'Days_Since_{date_column}'] = (most_recent_date - df[date_column]).dt.days
    
    # Extract year, month, and day
    df[f'{date_column}_Year'] = df[date_column].dt.year
    df[f'{date_column}_Month'] = df[date_column].dt.month
    df[f'{date_column}_Day'] = df[date_column].dt.day
    
    # Extract day of week (0 = Monday, 6 = Sunday)
    df[f'{date_column}_DayOfWeek'] = df[date_column].dt.dayofweek
    
    # Calculate quarter of year
    df[f'{date_column}_Quarter'] = df[date_column].dt.quarter
    
    logger.info(f"Date range: {oldest_date} to {most_recent_date}")
    logger.info(f"Added 5 new date-based features from '{date_column}'")
    
    return df

def calculate_age(data: pd.DataFrame, birth_year_column: str, reference_year: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate age based on birth year.
    
    Args:
        data (pd.DataFrame): The dataset to process
        birth_year_column (str): The column containing birth years
        reference_year (int, optional): The reference year for age calculation. 
                                     If None, current year is used.
        
    Returns:
        pd.DataFrame: Dataset with new 'Age' column
    """
    df = data.copy()
    
    # Use current year if reference year is not provided
    if reference_year is None:
        reference_year = pd.to_datetime('today').year
    
    logger.info(f"Calculating age using {reference_year} as the reference year")
    
    # Calculate age
    df['Age'] = reference_year - df[birth_year_column]
    
    # Log age statistics
    logger.info(f"Age statistics: min={df['Age'].min()}, max={df['Age'].max()}, mean={df['Age'].mean():.1f}")
    
    return df

def create_expenditure_features(data: pd.DataFrame, amount_columns: List[str], 
                              new_column_name: str = 'Total_Expenditure') -> pd.DataFrame:
    """
    Calculate total expenditure by summing specified amount columns.
    
    Args:
        data (pd.DataFrame): The dataset to process
        amount_columns (List[str]): The columns containing amounts to sum
        new_column_name (str): The name for the new column
        
    Returns:
        pd.DataFrame: Dataset with new expenditure column
    """
    df = data.copy()
    
    # Ensure all amount columns exist
    missing_columns = [col for col in amount_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Columns not found: {missing_columns}")
        amount_columns = [col for col in amount_columns if col in df.columns]
    
    if not amount_columns:
        logger.error("No valid amount columns to sum")
        return df
    
    logger.info(f"Calculating {new_column_name} by summing {len(amount_columns)} columns")
    
    # Calculate total expenditure
    df[new_column_name] = df[amount_columns].sum(axis=1)
    
    # Log expenditure statistics
    logger.info(f"{new_column_name} statistics: min={df[new_column_name].min()}, "
               f"max={df[new_column_name].max()}, mean={df[new_column_name].mean():.2f}")
    
    return df

def simplify_categories(data: pd.DataFrame, column: str, 
                       mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Simplify categories in a categorical column using a mapping dictionary.
    
    Args:
        data (pd.DataFrame): The dataset to process
        column (str): The column to modify
        mapping (Dict[str, str]): Dictionary mapping original categories to new categories
        
    Returns:
        pd.DataFrame: Dataset with simplified categories
    """
    df = data.copy()
    
    # Ensure the column exists
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in the dataset")
        return df
    
    # Get original category counts
    original_categories = df[column].value_counts()
    logger.info(f"Original categories in '{column}': {len(original_categories)}")
    
    # Apply the mapping
    new_column_name = f"{column}_Simplified"
    df[new_column_name] = df[column].replace(mapping)
    
    # Get new category counts
    new_categories = df[new_column_name].value_counts()
    logger.info(f"Simplified categories in '{new_column_name}': {len(new_categories)}")
    
    # Log the mapping for each category
    for orig_cat, new_cat in mapping.items():
        # Check if the original category exists in the data
        if orig_cat in original_categories:
            count = original_categories[orig_cat]
            logger.info(f"Mapped '{orig_cat}' ({count} instances) to '{new_cat}'")
    
    return df

def encode_categorical_variables(data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encode categorical variables using Label Encoding.
    
    Args:
        data (pd.DataFrame): The dataset to process
        columns (List[str], optional): List of categorical columns to encode.
                                  If None, all object columns are encoded.
        
    Returns:
        pd.DataFrame: Dataset with encoded categorical variables
    """
    df = data.copy()
    
    # If no columns specified, use all object columns
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not columns:
        logger.info("No categorical columns to encode")
        return df
    
    logger.info(f"Encoding {len(columns)} categorical columns")
    
    # Create a label encoder
    le = LabelEncoder()
    
    # Encode each categorical column
    for col in columns:
        if col in df.columns:
            # Get original categories and their counts
            categories = df[col].value_counts()
            
            # Apply label encoding
            df[col] = df[col].astype(str)  # Ensure all values are strings
            df[col] = le.fit_transform(df[col])
            
            logger.info(f"Encoded '{col}' ({len(categories)} categories) to numeric values 0-{len(categories)-1}")
        else:
            logger.warning(f"Column '{col}' not found in the dataset")
    
    return df

def scale_features(data: pd.DataFrame, columns: Optional[List[str]] = None, 
                  exclude_binary: bool = True) -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        data (pd.DataFrame): The dataset to process
        columns (List[str], optional): List of columns to scale.
                                    If None, all numeric columns are scaled.
        exclude_binary (bool): Whether to exclude binary columns (0-1) from scaling
        
    Returns:
        pd.DataFrame: Dataset with scaled features
    """
    df = data.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not columns:
        logger.info("No numeric columns to scale")
        return df
    
    # Identify binary columns if exclude_binary is True
    binary_columns = []
    if exclude_binary:
        for col in columns:
            if set(df[col].unique()).issubset({0, 1, 0.0, 1.0}):
                binary_columns.append(col)
        
        # Remove binary columns from the list of columns to scale
        columns = [col for col in columns if col not in binary_columns]
        logger.info(f"Excluded {len(binary_columns)} binary columns from scaling")
    
    if not columns:
        logger.info("No columns left to scale after excluding binary columns")
        return df
    
    logger.info(f"Scaling {len(columns)} numeric columns")
    
    # Create a standard scaler
    scaler = StandardScaler()
    
    # Scale the selected columns
    df[columns] = scaler.fit_transform(df[columns])
    
    return df

def handle_outliers(data: pd.DataFrame, columns: Optional[List[str]] = None, 
                   method: str = 'clip', threshold: float = 3.0) -> pd.DataFrame:
    """
    Handle outliers in numeric columns.
    
    Args:
        data (pd.DataFrame): The dataset to process
        columns (List[str], optional): List of columns to process.
                                    If None, all numeric columns are processed.
        method (str): Method to handle outliers ('clip', 'remove', or 'winsorize')
        threshold (float): Threshold for outlier detection (number of standard deviations or IQR multiplier)
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    df = data.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not columns:
        logger.info("No numeric columns to handle outliers")
        return df
    
    logger.info(f"Handling outliers in {len(columns)} columns using {method} method")
    
    for col in columns:
        # Calculate bounds for outliers
        if method == 'clip' or method == 'winsorize':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Count outliers
            outliers_lower = (df[col] < lower_bound).sum()
            outliers_upper = (df[col] > upper_bound).sum()
            total_outliers = outliers_lower + outliers_upper
            
            if total_outliers > 0:
                logger.info(f"Found {total_outliers} outliers in '{col}' ({outliers_lower} below, {outliers_upper} above)")
                
                # Handle outliers
                if method == 'clip':
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Clipped outliers in '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                elif method == 'winsorize':
                    # Winsorizing is like clipping, but we set to the percentile value instead of a fixed threshold
                    lower_percentile = 1  # 1st percentile
                    upper_percentile = 99  # 99th percentile
                    lower_bound_win = df[col].quantile(lower_percentile / 100)
                    upper_bound_win = df[col].quantile(upper_percentile / 100)
                    df[col] = df[col].clip(lower=lower_bound_win, upper=upper_bound_win)
                    logger.info(f"Winsorized outliers in '{col}' to [{lower_bound_win:.2f}, {upper_bound_win:.2f}]")
            else:
                logger.info(f"No outliers found in '{col}' using threshold of {threshold} standard deviations")
                
        elif method == 'remove':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Identify outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"Removing {outlier_count} outliers in '{col}'")
                df = df[~outliers]
        
        else:
            logger.warning(f"Unknown method: {method}. Using 'clip' instead.")
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def create_family_features(data: pd.DataFrame, kids_column: str, teens_column: str, 
                          living_status_column: str) -> pd.DataFrame:
    """
    Create family-related features such as total children and family size.
    
    Args:
        data (pd.DataFrame): The dataset to process
        kids_column (str): Column with number of young children
        teens_column (str): Column with number of teenagers
        living_status_column (str): Column with living status ('Alone' or 'Partner')
        
    Returns:
        pd.DataFrame: Dataset with new family features
    """
    df = data.copy()
    
    # Ensure all required columns exist
    required_columns = [kids_column, teens_column, living_status_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Required columns not found: {missing_columns}")
        return df
    
    logger.info("Creating family-related features")
    
    # Total children
    df['Children_Total'] = df[kids_column] + df[teens_column]
    
    # Is parent indicator
    df['Is_Parent'] = np.where(df['Children_Total'] > 0, 1, 0)
    
    # Family size
    # Convert living status to numeric (1 for alone, 2 for partner)
    if df[living_status_column].dtype == 'object':
        df['Family_Size'] = df[living_status_column].replace({'Alone': 1, 'Partner': 2}) + df['Children_Total']
    else:
        logger.warning(f"Column '{living_status_column}' is not categorical, assuming it's already numeric")
        df['Family_Size'] = df[living_status_column] + df['Children_Total']
    
    # Log statistics
    logger.info(f"Children_Total statistics: mean={df['Children_Total'].mean():.2f}, max={df['Children_Total'].max()}")
    logger.info(f"Is_Parent distribution: {df['Is_Parent'].value_counts(normalize=True).to_dict()}")
    logger.info(f"Family_Size statistics: mean={df['Family_Size'].mean():.2f}, max={df['Family_Size'].max()}")
    
    return df

def drop_redundant_features(data: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop redundant or unnecessary features.
    
    Args:
        data (pd.DataFrame): The dataset to process
        columns_to_drop (List[str]): List of columns to drop
        
    Returns:
        pd.DataFrame: Dataset with specified columns dropped
    """
    df = data.copy()
    
    # Find columns that exist in the dataset
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    if not existing_columns:
        logger.info("No columns to drop")
        return df
    
    logger.info(f"Dropping {len(existing_columns)} redundant features")
    
    # Drop the columns
    df.drop(existing_columns, axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    
    data = load_data("marketing_campaign.csv")
    
    # Example preprocessing
    data_cleaned = handle_missing_values(data, {'Income': 'mean'}, visualize=True)
    data_with_age = calculate_age(data_cleaned, 'Year_Birth', reference_year=2014)
    data_with_dates = create_date_features(data_with_age, 'Dt_Customer')
    data_with_expenditure = create_expenditure_features(
        data_with_dates, 
        ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
        'Total_Spent'
    )
    
    # Simplify marital status categories
    marital_mapping = {
        "Married": "Partner", 
        "Together": "Partner", 
        "Absurd": "Alone", 
        "Widow": "Alone", 
        "YOLO": "Alone", 
        "Divorced": "Alone", 
        "Single": "Alone"
    }
    data_with_simple_categories = simplify_categories(data_with_expenditure, 'Marital_Status', marital_mapping)
    
    # Create family features
    data_with_family = create_family_features(
        data_with_simple_categories, 
        'Kidhome', 
        'Teenhome', 
        'Marital_Status_Simplified'
    )
    
    # Encode categorical variables
    data_encoded = encode_categorical_variables(data_with_family)
    
    # Scale features
    data_scaled = scale_features(data_encoded, exclude_binary=True)
    
    # Handle outliers
    data_final = handle_outliers(data_scaled, ['Income', 'Age'], method='clip')
    
    print(data_final.head())