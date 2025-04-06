"""
Evaluation Module

This module contains functions for evaluating clustering algorithms using various metrics and techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_clustering(data: pd.DataFrame, cluster_columns: List[str]) -> pd.DataFrame:
    """
    Evaluate clustering results using multiple metrics.
    
    Args:
        data (pd.DataFrame): Dataset with cluster assignments
        cluster_columns (List[str]): List of columns containing cluster assignments
        
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each clustering method
    """
    logger.info(f"Evaluating clustering results for {len(cluster_columns)} methods")
    
    # Dictionary to store results
    results = {}
    
    for cluster_col in cluster_columns:
        # Ensure the column exists
        if cluster_col not in data.columns:
            logger.warning(f"Cluster column '{cluster_col}' not found in the dataset")
            continue
        
        # Extract cluster labels
        labels = data[cluster_col]
        
        # Create a copy of the dataset without cluster columns
        X = data.drop(columns=cluster_columns)
        
        try:
            # Calculate Silhouette Score
            silhouette = silhouette_score(X, labels)
            
            # Calculate Calinski-Harabasz Index
            calinski = calinski_harabasz_score(X, labels)
            
            # Calculate Davies-Bouldin Index
            davies = davies_bouldin_score(X, labels)
            
            # Store results
            results[cluster_col] = {
                'Silhouette Score': silhouette,
                'Calinski-Harabasz Index': calinski,
                'Davies-Bouldin Index': davies,
                'Number of Clusters': len(np.unique(labels))
            }
            
            logger.info(f"Evaluation for {cluster_col}:")
            logger.info(f"  Silhouette Score: {silhouette:.4f}")
            logger.info(f"  Calinski-Harabasz Index: {calinski:.4f}")
            logger.info(f"  Davies-Bouldin Index: {davies:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {cluster_col}: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    return results_df

def plot_evaluation_metrics(eval_results: pd.DataFrame) -> plt.Figure:
    """
    Plot evaluation metrics for visual comparison.
    
    Args:
        eval_results (pd.DataFrame): DataFrame with evaluation metrics
        
    Returns:
        plt.Figure: Figure object with the plot
    """
    # Create a 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Silhouette Score (higher is better)
    eval_results['Silhouette Score'].sort_values().plot(
        kind='barh', ax=axes[0], color='skyblue'
    )
    axes[0].set_title('Silhouette Score (higher is better)', fontsize=14)
    axes[0].set_xlabel('Score')
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Calinski-Harabasz Index (higher is better)
    eval_results['Calinski-Harabasz Index'].sort_values().plot(
        kind='barh', ax=axes[1], color='lightgreen'
    )
    axes[1].set_title('Calinski-Harabasz Index (higher is better)', fontsize=14)
    axes[1].set_xlabel('Score')
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Davies-Bouldin Index (lower is better)
    eval_results['Davies-Bouldin Index'].sort_values(ascending=False).plot(
        kind='barh', ax=axes[2], color='salmon'
    )
    axes[2].set_title('Davies-Bouldin Index (lower is better)', fontsize=14)
    axes[2].set_xlabel('Score')
    axes[2].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def analyze_clusters(data: pd.DataFrame, cluster_column: str, 
                    feature_columns: List[str]) -> pd.DataFrame:
    """
    Analyze clusters based on feature distributions.
    
    Args:
        data (pd.DataFrame): Dataset with cluster assignments
        cluster_column (str): Column containing cluster assignments
        feature_columns (List[str]): List of features to analyze
        
    Returns:
        pd.DataFrame: DataFrame with cluster statistics
    """
    logger.info(f"Analyzing clusters based on {len(feature_columns)} features")
    
    # Ensure the cluster column exists
    if cluster_column not in data.columns:
        logger.error(f"Cluster column '{cluster_column}' not found in the dataset")
        return pd.DataFrame()
    
    # Ensure feature columns exist
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        logger.warning(f"Features not found in the dataset: {missing_features}")
        feature_columns = [col for col in feature_columns if col in data.columns]
    
    if not feature_columns:
        logger.error("No valid features to analyze")
        return pd.DataFrame()
    
    # Group by cluster and calculate statistics
    cluster_analysis = data.groupby(cluster_column)[feature_columns].agg(
        ['mean', 'std', 'min', 'max', 'median', 'count']
    )
    
    # Reformat the MultiIndex columns
    cluster_analysis.columns = [f"{col}_{stat}" for col, stat in cluster_analysis.columns]
    
    # Calculate cluster sizes and proportions
    cluster_sizes = data[cluster_column].value_counts()
    cluster_proportions = cluster_sizes / len(data)
    
    # Add size and proportion columns
    cluster_analysis['Cluster_Size'] = cluster_sizes
    cluster_analysis['Cluster_Proportion'] = cluster_proportions
    
    return cluster_analysis

def plot_feature_distributions(data: pd.DataFrame, cluster_column: str, 
                              feature_columns: List[str], figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot feature distributions by cluster.
    
    Args:
        data (pd.DataFrame): Dataset with cluster assignments
        cluster_column (str): Column containing cluster assignments
        feature_columns (List[str]): List of features to plot
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: Figure object with the plots
    """
    # Ensure the cluster column exists
    if cluster_column not in data.columns:
        logger.error(f"Cluster column '{cluster_column}' not found in the dataset")
        return None
    
    # Ensure feature columns exist
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        logger.warning(f"Features not found in the dataset: {missing_features}")
        feature_columns = [col for col in feature_columns if col in data.columns]
    
    if not feature_columns:
        logger.error("No valid features to plot")
        return None
    
    # Determine layout
    n_features = len(feature_columns)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each feature
    for i, feature in enumerate(feature_columns):
        if i < len(axes):
            # Create box plot
            sns.boxplot(x=cluster_column, y=feature, data=data, ax=axes[i])
            axes[i].set_title(f'{feature} by Cluster')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel(feature)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def compare_cluster_stability(data_list: List[pd.DataFrame], 
                             cluster_columns: List[str], 
                             feature_columns: List[str]) -> pd.DataFrame:
    """
    Compare cluster stability across different datasets or preprocessing methods.
    
    Args:
        data_list (List[pd.DataFrame]): List of datasets with cluster assignments
        cluster_columns (List[str]): List of columns containing cluster assignments
        feature_columns (List[str]): List of features to analyze
        
    Returns:
        pd.DataFrame: DataFrame with stability analysis
    """
    logger.info(f"Comparing cluster stability across {len(data_list)} datasets")
    
    # Dictionary to store results
    results = {}
    
    # Iterate over datasets
    for i, data in enumerate(data_list):
        dataset_name = f"Dataset_{i+1}"
        logger.info(f"Analyzing {dataset_name}")
        
        # Iterate over cluster columns
        for cluster_col in cluster_columns:
            if cluster_col in data.columns:
                # Analyze clusters
                cluster_stats = analyze_clusters(data, cluster_col, feature_columns)
                
                # Store results
                key = f"{dataset_name}_{cluster_col}"
                results[key] = cluster_stats
            else:
                logger.warning(f"Cluster column '{cluster_col}' not found in {dataset_name}")
    
    # Calculate stability metrics
    stability_metrics = pd.DataFrame()
    
    if results:
        # Identify common features across all results
        common_features = []
        for stats_df in results.values():
            feature_stats = [col.split('_')[0] for col in stats_df.columns if '_mean' in col]
            if not common_features:
                common_features = feature_stats
            else:
                common_features = [f for f in common_features if f in feature_stats]
        
        logger.info(f"Found {len(common_features)} common features for stability analysis")
        
        # Calculate coefficient of variation for each feature's mean across datasets
        for feature in common_features:
            means = []
            for key, stats_df in results.items():
                mean_col = f"{feature}_mean"
                if mean_col in stats_df.columns:
                    means.extend(stats_df[mean_col].values)
            
            if means:
                # Calculate coefficient of variation (CV)
                cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else np.nan
                stability_metrics.loc[feature, 'Coefficient_of_Variation'] = cv
                stability_metrics.loc[feature, 'Mean'] = np.mean(means)
                stability_metrics.loc[feature, 'Std'] = np.std(means)
    
    return stability_metrics

def create_cluster_summary(data: pd.DataFrame, cluster_column: str, 
                         feature_columns: List[str]) -> Dict[str, str]:
    """
    Create a textual summary of cluster characteristics.
    
    Args:
        data (pd.DataFrame): Dataset with cluster assignments
        cluster_column (str): Column containing cluster assignments
        feature_columns (List[str]): List of features to include in the summary
        
    Returns:
        Dict[str, str]: Dictionary mapping cluster IDs to summaries
    """
    logger.info(f"Creating cluster summaries based on {len(feature_columns)} features")
    
    # Analyze clusters
    cluster_stats = analyze_clusters(data, cluster_column, feature_columns)
    
    # Dictionary to store cluster summaries
    summaries = {}
    
    # Calculate overall statistics for comparison
    overall_means = {}
    for feature in feature_columns:
        mean_col = f"{feature}_mean"
        if mean_col in cluster_stats.columns:
            overall_means[feature] = data[feature].mean()
    
    # Create summary for each cluster
    for cluster_id in cluster_stats.index:
        # Get cluster statistics
        stats = cluster_stats.loc[cluster_id]
        
        # Prepare summary
        summary = []
        
        # Add cluster size information
        cluster_size = stats['Cluster_Size']
        cluster_proportion = stats['Cluster_Proportion'] * 100
        summary.append(f"Cluster {cluster_id}: {cluster_size} customers ({cluster_proportion:.2f}% of total)")
        
        # Compare features to overall means
        distinctive_features = []
        for feature in feature_columns:
            mean_col = f"{feature}_mean"
            if mean_col in stats.index and feature in overall_means:
                # Calculate percent difference from overall mean
                cluster_mean = stats[mean_col]
                overall_mean = overall_means[feature]
                
                if overall_mean != 0:
                    percent_diff = ((cluster_mean - overall_mean) / abs(overall_mean)) * 100
                    
                    # Only include significant differences
                    if abs(percent_diff) >= 10:
                        direction = "higher" if percent_diff > 0 else "lower"
                        distinctive_features.append(
                            f"{feature}: {cluster_mean:.2f} ({abs(percent_diff):.0f}% {direction} than average)"
                        )
        
        # Add distinctive features to summary
        if distinctive_features:
            summary.append("Distinctive features:")
            summary.extend([f"- {feature}" for feature in distinctive_features])
        
        # Combine summary into a single string
        summaries[str(cluster_id)] = "\n".join(summary)
    
    return summaries

if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    import data_preprocessing as prep
    import clustering as cl
    
    # Load and preprocess data
    data = load_data("marketing_campaign.csv")
    data = prep.handle_missing_values(data, {'Income': 'mean'})
    data = prep.calculate_age(data, 'Year_Birth', reference_year=2014)
    
    # Apply different clustering methods
    data_kmeans = cl.apply_kmeans(data, n_clusters=3)
    data_hierarchical = cl.apply_hierarchical_clustering(data, n_clusters=3)
    
    # Evaluate clustering results
    eval_results = evaluate_clustering(
        data_kmeans, 
        ['Cluster_KMeans', 'Cluster_Hierarchical']
    )
    
    # Plot evaluation metrics
    plot_evaluation_metrics(eval_results)
    
    # Analyze clusters
    feature_columns = ['Age', 'Income', 'MntWines', 'MntMeatProducts', 'NumWebPurchases']
    cluster_analysis = analyze_clusters(data_kmeans, 'Cluster_KMeans', feature_columns)
    
    print(cluster_analysis)