"""
Clustering Module

This module contains functions for applying different clustering algorithms to the marketing campaign data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import skfuzzy as fuzz
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_optimal_clusters(data: pd.DataFrame, max_clusters: int = 10, 
                          method: str = 'elbow', 
                          random_state: int = 42) -> Tuple[plt.Figure, int]:
    """
    Find the optimal number of clusters using the elbow method or silhouette score.
    
    Args:
        data (pd.DataFrame): The preprocessed dataset
        max_clusters (int): Maximum number of clusters to consider
        method (str): Method to use ('elbow' or 'silhouette')
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[plt.Figure, int]: Figure object with the plot and the optimal number of clusters
    """
    from sklearn.metrics import silhouette_score
    
    # Setup
    K = range(2, max_clusters + 1)
    distortions = []
    silhouette_scores = []
    
    # Compute metrics for different cluster counts
    for k in K:
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        
        # Store distortion (inertia)
        distortions.append(kmeans.inertia_)
        
        # Calculate silhouette score if using that method
        if method == 'silhouette':
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if method == 'elbow':
        # Plot elbow curve
        ax.plot(K, distortions, 'bo-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Distortion (Inertia)')
        ax.set_title('Elbow Method for Determining Optimal k')
        
        # Find the optimal k using the elbow method
        # Calculate the rate of change in distortions
        deltas = np.diff(distortions)
        delta_deltas = np.diff(deltas)
        
        # The elbow is where the rate of change of the rate of change is maximum
        optimal_k = K[np.argmax(np.abs(delta_deltas)) + 1]
        
        # Highlight the optimal k
        ax.axvline(x=optimal_k, color='red', linestyle='--')
        ax.text(optimal_k + 0.1, np.mean(distortions), f'Optimal k = {optimal_k}', color='red')
        
    elif method == 'silhouette':
        # Plot silhouette scores
        ax.plot(K, silhouette_scores, 'bo-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Method for Determining Optimal k')
        
        # Find the optimal k using the silhouette method
        optimal_k = K[np.argmax(silhouette_scores)]
        
        # Highlight the optimal k
        ax.axvline(x=optimal_k, color='red', linestyle='--')
        ax.text(optimal_k + 0.1, np.mean(silhouette_scores), f'Optimal k = {optimal_k}', color='red')
    
    else:
        logger.error(f"Unknown method: {method}. Use 'elbow' or 'silhouette'.")
        optimal_k = 3  # Default to 3 clusters if method is unknown
    
    logger.info(f"Optimal number of clusters using {method} method: {optimal_k}")
    
    return fig, optimal_k

def apply_kmeans(data: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Apply K-means clustering to the dataset.
    
    Args:
        data (pd.DataFrame): The preprocessed dataset
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Original dataset with cluster assignments
    """
    # Create a copy of the dataset
    df = data.copy()
    
    logger.info(f"Applying K-means clustering with {n_clusters} clusters")
    
    # Create and fit the K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(df)
    
    # Add cluster assignments to the dataset
    df['Cluster_KMeans'] = clusters
    
    # Log cluster distribution
    cluster_counts = df['Cluster_KMeans'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"Cluster {cluster_id}: {count} samples ({percentage:.2f}%)")
    
    # Return the dataset with cluster assignments
    return df

def apply_hierarchical_clustering(data: pd.DataFrame, n_clusters: int = 3, 
                                 linkage: str = 'ward', 
                                 plot_dendrogram: bool = True) -> pd.DataFrame:
    """
    Apply hierarchical clustering to the dataset.
    
    Args:
        data (pd.DataFrame): The preprocessed dataset
        n_clusters (int): Number of clusters
        linkage (str): Linkage method ('ward', 'complete', 'average', 'single')
        plot_dendrogram (bool): Whether to plot the dendrogram
        
    Returns:
        pd.DataFrame: Original dataset with cluster assignments
    """
    # Create a copy of the dataset
    df = data.copy()
    
    logger.info(f"Applying hierarchical clustering with {n_clusters} clusters using {linkage} linkage")
    
    # Plot dendrogram if requested
    if plot_dendrogram:
        plt.figure(figsize=(12, 8))
        dendrogram = sch.dendrogram(sch.linkage(df, method=linkage))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Euclidean Distance')
        plt.show()
    
    # Create and fit the hierarchical clustering model
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = hierarchical.fit_predict(df)
    
    # Add cluster assignments to the dataset
    df['Cluster_Hierarchical'] = clusters
    
    # Log cluster distribution
    cluster_counts = df['Cluster_Hierarchical'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"Cluster {cluster_id}: {count} samples ({percentage:.2f}%)")
    
    # Return the dataset with cluster assignments
    return df

def apply_fuzzy_cmeans(data: pd.DataFrame, n_clusters: int = 3, m: float = 2.0, 
                      error: float = 0.005, max_iter: int = 1000) -> pd.DataFrame:
    """
    Apply Fuzzy C-means clustering to the dataset.
    
    Args:
        data (pd.DataFrame): The preprocessed dataset
        n_clusters (int): Number of clusters
        m (float): Fuzziness parameter
        error (float): Error threshold for stopping iterations
        max_iter (int): Maximum number of iterations
        
    Returns:
        pd.DataFrame: Original dataset with cluster assignments and membership degrees
    """
    # Create a copy of the dataset
    df = data.copy()
    
    logger.info(f"Applying Fuzzy C-means clustering with {n_clusters} clusters")
    
    # Transpose data for scikit-fuzzy compatibility
    data_array = np.array(df.T)
    
    # Apply Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_array, n_clusters, m, error=error, maxiter=max_iter, init=None
    )
    
    # Get cluster assignments (hard clustering based on highest membership)
    clusters = np.argmax(u, axis=0)
    
    # Add cluster assignments to the dataset
    df['Cluster_FCM'] = clusters
    
    # Add membership degrees for each cluster
    for i in range(n_clusters):
        df[f'FCM_Membership_{i}'] = u[i]
    
    # Log cluster distribution
    cluster_counts = df['Cluster_FCM'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"Cluster {cluster_id}: {count} samples ({percentage:.2f}%)")
    
    # Log fuzzy partition coefficient
    logger.info(f"Fuzzy Partition Coefficient: {fpc}")
    
    # Return the dataset with cluster assignments and memberships
    return df

def apply_all_clustering_methods(data: pd.DataFrame, n_clusters: int = 3, 
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Apply all clustering methods to the dataset.
    
    Args:
        data (pd.DataFrame): The preprocessed dataset
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping method names to datasets with cluster assignments
    """
    logger.info(f"Applying all clustering methods with {n_clusters} clusters")
    
    # Dictionary to store results
    results = {}
    
    # Apply K-means
    logger.info("Applying K-means clustering")
    results['kmeans'] = apply_kmeans(data, n_clusters, random_state)
    
    # Apply hierarchical clustering with different linkage methods
    for linkage in ['ward', 'complete', 'average', 'single']:
        logger.info(f"Applying hierarchical clustering with {linkage} linkage")
        results[f'hierarchical_{linkage}'] = apply_hierarchical_clustering(
            data, n_clusters, linkage, plot_dendrogram=False
        )
    
    # Apply Fuzzy C-means
    logger.info("Applying Fuzzy C-means clustering")
    results['fuzzy_cmeans'] = apply_fuzzy_cmeans(data, n_clusters)
    
    return results

def plot_cluster_scatter(data: pd.DataFrame, cluster_column: str, 
                        x_var: str, y_var: str,
                        title: Optional[str] = None) -> plt.Figure:
    """
    Create a scatter plot showing the clusters.
    
    Args:
        data (pd.DataFrame): Dataset with cluster assignments
        cluster_column (str): Column containing cluster assignments
        x_var (str): Column to plot on the x-axis
        y_var (str): Column to plot on the y-axis
        title (str, optional): Plot title
        
    Returns:
        plt.Figure: Figure object with the plot
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        data[x_var], 
        data[y_var], 
        c=data[cluster_column], 
        cmap='viridis', 
        s=50, 
        alpha=0.7,
        edgecolors='k'
    )
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(),
                        title="Clusters")
    ax.add_artist(legend1)
    
    # Set labels and title
    ax.set_xlabel(x_var, fontsize=12)
    ax.set_ylabel(y_var, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Clusters by {x_var} and {y_var}', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def visualize_cluster_profiles(data: pd.DataFrame, cluster_column: str, 
                              features: List[str]) -> plt.Figure:
    """
    Create a visualization of cluster profiles using key features.
    
    Args:
        data (pd.DataFrame): Dataset with cluster assignments
        cluster_column (str): Column containing cluster assignments
        features (List[str]): List of features to include in the profiles
        
    Returns:
        plt.Figure: Figure object with the plot
    """
    # Validate inputs
    if cluster_column not in data.columns:
        logger.error(f"Cluster column '{cluster_column}' not found in the dataset")
        return None
    
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        logger.error(f"Features not found in the dataset: {missing_features}")
        return None
    
    # Get cluster statistics
    cluster_stats = {}
    clusters = sorted(data[cluster_column].unique())
    
    for cluster in clusters:
        cluster_data = data[data[cluster_column] == cluster]
        
        # Calculate statistics for each feature
        feature_stats = {}
        for feature in features:
            feature_stats[feature] = {
                'mean': cluster_data[feature].mean(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max(),
                'median': cluster_data[feature].median()
            }
        
        cluster_stats[cluster] = feature_stats
    
    # Create radar chart for cluster profiles
    # Set up the figure
    fig = plt.figure(figsize=(15, 10))
    
    # Number of variables
    N = len(features)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Number of clusters
    num_clusters = len(clusters)
    
    # Create subplots
    for i, cluster in enumerate(clusters):
        ax = plt.subplot(2, (num_clusters + 1) // 2, i + 1, polar=True)
        
        # Get standardized means for the current cluster
        means = []
        for feature in features:
            # Calculate min and max across all clusters for normalization
            feature_min = min(cluster_stats[c][feature]['mean'] for c in clusters)
            feature_max = max(cluster_stats[c][feature]['mean'] for c in clusters)
            
            # Normalize to [0, 1]
            if feature_max > feature_min:
                normalized_mean = (cluster_stats[cluster][feature]['mean'] - feature_min) / (feature_max - feature_min)
            else:
                normalized_mean = 0.5  # If min == max, set to middle
            
            means.append(normalized_mean)
        
        # Close the loop
        means += means[:1]
        
        # Plot
        ax.plot(angles, means, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
        ax.fill(angles, means, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=8)
        
        # Set title
        ax.set_title(f'Cluster {cluster} Profile', fontsize=11)
        
        # Remove radial labels
        ax.set_yticklabels([])
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    import data_preprocessing as prep
    
    # Load and preprocess data
    data = load_data("marketing_campaign.csv")
    data = prep.handle_missing_values(data, {'Income': 'mean'})
    data = prep.calculate_age(data, 'Year_Birth', reference_year=2014)
    data = prep.create_date_features(data, 'Dt_Customer')
    data = prep.create_expenditure_features(
        data, 
        ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
        'Total_Spent'
    )
    
    # Drop unnecessary columns for clustering
    cols_to_drop = ['ID', 'Dt_Customer', 'Year_Birth', 'Z_CostContact', 'Z_Revenue']
    data = prep.drop_redundant_features(data, cols_to_drop)
    
    # Encode categorical variables and scale numeric features
    data = prep.encode_categorical_variables(data)
    data = prep.scale_features(data, exclude_binary=True)
    
    # Find optimal number of clusters
    _, optimal_k = find_optimal_clusters(data, max_clusters=10, method='elbow')
    
    # Apply K-means clustering with optimal k
    clustered_data = apply_kmeans(data, n_clusters=optimal_k)
    
    # Visualize clusters
    plot_cluster_scatter(clustered_data, 'Cluster_KMeans', 'Income', 'Total_Spent', 
                         'Customer Segments by Income and Total Spending')
    
    # Create cluster profiles
    key_features = ['Age', 'Income', 'Total_Spent', 'Children_Total', 'MntWines', 'NumWebPurchases']
    visualize_cluster_profiles(clustered_data, 'Cluster_KMeans', key_features)