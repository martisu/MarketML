"""
Main Module for Marketing Campaign Analysis

This is the main orchestration script for the marketing campaign portfolio project.
It combines all the modules to perform a complete analysis workflow.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from typing import List, Dict, Tuple, Optional

# Import project modules
from src import data_loader as dl
from src import eda
from src import preprocessing as prep
from src import clustering as cl
from src import evaluation as eval
from src import visualization as viz
from src import utils

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("marketing_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Marketing Campaign Clustering Analysis')
    
    parser.add_argument('--data', type=str, default='marketing_campaign.csv',
                        help='Path to the input CSV file')
    
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for results and visualizations')
    
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='Number of clusters to create')
    
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                        choices=['kmeans', 'hierarchical', 'fuzzy'],
                        help='Clustering method to use')
    
    parser.add_argument('--evaluation_method', type=str, default='all',
                        choices=['silhouette', 'calinski', 'davies', 'all'],
                        help='Evaluation method for clusters')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def run_analysis(args):
    """
    Run the complete analysis workflow based on the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Start timing
    start_time = time.time()
    
    # Create output directory
    output_dir = utils.create_output_directory(args.output)
    
    logger.info("=" * 80)
    logger.info("STARTING MARKETING CAMPAIGN CLUSTERING ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.data}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Clustering method: {args.clustering_method}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    logger.info(f"Random state: {args.random_state}")
    logger.info("=" * 80)
    
    # STEP 1: Load Data
    logger.info("STEP 1: Loading data")
    data = dl.load_data(args.data)
    dl.display_data_summary(data)
    
    # Store initial data stats
    data_overview = {
        "total_records": len(data),
        "total_features": len(data.columns),
        "numeric_features": len(data.select_dtypes(include=['float64', 'int64']).columns),
        "categorical_features": len(data.select_dtypes(include=['object']).columns),
        "missing_values": data.isnull().sum().sum()
    }
    
    # STEP 2: Exploratory Data Analysis
    if args.visualize:
        logger.info("STEP 2: Performing exploratory data analysis")
        # Set visualization style
        viz.set_visualization_style()
        
        # Plot distributions for selected numeric variables
        numeric_columns = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'NumWebPurchases', 'NumCatalogPurchases']
        eda.plot_numeric_distributions(data, numeric_columns)
        
        # Plot distributions for categorical variables
        categorical_columns = ['Education', 'Marital_Status']
        eda.plot_categorical_distributions(data, categorical_columns)
        
        # Plot correlation matrix
        correlation_matrix = eda.plot_correlation_matrix(data, method='pearson', threshold=0.7)
        
        # Plot relationships between key variables
        eda.plot_relationship(data, 'Income', 'MntWines', 'Education')
        eda.plot_relationship(data, 'Age', 'NumWebPurchases')
        
        # Check for outliers
        outliers = eda.detect_outliers(data, ['Income', 'MntWines'], method='iqr')
    else:
        logger.info("STEP 2: Skipping exploratory visualizations (use --visualize to enable)")
    
    # STEP 3: Data Preprocessing
    logger.info("STEP 3: Preprocessing data")
    
    # Handle missing values
    data_cleaned = prep.handle_missing_values(data, {'Income': 'mean'})
    
    # Calculate age based on birth year
    data_with_age = prep.calculate_age(data_cleaned, 'Year_Birth', reference_year=2014)
    
    # Process date features
    data_with_dates = prep.create_date_features(data_with_age, 'Dt_Customer')
    
    # Create total expenditure feature
    amount_columns = [
        'MntWines', 'MntFruits', 'MntMeatProducts', 
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
    ]
    data_with_expenditure = prep.create_expenditure_features(
        data_with_dates, amount_columns, 'Total_Expenditure'
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
    data_with_simple_categories = prep.simplify_categories(
        data_with_expenditure, 'Marital_Status', marital_mapping
    )
    
    # Create family features
    data_with_family = prep.create_family_features(
        data_with_simple_categories, 
        'Kidhome', 
        'Teenhome', 
        'Marital_Status_Simplified'
    )
    
    # Drop redundant features
    cols_to_drop = [
        'ID', 'Dt_Customer', 'Year_Birth', 'Z_CostContact', 
        'Z_Revenue', 'Marital_Status'
    ]
    data_reduced = prep.drop_redundant_features(data_with_family, cols_to_drop)
    
    # Encode categorical variables
    data_encoded = prep.encode_categorical_variables(data_reduced)
    
    # Handle outliers
    data_no_outliers = prep.handle_outliers(
        data_encoded, ['Income', 'Total_Expenditure'], method='clip'
    )
    
    # Scale features for clustering
    data_scaled = prep.scale_features(data_no_outliers, exclude_binary=True)
    
    # Store preprocessing steps
    preprocessing_steps = [
        "Missing value imputation",
        "Age calculation",
        "Date feature extraction",
        "Total expenditure calculation",
        "Category simplification",
        "Family feature creation",
        "Categorical encoding",
        "Outlier handling",
        "Feature scaling"
    ]
    
    # STEP 4: Clustering
    logger.info("STEP 4: Performing clustering")
    
    # Find optimal number of clusters if not specified
    if args.n_clusters <= 0:
        _, optimal_k = cl.find_optimal_clusters(data_scaled, max_clusters=10, method='elbow')
        n_clusters = optimal_k
        logger.info(f"Determined optimal number of clusters: {n_clusters}")
    else:
        n_clusters = args.n_clusters
        logger.info(f"Using specified number of clusters: {n_clusters}")
    
    # Apply the selected clustering method
    clustering_method = args.clustering_method.lower()
    
    if clustering_method == 'kmeans':
        clustered_data = cl.apply_kmeans(
            data_scaled, n_clusters=n_clusters, random_state=args.random_state
        )
        cluster_column = 'Cluster_KMeans'
        
    elif clustering_method == 'hierarchical':
        clustered_data = cl.apply_hierarchical_clustering(
            data_scaled, n_clusters=n_clusters, linkage='ward'
        )
        cluster_column = 'Cluster_Hierarchical'
        
    elif clustering_method == 'fuzzy':
        clustered_data = cl.apply_fuzzy_cmeans(
            data_scaled, n_clusters=n_clusters
        )
        cluster_column = 'Cluster_FCM'
        
    else:
        logger.error(f"Unknown clustering method: {clustering_method}")
        return
    
    # Add cluster assignments back to the original (non-scaled) data
    original_with_clusters = data_no_outliers.copy()
    original_with_clusters[cluster_column] = clustered_data[cluster_column]
    
    # STEP 5: Evaluation
    logger.info("STEP 5: Evaluating clustering results")
    
    # Select important features for analysis
    key_features = [
        'Age', 'Income', 'Total_Expenditure', 'Children_Total', 
        'MntWines', 'MntMeatProducts', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth', 'Recency'
    ]
    
    # Evaluate clusters
    clustering_results = eval.evaluate_clustering(
        clustered_data, [cluster_column]
    )
    
    # Analyze clusters
    cluster_analysis = eval.analyze_clusters(
        original_with_clusters, cluster_column, key_features
    )
    
    # Create cluster summaries
    cluster_summaries = eval.create_cluster_summary(
        original_with_clusters, cluster_column, key_features
    )
    
    # STEP 6: Visualization of Results
    if args.visualize:
        logger.info("STEP 6: Creating visualizations")
        
        # Create PCA visualization
        pca_fig = viz.plot_pca_clusters(clustered_data, cluster_column)
        utils.save_figure(pca_fig, "cluster_pca", output_dir)
        
        # Create t-SNE visualization
        tsne_fig = viz.plot_tsne_clusters(clustered_data, cluster_column)
        utils.save_figure(tsne_fig, "cluster_tsne", output_dir)
        
        # Create radar chart for cluster profiles
        radar_fig = viz.create_cluster_profile_radar(original_with_clusters, cluster_column, key_features)
        utils.save_figure(radar_fig, "cluster_profiles_radar", output_dir)
        
        # Create feature importance heatmap
        heatmap_fig = viz.create_feature_importance_heatmap(original_with_clusters, cluster_column, key_features)
        utils.save_figure(heatmap_fig, "feature_importance_heatmap", output_dir)
        
        # Create cluster distribution plot
        dist_fig = viz.plot_cluster_distribution(original_with_clusters, cluster_column)
        utils.save_figure(dist_fig, "cluster_distribution", output_dir)
        
        # Create feature comparison plots
        for feature in key_features:
            feature_fig = viz.create_cluster_comparison_plot(
                original_with_clusters, feature, cluster_column, 'boxplot'
            )
            utils.save_figure(feature_fig, f"comparison_{feature}", output_dir)
    else:
        logger.info("STEP 6: Skipping result visualizations (use --visualize to enable)")
    
    # STEP 7: Save Results
    logger.info("STEP 7: Saving results")
    
    # Prepare results dictionary
    results = {
        "data_overview": {
            "total_records": data_overview["total_records"],
            "features": key_features,
            "preprocessing_steps": preprocessing_steps
        },
        "clustering_results": {
            "method": args.clustering_method,
            "n_clusters": n_clusters,
            "silhouette_score": float(clustering_results.loc[cluster_column, 'Silhouette Score']),
            "calinski_harabasz_score": float(clustering_results.loc[cluster_column, 'Calinski-Harabasz Index']),
            "davies_bouldin_score": float(clustering_results.loc[cluster_column, 'Davies-Bouldin Index'])
        },
        "cluster_profiles": {}
    }
    
    # Add cluster profiles to results
    for cluster_id in sorted(cluster_summaries.keys()):
        # Parse size and percentage from summary
        import re
        summary = cluster_summaries[cluster_id]
        size_pattern = r"Cluster \d+: (\d+) customers \((\d+\.\d+)% of total\)"
        match = re.search(size_pattern, summary)
        
        if match:
            size = int(match.group(1))
            percentage = float(match.group(2))
        else:
            size = "N/A"
            percentage = "N/A"
        
        # Extract key features
        key_features_dict = {}
        for feature in key_features:
            if feature in summary:
                # Try to extract the feature value from the summary
                pattern = f"{feature}: (-?\d+\.\d+)"
                match = re.search(pattern, summary)
                if match:
                    key_features_dict[feature] = float(match.group(1))
                else:
                    key_features_dict[feature] = "N/A"
        
        results["cluster_profiles"][cluster_id] = {
            "size": size,
            "percentage": percentage,
            "key_features": key_features_dict,
            "summary": summary
        }
    
    # Save results to JSON
    utils.save_results(results, "clustering_results", output_dir)
    
    # Save clustered data to CSV
    utils.save_dataframe(original_with_clusters, "data_with_clusters", output_dir)
    
    # Generate summary report
    report_path = utils.generate_summary_report(results, output_dir)
    
    # Calculate execution time
    end_time = time.time()
    duration, formatted_time = utils.calculate_execution_time(start_time, end_time)
    
    logger.info("=" * 80)
    logger.info(f"ANALYSIS COMPLETED IN {formatted_time}")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Summary report: {report_path}")
    logger.info("=" * 80)

def main():
    """
    Main function to run the marketing campaign analysis.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Run the analysis
        run_analysis(args)
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)