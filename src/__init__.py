"""
MarketML - Customer Segmentation for Marketing Campaigns

A comprehensive toolkit for analyzing customer data and identifying market segments 
using various clustering algorithms.

This module initializes the MarketML package and manages imports.
"""

__version__ = '0.1.0'
__author__ = 'Kevin Suin'
__email__ = 'martinsuin@gmail.com'

# Import key functions for easier access
from .data_loader import load_data, display_data_summary
from .preprocessing import handle_missing_values, scale_features, encode_categorical_variables
from .clustering import apply_kmeans, apply_hierarchical_clustering, apply_fuzzy_cmeans
from .evaluation import evaluate_clustering, analyze_clusters
from .visualization import plot_pca_clusters, create_cluster_profile_radar