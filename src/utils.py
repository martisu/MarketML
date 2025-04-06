"""
Utilities Module

This module contains utility functions that are used across different modules
in the marketing campaign portfolio project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_directory(dir_name: str = 'output') -> str:
    """
    Create an output directory for saving results and figures.
    
    Args:
        dir_name (str): Name of the directory to create
        
    Returns:
        str: Path to the created directory
    """
    # Create directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f"Created output directory: {dir_name}")
    
    return dir_name

def save_figure(fig: plt.Figure, filename: str, output_dir: str = 'output', 
               dpi: int = 300, formats: List[str] = ['png', 'pdf']) -> List[str]:
    """
    Save a figure to multiple formats.
    
    Args:
        fig (plt.Figure): The figure to save
        filename (str): Base filename (without extension)
        output_dir (str): Output directory
        dpi (int): Resolution in dots per inch
        formats (List[str]): List of formats to save (e.g., ['png', 'pdf'])
        
    Returns:
        List[str]: List of saved file paths
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    saved_files = []
    
    # Save figure in each specified format
    for fmt in formats:
        file_path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        saved_files.append(file_path)
        logger.info(f"Saved figure to {file_path}")
    
    return saved_files

def save_results(results: Dict[str, Any], filename: str, output_dir: str = 'output') -> str:
    """
    Save results dictionary to a JSON file.
    
    Args:
        results (Dict[str, Any]): Results to save
        filename (str): Filename (without extension)
        output_dir (str): Output directory
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Convert data types that are not JSON serializable
    def json_serializer(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    # Save results to JSON file
    file_path = os.path.join(output_dir, f"{filename}.json")
    with open(file_path, 'w') as f:
        json.dump(results, f, default=json_serializer, indent=4)
    
    logger.info(f"Saved results to {file_path}")
    
    return file_path

def save_dataframe(df: pd.DataFrame, filename: str, output_dir: str = 'output', 
                  format: str = 'csv') -> str:
    """
    Save a DataFrame to a file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Filename (without extension)
        output_dir (str): Output directory
        format (str): Format to save ('csv', 'excel', or 'pickle')
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Save DataFrame based on format
    if format.lower() == 'csv':
        file_path = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(file_path, index=True)
    elif format.lower() == 'excel':
        file_path = os.path.join(output_dir, f"{filename}.xlsx")
        df.to_excel(file_path, index=True)
    elif format.lower() == 'pickle':
        file_path = os.path.join(output_dir, f"{filename}.pkl")
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv', 'excel', or 'pickle'.")
    
    logger.info(f"Saved DataFrame to {file_path}")
    
    return file_path

def generate_summary_report(results: Dict[str, Any], output_dir: str = 'output',
                           filename: str = 'summary_report') -> str:
    """
    Generate a summary report in markdown format.
    
    Args:
        results (Dict[str, Any]): Dictionary containing results
        output_dir (str): Output directory
        filename (str): Filename (without extension)
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build markdown content
    md_content = f"""# Marketing Campaign Clustering Analysis
    
## Summary Report
Generated on: {timestamp}

## Overview
This report summarizes the results of clustering analysis performed on marketing campaign data.

"""
    
    # Add data overview if available
    if 'data_overview' in results:
        data_overview = results['data_overview']
        md_content += f"""
## Data Overview
- Total records: {data_overview.get('total_records', 'N/A')}
- Features used: {', '.join(data_overview.get('features', []))}
- Preprocessing steps: {', '.join(data_overview.get('preprocessing_steps', []))}
"""
    
    # Add clustering results if available
    if 'clustering_results' in results:
        clustering_results = results['clustering_results']
        md_content += f"""
## Clustering Results
- Method: {clustering_results.get('method', 'N/A')}
- Number of clusters: {clustering_results.get('n_clusters', 'N/A')}
- Silhouette Score: {clustering_results.get('silhouette_score', 'N/A'):.4f}
- Calinski-Harabasz Index: {clustering_results.get('calinski_harabasz_score', 'N/A'):.4f}
- Davies-Bouldin Index: {clustering_results.get('davies_bouldin_score', 'N/A'):.4f}
"""
    
    # Add cluster profiles if available
    if 'cluster_profiles' in results:
        cluster_profiles = results['cluster_profiles']
        md_content += """
## Cluster Profiles
"""
        for cluster_id, profile in cluster_profiles.items():
            md_content += f"""
### Cluster {cluster_id}
- Size: {profile.get('size', 'N/A')} ({profile.get('percentage', 'N/A'):.2f}% of total)
- Key characteristics:
"""
            for feature, value in profile.get('key_features', {}).items():
                md_content += f"  - {feature}: {value}\n"
    
    # Add recommendations if available
    if 'recommendations' in results:
        recommendations = results['recommendations']
        md_content += """
## Marketing Recommendations
"""
        for cluster_id, recs in recommendations.items():
            md_content += f"""
### For Cluster {cluster_id}
"""
            for rec in recs:
                md_content += f"- {rec}\n"
    
    # Add footer
    md_content += """
## Next Steps
- Validate clusters with business stakeholders
- Develop targeted marketing strategies for each segment
- Implement A/B testing for different approaches
- Monitor campaign performance by segment

---
Generated automatically by Marketing Campaign Portfolio Analysis Tool
"""
    
    # Save markdown file
    file_path = os.path.join(output_dir, f"{filename}.md")
    with open(file_path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Generated summary report: {file_path}")
    
    return file_path

def calculate_execution_time(start_time: float, end_time: float = None) -> Tuple[float, str]:
    """
    Calculate execution time and format it.
    
    Args:
        start_time (float): Start time (from time.time())
        end_time (float, optional): End time. If None, current time is used.
        
    Returns:
        Tuple[float, str]: Execution time in seconds and formatted string
    """
    import time
    
    # Get end time if not provided
    if end_time is None:
        end_time = time.time()
    
    # Calculate duration in seconds
    duration_seconds = end_time - start_time
    
    # Format duration
    if duration_seconds < 60:
        formatted_time = f"{duration_seconds:.2f} seconds"
    elif duration_seconds < 3600:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        formatted_time = f"{minutes} minutes and {seconds:.2f} seconds"
    else:
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60
        formatted_time = f"{hours} hours, {minutes} minutes, and {seconds:.2f} seconds"
    
    return duration_seconds, formatted_time

def get_memory_usage() -> Tuple[float, str]:
    """
    Get current memory usage of the process.
    
    Returns:
        Tuple[float, str]: Memory usage in MB and formatted string
    """
    import os
    import psutil
    
    # Get the current process
    process = psutil.Process(os.getpid())
    
    # Get memory info in bytes
    memory_bytes = process.memory_info().rss
    
    # Convert to MB
    memory_mb = memory_bytes / (1024 * 1024)
    
    # Format
    formatted_memory = f"{memory_mb:.2f} MB"
    
    return memory_mb, formatted_memory

def generate_random_color_palette(n_colors: int, seed: Optional[int] = None) -> List[str]:
    """
    Generate a list of random distinct colors in hexadecimal format.
    
    Args:
        n_colors (int): Number of colors to generate
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        List[str]: List of hexadecimal color codes
    """
    import random
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Generate colors
    colors = []
    for _ in range(n_colors):
        # Generate random RGB values
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Convert to hexadecimal
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        colors.append(hex_color)
    
    return colors

if __name__ == "__main__":
    # Example usage
    output_dir = create_output_directory("example_output")
    
    # Example data
    results = {
        "data_overview": {
            "total_records": 2240,
            "features": ["Age", "Income", "Recency", "NumWebPurchases"],
            "preprocessing_steps": ["Missing value imputation", "Feature scaling", "Outlier handling"]
        },
        "clustering_results": {
            "method": "KMeans",
            "n_clusters": 3,
            "silhouette_score": 0.7123,
            "calinski_harabasz_score": 453.78,
            "davies_bouldin_score": 0.32
        },
        "cluster_profiles": {
            "0": {
                "size": 823,
                "percentage": 36.74,
                "key_features": {
                    "Income": "High (avg: $78,500)",
                    "Age": "Middle-aged (avg: 45)",
                    "NumWebPurchases": "Frequent (avg: 8.2)"
                }
            },
            "1": {
                "size": 912,
                "percentage": 40.71,
                "key_features": {
                    "Income": "Medium (avg: $52,300)",
                    "Age": "Young (avg: 32)",
                    "NumWebPurchases": "Average (avg: 4.7)"
                }
            },
            "2": {
                "size": 505,
                "percentage": 22.55,
                "key_features": {
                    "Income": "Low (avg: $31,200)",
                    "Age": "Senior (avg: 61)",
                    "NumWebPurchases": "Rare (avg: 2.1)"
                }
            }
        },
        "recommendations": {
            "0": [
                "Target with premium products and exclusive offers",
                "Focus on digital channels and web campaigns",
                "Create loyalty programs with high-value rewards"
            ],
            "1": [
                "Offer mid-range products with good value proposition",
                "Use a mix of digital and traditional marketing channels",
                "Provide special offers for increasing purchase frequency"
            ],
            "2": [
                "Emphasize affordability and essential products",
                "Focus more on traditional marketing channels",
                "Offer special assistance and education on web purchasing"
            ]
        }
    }
    
    # Generate report
    report_path = generate_summary_report(results, output_dir)
    print(f"Report generated: {report_path}")
    
    # Test execution time formatting
    import time
    start = time.time()
    time.sleep(2)  # Simulate computation
    duration, formatted = calculate_execution_time(start)
    print(f"Execution time: {formatted}")