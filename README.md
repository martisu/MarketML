# MarketML: Customer Segmentation for Marketing Campaigns

A comprehensive data science portfolio project demonstrating customer segmentation techniques for marketing campaign optimization.

## Overview

MarketML is a robust Python-based toolkit that analyzes customer data to identify distinct market segments using various clustering algorithms. This project showcases the complete data science workflow from exploratory data analysis to model evaluation and visualization, allowing marketers to develop targeted campaigns based on customer behavior and preferences.

## Features

- **Data Loading & Exploration**: Automated data profiling and exploratory analysis with visualizations
- **Advanced Preprocessing**: Comprehensive data cleaning, feature engineering, and transformation
- **Multiple Clustering Algorithms**: Implementation of K-Means, Hierarchical, and Fuzzy C-Means clustering
- **Cluster Evaluation**: Statistical validation using silhouette score, Calinski-Harabasz, and Davies-Bouldin indices
- **Interactive Visualizations**: PCA, t-SNE, radar charts, heatmaps, and comparative plots
- **Detailed Reporting**: Automated generation of comprehensive analysis reports

## Project Structure

```
MarketML/
├── main.py                # Main orchestration script
├── data/                  # Data directory
│   └── marketing_campaign.csv
├── src/                   # Source code directory
│   ├── __init__.py
│   ├── data_loader.py     # Functions for loading and initial data exploration
│   ├── eda.py             # EDA and visualization functions
│   ├── preprocessing.py   # Data cleaning and feature engineering
│   ├── clustering.py      # Implementation of clustering algorithms
│   ├── evaluation.py      # Metrics and cluster analysis
│   ├── visualization.py   # Advanced visualization functions
│   └── utils.py           # Utility functions for the entire project
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── output/                # Generated reports and visualizations
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/martisu/MarketML.git
   cd MarketML
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the complete analysis with default parameters:

```
python main.py --data marketing_campaign.csv --visualize
```

### Advanced Options

```
python main.py --data marketing_campaign.csv \
               --output custom_output \
               --n_clusters 4 \
               --clustering_method hierarchical \
               --evaluation_method all \
               --visualize
```

### Command Line Arguments

- `--data`: Path to the input CSV file (default: marketing_campaign.csv)
- `--output`: Output directory for results and visualizations (default: output)
- `--n_clusters`: Number of clusters to create (default: 3)
- `--clustering_method`: Clustering method to use (options: kmeans, hierarchical, fuzzy; default: kmeans)
- `--evaluation_method`: Evaluation method for clusters (options: silhouette, calinski, davies, all; default: all)
- `--visualize`: Generate visualizations (flag, include to enable)
- `--random_state`: Random seed for reproducibility (default: 42)

## Example Results

When you run the project, you'll get:

1. **Cluster Profiles**: Detailed analysis of each customer segment
2. **Feature Importance**: Key drivers for segmentation
3. **Visual Representation**: Multiple visualizations to understand segment distribution
4. **Report Generation**: Comprehensive markdown report summarizing findings
5. **Marketing Recommendations**: Suggestions for targeting each segment

## Dataset Information

The default dataset contains customer information including:

- Demographics (age, education, marital status)
- Purchase behavior (spending on different product categories)
- Campaign responses
- Web and catalog interactions

You can substitute your own dataset, ensuring it follows a similar format.

## Key Insights from Clustering

The analysis typically reveals several distinct customer segments, such as:

1. **High-Value Loyalists**: High income, frequent purchases across channels
2. **Digital-First Shoppers**: Medium income, prefer online shopping
3. **Catalog Enthusiasts**: Older demographic who respond to traditional marketing
4. **Budget-Conscious Occasionals**: Lower spending, infrequent engagement

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scikit-fuzzy
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.