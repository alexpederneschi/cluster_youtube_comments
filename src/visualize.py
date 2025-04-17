"""
Visualization of vector embeddings using dimensionality reduction techniques.

Takes a list of JSON objects with "Embedding" fields and associated metadata.
Generates visualizations using PCA, t-SNE, and UMAP.
Outputs visualizations as PNG files in the specified output directory.
"""
import argparse
import json
from typing import List, Dict, Any
import logging
import sys
from pathlib import Path
import time
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("visualize")

warnings.filterwarnings("ignore", message="n_jobs value .* overridden to .* by setting random_state.*")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'.*")

DEFAULT_CLUSTER_COUNT = 6
DEFAULT_SIZE_BY = None
DEFAULT_PCA_MAX_POINTS = 10000
DEFAULT_TSNE_MAX_POINTS = 5000
DEFAULT_UMAP_MAX_POINTS = 8000

def load_embeddings_from_json(input_file) -> tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load embeddings from input file into a numpy array and keep the original data
    """
    
    data = []
    embeddings = []

    total_lines = sum(1 for _ in input_file)
    input_file.seek(0)

    for line in tqdm(input_file, total=total_lines, desc="Loading embeddings"):
        item = json.loads(line)
        embeddings.append(item["Embedding"])
        data.append({k: v for k, v in item.items()})

    return pd.DataFrame(data), np.array(embeddings, dtype=np.float32)

def sample_data(metadata, embeddings, max_points):
    """
    Randomly sample data if there are more points than max_points
    
    Parameters:
    - metadata: pandas DataFrame with metadata
    - embeddings: numpy array of embeddings
    - max_points: maximum number of points to include
    
    Returns:
    - sampled_metadata: sampled metadata
    - sampled_embeddings: sampled embeddings
    """

    logger.info(f"Sampling {max_points} points from {len(embeddings)} total points")
    
    # Generate random indices for sampling
    np.random.seed(42)
    indices = np.random.choice(len(embeddings), max_points, replace=False)
    
    # Sample both metadata and embeddings
    sampled_metadata = metadata.iloc[indices].reset_index(drop=True)
    sampled_embeddings = embeddings[indices]
    
    return sampled_metadata, sampled_embeddings

def plot_embeddings(embeddings, metadata, method, n_clusters, color_by, figsize=(14, 10),
                    size_by=None, output_dir='output', max_points=None):
    """
    Plot embeddings with dimensionality reduction
    Parameters:
    - embeddings: numpy array of embeddings
    - metadata: pandas DataFrame with metadata
    - method: 'tsne', 'pca', or 'umap'
    - size_by: metadata field to use for point sizes
    - n_clusters: number of clusters for K-means clustering
    - output_dir: directory to save output plots
    - max_points: maximum number of points to visualize (will sample if exceeded)
    """
    # Apply sampling if data points are greater than max_points
    if len(embeddings) > max_points:
        sampled_metadata, sampled_embeddings = sample_data(metadata, embeddings, max_points)
    else:
        sampled_metadata, sampled_embeddings = metadata, embeddings
    
    plt.figure(figsize=figsize)
    
    # Determine colors for points
    sm = None  # Initialize sm for potential use in colorbar
    if color_by == 'cluster':
        # Use K-means clustering for colors
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(sampled_embeddings)
        clustering_time = time.time() - start_time
        logger.info(f"K-means clustering completed in {clustering_time:.2f} seconds")
        
        # Create a colormap based on clusters
        norm = Normalize(vmin=0, vmax=n_clusters-1)
        cmap = plt.colormaps['tab10'].resampled(n_clusters)
        colors = cmap(norm(clusters))
        
        # For legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          label=f'Cluster {i + 1}', markerfacecolor=cmap(norm(i)), markersize=8)
                          for i in range(n_clusters)]
        legend_title = "Clusters"
        
    elif color_by in sampled_metadata.columns:
        # Use metadata field for colors
        color_values = sampled_metadata[color_by]

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(color_values):
            # Use a continuous colormap for numeric data
            norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
            cmap = plt.get_cmap('viridis')
            colors = cmap(norm(color_values))
            
            # For colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            legend_elements = None
            legend_title = color_by
        else:
            # Use categorical colors for non-numeric data
            unique_values = color_values.unique()
            value_to_int = {val: i for i, val in enumerate(unique_values)}
            color_indices = [value_to_int[val] for val in color_values]
            
            cmap = cm.get_cmap('tab10', len(unique_values))
            norm = Normalize(vmin=0, vmax=len(unique_values)-1)
            colors = cmap(norm(color_indices))
            
            # For legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              label=f'{val}', markerfacecolor=cmap(norm(i)), markersize=8)
                              for i, val in enumerate(unique_values)]
            legend_title = color_by
    else:
        # Default coloring
        colors = 'blue'
        legend_elements = None
        legend_title = None
    
    # Determine point sizes
    if size_by is not None and isinstance(sampled_metadata, pd.DataFrame) and size_by in sampled_metadata.columns:
        # Use metadata field for sizes if it's numeric
        if pd.api.types.is_numeric_dtype(sampled_metadata[size_by]):
            # Normalize sizes between 20 and 200
            size_values = sampled_metadata[size_by]
            min_size, max_size = 20, 200
            if size_values.max() > size_values.min():
                sizes = min_size + (max_size - min_size) * (size_values - size_values.min()) / (size_values.max() - size_values.min())
            else:
                sizes = min_size
        else:
            sizes = 30
    else:
        sizes = 30
    
    # Apply dimensionality reduction with timing
    start_time = time.time()
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(sampled_embeddings)
        explained_variance = sum(reducer.explained_variance_ratio_)
        title = f"PCA visualization ({len(sampled_embeddings)} samples, variance: {explained_variance:.2f})"
        if len(sampled_embeddings) < len(embeddings):
            title += f" (sampled from {len(embeddings)})"
    elif method == 'tsne':
        perplexity = min(30, len(sampled_embeddings)-1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embedding_2d = reducer.fit_transform(sampled_embeddings)
        title = f"t-SNE visualization of {len(sampled_embeddings)} documents"
        if len(sampled_embeddings) < len(embeddings):
            title += f" (sampled from {len(embeddings)})"
    elif method == 'umap':
        n_neighbors = min(15, len(sampled_embeddings)-1)
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        embedding_2d = reducer.fit_transform(sampled_embeddings)
        title = f"UMAP visualization of {len(sampled_embeddings)} documents"
        if len(sampled_embeddings) < len(embeddings):
            title += f" (sampled from {len(embeddings)})"
    
    computation_time = time.time() - start_time
    logger.info(f"{method.upper()} computation completed in {computation_time:.2f} seconds")

    # Create scatter plot
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=sizes, alpha=0.7, c=colors)
    
    # Add legend or colorbar
    if legend_elements is not None:
        plt.legend(handles=legend_elements, title=legend_title, loc="best")
    elif legend_title is not None and sm is not None:
        # Add colorbar if we have a ScalarMappable
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(legend_title)
    
    plt.title(f"{title}\nComputation time: {computation_time:.2f}s")
    plt.tight_layout()

    # Save the figure
    output_path = f"{output_dir}/{method}_visualization.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved {method} visualization to {output_path}")
    plt.close()  # Close the figure to free memory
    
    return embedding_2d

def main():
    """Main function to run the visualization pipeline"""
    # Set up argparse
    parser = argparse.ArgumentParser(description='Visualize embeddings using dimensionality reduction techniques.')
    parser.add_argument('--input', type=argparse.FileType('r'), default=sys.stdin, help='Path to the JSON file containing embeddings and metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--n_clusters', type=int, default=DEFAULT_CLUSTER_COUNT, help='Number of clusters for K-means clustering')
    parser.add_argument('--size_by', type=str, default=DEFAULT_SIZE_BY, help='Metadata field to use for point sizes')
    parser.add_argument('--pca_max_points', type=int, default=DEFAULT_PCA_MAX_POINTS, help='Maximum number of points for PCA visualization')
    parser.add_argument('--tsne_max_points', type=int, default=DEFAULT_TSNE_MAX_POINTS, help='Maximum number of points for t-SNE visualization')
    parser.add_argument('--umap_max_points', type=int, default=DEFAULT_UMAP_MAX_POINTS, help='Maximum number of points for UMAP visualization')    
    args = parser.parse_args()
    
    # Validate output directory (fail fast if invalid)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    metadata, embeddings = load_embeddings_from_json(args.input)

    # Plot PCA
    logger.info("Processing PCA visualization...")
    plot_embeddings(
        embeddings, metadata, method='pca',
        color_by='Likes',  # override for PCA
        n_clusters=args.n_clusters,
        max_points=args.pca_max_points,
        output_dir=args.output_dir
    )

    # Plot t-SNE
    logger.info("Processing t-SNE visualization...")
    plot_embeddings(
        embeddings, metadata, method='tsne',
        color_by='cluster',
        n_clusters=args.n_clusters,
        size_by=args.size_by, 
        max_points=args.tsne_max_points,
        output_dir=args.output_dir
    )

    # Plot UMAP
    logger.info("Processing UMAP visualization...")
    plot_embeddings(
        embeddings, metadata, method='umap',
        color_by='cluster',
        n_clusters=args.n_clusters,
        size_by=args.size_by, 
        max_points=args.umap_max_points,
        output_dir=args.output_dir
    )
    
    logger.info("Plots saved successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")