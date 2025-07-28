"""
Visualization Module for CUSTO CLARITY Project

This module contains comprehensive visualization functions for customer segmentation analysis,
including EDA plots, cluster visualizations, and business insights dashboards.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CustomerVisualizationSuite:
    """Comprehensive visualization suite for customer segmentation analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualization suite.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        self.plots_created = []
        
    def plot_data_overview(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive data overview plots.
        
        Args:
            df (pd.DataFrame): Input dataset
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating data overview visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Data Overview - CUSTO CLARITY Analysis\nby Neelanjan Chakraborty', 
                    fontsize=16, fontweight='bold')
        
        # Age distribution
        if 'Age' in df.columns:
            sns.histplot(data=df, x='Age', bins=20, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Age Distribution')
            axes[0, 0].set_xlabel('Age')
            axes[0, 0].set_ylabel('Frequency')
        
        # Income distribution
        if 'Annual Income (k$)' in df.columns:
            sns.histplot(data=df, x='Annual Income (k$)', bins=20, kde=True, ax=axes[0, 1])
            axes[0, 1].set_title('Annual Income Distribution')
            axes[0, 1].set_xlabel('Annual Income (k$)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Spending Score distribution
        if 'Spending Score (1-100)' in df.columns:
            sns.histplot(data=df, x='Spending Score (1-100)', bins=20, kde=True, ax=axes[0, 2])
            axes[0, 2].set_title('Spending Score Distribution')
            axes[0, 2].set_xlabel('Spending Score (1-100)')
            axes[0, 2].set_ylabel('Frequency')
        
        # Gender distribution
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Gender Distribution')
        
        # Age vs Income scatter
        if 'Age' in df.columns and 'Annual Income (k$)' in df.columns:
            sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', 
                          hue='Gender' if 'Gender' in df.columns else None, ax=axes[1, 1])
            axes[1, 1].set_title('Age vs Annual Income')
        
        # Income vs Spending Score scatter
        if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
            sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                          hue='Gender' if 'Gender' in df.columns else None, ax=axes[1, 2])
            axes[1, 2].set_title('Annual Income vs Spending Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data overview plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('data_overview')
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Create correlation matrix heatmap.
        
        Args:
            df (pd.DataFrame): Input dataset
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating correlation matrix visualization...")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove CustomerID if present
        if 'CustomerID' in numeric_df.columns:
            numeric_df = numeric_df.drop('CustomerID', axis=1)
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Customer Data Correlation Matrix\nCUSTO CLARITY - by Neelanjan Chakraborty', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('correlation_matrix')
    
    def plot_outlier_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Create outlier analysis visualization using box plots.
        
        Args:
            df (pd.DataFrame): Input dataset
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating outlier analysis visualization...")
        
        # Select numeric columns excluding CustomerID
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'CustomerID' in numeric_cols:
            numeric_cols.remove('CustomerID')
        
        if not numeric_cols:
            logger.warning("No numeric columns found for outlier analysis")
            return
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
        fig.suptitle('Outlier Analysis - Box Plots\nCUSTO CLARITY - by Neelanjan Chakraborty', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(f'{col} - Box Plot')
                axes[i].set_ylabel(col)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Outlier analysis plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('outlier_analysis')
    
    def plot_cluster_analysis_2d(self, X: np.ndarray, labels: np.ndarray, 
                               title: str = "Customer Segmentation",
                               feature_names: List[str] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Create 2D cluster visualization.
        
        Args:
            X (np.ndarray): Feature matrix (2D)
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            feature_names (List[str]): Names of features
            save_path (str, optional): Path to save the plot
        """
        logger.info(f"Creating 2D cluster visualization: {title}")
        
        if X.shape[1] < 2:
            logger.warning("Need at least 2 features for 2D visualization")
            return
        
        plt.figure(figsize=self.figsize)
        
        # Use first two features if more than 2
        x_data = X[:, 0]
        y_data = X[:, 1]
        
        # Create scatter plot with different colors for each cluster
        unique_labels = np.unique(labels)
        colors = self.color_palette[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_name = f'Noise' if label == -1 else f'Cluster {label}'
            plt.scatter(x_data[mask], y_data[mask], 
                       c=[colors[i % len(colors)]], 
                       label=cluster_name, 
                       alpha=0.7, 
                       s=50)
        
        plt.xlabel(feature_names[0] if feature_names else 'Feature 1')
        plt.ylabel(feature_names[1] if feature_names else 'Feature 2')
        plt.title(f'{title}\nCUSTO CLARITY - by Neelanjan Chakraborty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D cluster plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('cluster_2d')
    
    def plot_cluster_analysis_3d(self, X: np.ndarray, labels: np.ndarray,
                               title: str = "Customer Segmentation 3D",
                               feature_names: List[str] = None) -> None:
        """
        Create interactive 3D cluster visualization using Plotly.
        
        Args:
            X (np.ndarray): Feature matrix (at least 3D)
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            feature_names (List[str]): Names of features
        """
        logger.info(f"Creating 3D cluster visualization: {title}")
        
        if X.shape[1] < 3:
            logger.warning("Need at least 3 features for 3D visualization")
            return
        
        # Use first three features
        df_plot = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1],
            'z': X[:, 2],
            'cluster': labels
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(df_plot, x='x', y='y', z='z', color='cluster',
                          title=f'{title}<br>CUSTO CLARITY - by Neelanjan Chakraborty',
                          labels={
                              'x': feature_names[0] if feature_names else 'Feature 1',
                              'y': feature_names[1] if feature_names else 'Feature 2',
                              'z': feature_names[2] if feature_names else 'Feature 3'
                          })
        
        fig.update_layout(scene=dict(
            xaxis_title=feature_names[0] if feature_names else 'Feature 1',
            yaxis_title=feature_names[1] if feature_names else 'Feature 2',
            zaxis_title=feature_names[2] if feature_names else 'Feature 3'
        ))
        
        fig.show()
        self.plots_created.append('cluster_3d')
    
    def plot_dimensionality_reduction(self, original_data: np.ndarray, 
                                    reduced_data: Dict[str, np.ndarray],
                                    labels: np.ndarray,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot dimensionality reduction results (PCA, t-SNE).
        
        Args:
            original_data (np.ndarray): Original high-dimensional data
            reduced_data (Dict[str, np.ndarray]): Dictionary of reduced data
            labels (np.ndarray): Cluster labels
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating dimensionality reduction visualization...")
        
        n_methods = len(reduced_data)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
        
        colors = self.color_palette[:len(np.unique(labels))]
        
        for i, (method, data) in enumerate(reduced_data.items()):
            for j, label in enumerate(np.unique(labels)):
                mask = labels == label
                cluster_name = f'Noise' if label == -1 else f'Cluster {label}'
                axes[i].scatter(data[mask, 0], data[mask, 1], 
                              c=[colors[j % len(colors)]], 
                              label=cluster_name, alpha=0.7)
            
            axes[i].set_title(f'{method.upper()} Visualization')
            axes[i].set_xlabel(f'{method.upper()} Component 1')
            axes[i].set_ylabel(f'{method.upper()} Component 2')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle('Dimensionality Reduction Analysis\nCUSTO CLARITY - by Neelanjan Chakraborty', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dimensionality reduction plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('dimensionality_reduction')
    
    def plot_cluster_profiles(self, profiles: Dict[str, Any], 
                            save_path: Optional[str] = None) -> None:
        """
        Create cluster profile visualization.
        
        Args:
            profiles (Dict[str, Any]): Cluster profiles from clustering analysis
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating cluster profiles visualization...")
        
        if 'numeric' not in profiles:
            logger.warning("No numeric profiles found for visualization")
            return
        
        numeric_profiles = profiles['numeric']
        cluster_sizes = profiles.get('cluster_sizes', {})
        
        # Get mean values for each cluster
        mean_profiles = numeric_profiles.xs('mean', level=1, axis=1)
        
        # Create subplot for each feature
        n_features = len(mean_profiles.columns)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle('Cluster Profiles Analysis\nCUSTO CLARITY - by Neelanjan Chakraborty', 
                    fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(mean_profiles.columns):
            if i < len(axes):
                bars = axes[i].bar(range(len(mean_profiles)), mean_profiles[feature])
                axes[i].set_title(f'Average {feature} by Cluster')
                axes[i].set_xlabel('Cluster')
                axes[i].set_ylabel(f'Average {feature}')
                axes[i].set_xticks(range(len(mean_profiles)))
                axes[i].set_xticklabels([f'C{i}' for i in mean_profiles.index])
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}',
                               ha='center', va='bottom')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster profiles plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('cluster_profiles')
    
    def plot_cluster_sizes(self, cluster_sizes: pd.Series, 
                         save_path: Optional[str] = None) -> None:
        """
        Create cluster sizes visualization.
        
        Args:
            cluster_sizes (pd.Series): Cluster sizes
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating cluster sizes visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Customer Cluster Distribution\nCUSTO CLARITY - by Neelanjan Chakraborty', 
                    fontsize=16, fontweight='bold')
        
        # Bar plot
        bars = ax1.bar(range(len(cluster_sizes)), cluster_sizes.values)
        ax1.set_title('Cluster Sizes')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Customers')
        ax1.set_xticks(range(len(cluster_sizes)))
        ax1.set_xticklabels([f'Cluster {i}' for i in cluster_sizes.index])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(cluster_sizes.values, 
               labels=[f'Cluster {i}' for i in cluster_sizes.index],
               autopct='%1.1f%%', 
               startangle=90)
        ax2.set_title('Cluster Distribution Percentage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster sizes plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('cluster_sizes')
    
    def plot_elbow_analysis(self, k_range: List[int], inertias: List[float],
                          silhouette_scores: List[float],
                          save_path: Optional[str] = None) -> None:
        """
        Create elbow method and silhouette analysis plots.
        
        Args:
            k_range (List[int]): Range of k values tested
            inertias (List[float]): Inertia values for each k
            silhouette_scores (List[float]): Silhouette scores for each k
            save_path (str, optional): Path to save the plot
        """
        logger.info("Creating elbow analysis visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Optimal Clusters Analysis\nCUSTO CLARITY - by Neelanjan Chakraborty', 
                    fontsize=16, fontweight='bold')
        
        # Elbow plot
        ax1.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
        ax1.set_title('Elbow Method for Optimal k')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.grid(True, alpha=0.3)
        
        # Mark the elbow point (simplified method)
        if len(inertias) > 2:
            # Find point with maximum curvature (simplified)
            diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            elbow_idx = np.argmax(diffs) + 1
            ax1.axvline(x=k_range[elbow_idx], color='red', linestyle='--', 
                       label=f'Elbow at k={k_range[elbow_idx]}')
            ax1.legend()
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, 'ro-', markersize=8, linewidth=2)
        ax2.set_title('Silhouette Analysis')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Average Silhouette Score')
        ax2.grid(True, alpha=0.3)
        
        # Mark the best silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        ax2.axvline(x=k_range[best_k_idx], color='red', linestyle='--',
                   label=f'Best k={k_range[best_k_idx]}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Elbow analysis plot saved to {save_path}")
        
        plt.show()
        self.plots_created.append('elbow_analysis')
    
    def create_business_dashboard(self, df: pd.DataFrame, 
                                cluster_labels: np.ndarray,
                                profiles: Dict[str, Any]) -> None:
        """
        Create an interactive business dashboard using Plotly.
        
        Args:
            df (pd.DataFrame): Original dataset
            cluster_labels (np.ndarray): Cluster labels
            profiles (Dict[str, Any]): Cluster profiles
        """
        logger.info("Creating interactive business dashboard...")
        
        # Add cluster labels to dataframe
        df_viz = df.copy()
        df_viz['Cluster'] = cluster_labels
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age vs Income by Cluster', 'Income vs Spending by Cluster',
                          'Cluster Distribution', 'Average Metrics by Cluster'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # Age vs Income scatter
        if 'Age' in df_viz.columns and 'Annual Income (k$)' in df_viz.columns:
            for cluster in sorted(df_viz['Cluster'].unique()):
                cluster_data = df_viz[df_viz['Cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(x=cluster_data['Age'], 
                             y=cluster_data['Annual Income (k$)'],
                             mode='markers',
                             name=f'Cluster {cluster}',
                             showlegend=True),
                    row=1, col=1
                )
        
        # Income vs Spending scatter
        if 'Annual Income (k$)' in df_viz.columns and 'Spending Score (1-100)' in df_viz.columns:
            for cluster in sorted(df_viz['Cluster'].unique()):
                cluster_data = df_viz[df_viz['Cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(x=cluster_data['Annual Income (k$)'], 
                             y=cluster_data['Spending Score (1-100)'],
                             mode='markers',
                             name=f'Cluster {cluster}',
                             showlegend=False),
                    row=1, col=2
                )
        
        # Cluster distribution pie chart
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        fig.add_trace(
            go.Pie(labels=[f'Cluster {i}' for i in cluster_counts.index],
                  values=cluster_counts.values,
                  name="Distribution"),
            row=2, col=1
        )
        
        # Average metrics by cluster (if numeric profiles available)
        if 'numeric' in profiles:
            mean_profiles = profiles['numeric'].xs('mean', level=1, axis=1)
            for feature in mean_profiles.columns:
                fig.add_trace(
                    go.Bar(x=[f'C{i}' for i in mean_profiles.index],
                          y=mean_profiles[feature],
                          name=feature,
                          showlegend=False),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Customer Segmentation Business Dashboard<br>CUSTO CLARITY - by Neelanjan Chakraborty",
            title_x=0.5,
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Age", row=1, col=1)
        fig.update_yaxes(title_text="Annual Income (k$)", row=1, col=1)
        fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=2)
        fig.update_yaxes(title_text="Spending Score", row=1, col=2)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Average Value", row=2, col=2)
        
        fig.show()
        self.plots_created.append('business_dashboard')
    
    def generate_visualization_report(self) -> Dict[str, Any]:
        """
        Generate a report of all visualizations created.
        
        Returns:
            Dict[str, Any]: Report summary
        """
        report = {
            'total_plots_created': len(self.plots_created),
            'plot_types': self.plots_created,
            'recommended_insights': [
                "Review cluster profiles for customer segment characteristics",
                "Analyze dimensionality reduction for data patterns",
                "Use business dashboard for stakeholder presentations",
                "Consider outlier analysis for data quality assessment"
            ]
        }
        
        logger.info(f"Visualization report generated: {len(self.plots_created)} plots created")
        return report


def create_comprehensive_visualization_suite(df: pd.DataFrame, 
                                           cluster_labels: np.ndarray,
                                           X: np.ndarray,
                                           profiles: Dict[str, Any],
                                           reduced_data: Dict[str, np.ndarray] = None,
                                           feature_names: List[str] = None,
                                           save_directory: str = "outputs/figures") -> CustomerVisualizationSuite:
    """
    Create a comprehensive visualization suite for customer segmentation analysis.
    
    Args:
        df (pd.DataFrame): Original dataset
        cluster_labels (np.ndarray): Cluster labels
        X (np.ndarray): Feature matrix
        profiles (Dict[str, Any]): Cluster profiles
        reduced_data (Dict[str, np.ndarray], optional): Dimensionality reduction results
        feature_names (List[str], optional): Feature names
        save_directory (str): Directory to save plots
    
    Returns:
        CustomerVisualizationSuite: Visualization suite object
    """
    import os
    os.makedirs(save_directory, exist_ok=True)
    
    viz_suite = CustomerVisualizationSuite()
    
    # Create all visualizations
    viz_suite.plot_data_overview(df, f"{save_directory}/data_overview.png")
    viz_suite.plot_correlation_matrix(df, f"{save_directory}/correlation_matrix.png")
    viz_suite.plot_outlier_analysis(df, f"{save_directory}/outlier_analysis.png")
    
    if X.shape[1] >= 2:
        viz_suite.plot_cluster_analysis_2d(X, cluster_labels, 
                                         feature_names=feature_names,
                                         save_path=f"{save_directory}/clusters_2d.png")
    
    if X.shape[1] >= 3:
        viz_suite.plot_cluster_analysis_3d(X, cluster_labels, feature_names=feature_names)
    
    if reduced_data:
        viz_suite.plot_dimensionality_reduction(X, reduced_data, cluster_labels,
                                               f"{save_directory}/dimensionality_reduction.png")
    
    viz_suite.plot_cluster_profiles(profiles, f"{save_directory}/cluster_profiles.png")
    
    if 'cluster_sizes' in profiles:
        viz_suite.plot_cluster_sizes(profiles['cluster_sizes'], 
                                    f"{save_directory}/cluster_sizes.png")
    
    # Create business dashboard
    viz_suite.create_business_dashboard(df, cluster_labels, profiles)
    
    return viz_suite


if __name__ == "__main__":
    # Test the visualization suite
    from data_loader import load_sample_data
    from preprocessor import quick_preprocess
    from clustering import CustomerClusteringAnalyzer
    
    # Load and preprocess data
    df = load_sample_data()
    df_processed, X = quick_preprocess(df)
    
    # Perform clustering
    analyzer = CustomerClusteringAnalyzer()
    kmeans_results = analyzer.perform_kmeans_clustering(X)
    
    # Analyze cluster profiles
    feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    profiles = analyzer.analyze_cluster_profiles(df, kmeans_results['labels'], feature_names)
    
    # Create visualizations
    viz_suite = create_comprehensive_visualization_suite(
        df, kmeans_results['labels'], X, profiles, feature_names=feature_names
    )
    
    print("Visualization suite completed!")
    print(f"Total plots created: {len(viz_suite.plots_created)}")
