"""
Clustering Module for CUSTO CLARITY Project

This module contains clustering algorithms and analysis functions for customer segmentation.
Implements K-Means, DBSCAN, and other clustering techniques with evaluation metrics.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerClusteringAnalyzer:
    """Comprehensive clustering analysis class for customer segmentation."""
    
    def __init__(self):
        """Initialize the clustering analyzer."""
        self.models = {}
        self.evaluation_metrics = {}
        self.cluster_profiles = {}
        self.dimensionality_reduction = {}
        
    def find_optimal_clusters_kmeans(self, X: np.ndarray, 
                                   max_clusters: int = 10,
                                   random_state: int = 42) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            X (np.ndarray): Feature matrix
            max_clusters (int): Maximum number of clusters to test
            random_state (int): Random state for reproducibility
        
        Returns:
            Dict[str, Any]: Results containing inertias, silhouette scores, and optimal k
        """
        logger.info("Finding optimal number of clusters using elbow method and silhouette analysis...")
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            # Fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(X, cluster_labels))
        
        # Find optimal k using elbow method (simplified)
        # Calculate the rate of change in inertia
        inertia_diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        elbow_k = k_range[np.argmax(inertia_diffs)] + 1
        
        # Find optimal k using silhouette score
        silhouette_optimal_k = k_range[np.argmax(silhouette_scores)]
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'elbow_optimal_k': elbow_k,
            'silhouette_optimal_k': silhouette_optimal_k,
            'recommended_k': silhouette_optimal_k  # Prefer silhouette score
        }
        
        logger.info(f"Optimal clusters - Elbow method: {elbow_k}, Silhouette: {silhouette_optimal_k}")
        return results
    
    def perform_kmeans_clustering(self, X: np.ndarray, 
                                n_clusters: Optional[int] = None,
                                random_state: int = 42) -> Dict[str, Any]:
        """
        Perform K-Means clustering analysis.
        
        Args:
            X (np.ndarray): Feature matrix
            n_clusters (int, optional): Number of clusters. If None, finds optimal number.
            random_state (int): Random state for reproducibility
        
        Returns:
            Dict[str, Any]: Clustering results
        """
        logger.info("Performing K-Means clustering analysis...")
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            optimal_results = self.find_optimal_clusters_kmeans(X)
            n_clusters = optimal_results['recommended_k']
            logger.info(f"Using optimal number of clusters: {n_clusters}")
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate evaluation metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        # Store results
        results = {
            'model': kmeans,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        }
        
        self.models['kmeans'] = results
        logger.info(f"K-Means clustering completed. Silhouette Score: {silhouette_avg:.3f}")
        
        return results
    
    def perform_dbscan_clustering(self, X: np.ndarray, 
                                eps: float = 0.5, 
                                min_samples: int = 5) -> Dict[str, Any]:
        """
        Perform DBSCAN clustering analysis.
        
        Args:
            X (np.ndarray): Feature matrix
            eps (float): The maximum distance between two samples for one to be considered 
                        as in the neighborhood of the other
            min_samples (int): The number of samples in a neighborhood for a point to be 
                             considered as a core point
        
        Returns:
            Dict[str, Any]: Clustering results
        """
        logger.info("Performing DBSCAN clustering analysis...")
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        # Calculate metrics (excluding noise points for silhouette score)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate silhouette score only if we have more than 1 cluster and non-noise points
        silhouette_avg = None
        calinski_harabasz = None
        davies_bouldin = None
        
        if n_clusters > 1:
            # Create mask for non-noise points
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                X_non_noise = X[non_noise_mask]
                labels_non_noise = cluster_labels[non_noise_mask]
                
                if len(set(labels_non_noise)) > 1:
                    silhouette_avg = silhouette_score(X_non_noise, labels_non_noise)
                    calinski_harabasz = calinski_harabasz_score(X_non_noise, labels_non_noise)
                    davies_bouldin = davies_bouldin_score(X_non_noise, labels_non_noise)
        
        # Store results
        results = {
            'model': dbscan,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        }
        
        self.models['dbscan'] = results
        logger.info(f"DBSCAN clustering completed. Clusters: {n_clusters}, Noise points: {n_noise}")
        
        return results
    
    def optimize_dbscan_parameters(self, X: np.ndarray,
                                 eps_range: List[float] = None,
                                 min_samples_range: List[int] = None) -> Dict[str, Any]:
        """
        Optimize DBSCAN parameters using grid search.
        
        Args:
            X (np.ndarray): Feature matrix
            eps_range (List[float]): Range of eps values to test
            min_samples_range (List[int]): Range of min_samples values to test
        
        Returns:
            Dict[str, Any]: Optimization results with best parameters
        """
        logger.info("Optimizing DBSCAN parameters...")
        
        if eps_range is None:
            eps_range = [0.3, 0.5, 0.7, 1.0, 1.5]
        
        if min_samples_range is None:
            min_samples_range = [3, 5, 7, 10]
        
        best_score = -1
        best_params = {}
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Calculate silhouette score for valid clustering
                silhouette_avg = -1  # Default for invalid clustering
                if n_clusters > 1:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        X_non_noise = X[non_noise_mask]
                        labels_non_noise = labels[non_noise_mask]
                        if len(set(labels_non_noise)) > 1:
                            silhouette_avg = silhouette_score(X_non_noise, labels_non_noise)
                
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': silhouette_avg
                }
                results.append(result)
                
                # Update best parameters based on silhouette score and reasonable cluster count
                if (silhouette_avg > best_score and 
                    2 <= n_clusters <= 8 and 
                    n_noise < len(X) * 0.1):  # Less than 10% noise
                    best_score = silhouette_avg
                    best_params = {'eps': eps, 'min_samples': min_samples}
        
        optimization_results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
        
        logger.info(f"Best DBSCAN parameters: {best_params} with silhouette score: {best_score:.3f}")
        return optimization_results
    
    def perform_hierarchical_clustering(self, X: np.ndarray,
                                      n_clusters: int = 4,
                                      linkage: str = 'ward') -> Dict[str, Any]:
        """
        Perform Hierarchical (Agglomerative) clustering analysis.
        
        Args:
            X (np.ndarray): Feature matrix
            n_clusters (int): Number of clusters
            linkage (str): Linkage criterion ('ward', 'complete', 'average', 'single')
        
        Returns:
            Dict[str, Any]: Clustering results
        """
        logger.info("Performing Hierarchical clustering analysis...")
        
        # Fit Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = hierarchical.fit_predict(X)
        
        # Calculate evaluation metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        # Store results
        results = {
            'model': hierarchical,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'linkage': linkage,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        }
        
        self.models['hierarchical'] = results
        logger.info(f"Hierarchical clustering completed. Silhouette Score: {silhouette_avg:.3f}")
        
        return results
    
    def apply_dimensionality_reduction(self, X: np.ndarray,
                                     methods: List[str] = ['pca', 'tsne']) -> Dict[str, Any]:
        """
        Apply dimensionality reduction techniques for visualization.
        
        Args:
            X (np.ndarray): Feature matrix
            methods (List[str]): Methods to apply ('pca', 'tsne')
        
        Returns:
            Dict[str, Any]: Reduced dimension datasets
        """
        logger.info("Applying dimensionality reduction techniques...")
        
        results = {}
        
        if 'pca' in methods:
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            
            results['pca'] = {
                'data': X_pca,
                'model': pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'total_variance_explained': sum(pca.explained_variance_ratio_)
            }
            
            logger.info(f"PCA completed. Explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        
        if 'tsne' in methods:
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
            X_tsne = tsne.fit_transform(X)
            
            results['tsne'] = {
                'data': X_tsne,
                'model': tsne
            }
            
            logger.info("t-SNE completed")
        
        self.dimensionality_reduction = results
        return results
    
    def analyze_cluster_profiles(self, df: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               feature_columns: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Analyze cluster profiles and characteristics.
        
        Args:
            df (pd.DataFrame): Original dataset
            cluster_labels (np.ndarray): Cluster labels
            feature_columns (List[str]): Columns to analyze
        
        Returns:
            Dict[str, pd.DataFrame]: Cluster profile analysis
        """
        logger.info("Analyzing cluster profiles...")
        
        # Add cluster labels to dataframe
        df_analysis = df.copy()
        df_analysis['Cluster'] = cluster_labels
        
        # Calculate cluster statistics
        numeric_cols = [col for col in feature_columns if col in df_analysis.columns 
                       and df_analysis[col].dtype in ['int64', 'float64']]
        categorical_cols = [col for col in feature_columns if col in df_analysis.columns 
                          and df_analysis[col].dtype == 'object']
        
        profiles = {}
        
        # Numeric feature analysis
        if numeric_cols:
            numeric_profile = df_analysis.groupby('Cluster')[numeric_cols].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ]).round(2)
            profiles['numeric'] = numeric_profile
        
        # Categorical feature analysis
        if categorical_cols:
            categorical_profiles = {}
            for col in categorical_cols:
                cat_profile = df_analysis.groupby('Cluster')[col].value_counts(normalize=True).unstack(fill_value=0)
                categorical_profiles[col] = cat_profile
            profiles['categorical'] = categorical_profiles
        
        # Cluster sizes
        cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()
        profiles['cluster_sizes'] = cluster_sizes
        
        # Cluster percentages
        cluster_percentages = (cluster_sizes / len(df_analysis) * 100).round(2)
        profiles['cluster_percentages'] = cluster_percentages
        
        self.cluster_profiles = profiles
        logger.info(f"Cluster profile analysis completed for {len(set(cluster_labels))} clusters")
        
        return profiles
    
    def generate_business_insights(self, profiles: Dict[str, Any],
                                 cluster_labels: np.ndarray) -> Dict[str, str]:
        """
        Generate business insights based on cluster analysis.
        
        Args:
            profiles (Dict[str, Any]): Cluster profiles from analyze_cluster_profiles
            cluster_labels (np.ndarray): Cluster labels
        
        Returns:
            Dict[str, str]: Business insights for each cluster
        """
        logger.info("Generating business insights...")
        
        insights = {}
        n_clusters = len(set(cluster_labels))
        
        if 'numeric' in profiles:
            numeric_profile = profiles['numeric']
            
            for cluster in range(n_clusters):
                if cluster in numeric_profile.index:
                    cluster_data = numeric_profile.loc[cluster]
                    
                    # Extract key metrics (assuming standard column names)
                    age_mean = cluster_data.get(('Age', 'mean'), 0)
                    income_mean = cluster_data.get(('Annual Income (k$)', 'mean'), 0)
                    spending_mean = cluster_data.get(('Spending Score (1-100)', 'mean'), 0)
                    
                    # Generate insights based on characteristics
                    insight = f"Cluster {cluster}:\n"
                    
                    # Age characterization
                    if age_mean < 30:
                        age_desc = "Young customers"
                    elif age_mean < 50:
                        age_desc = "Middle-aged customers"
                    else:
                        age_desc = "Mature customers"
                    
                    # Income characterization
                    if income_mean < 40:
                        income_desc = "low income"
                    elif income_mean < 70:
                        income_desc = "moderate income"
                    else:
                        income_desc = "high income"
                    
                    # Spending characterization
                    if spending_mean < 35:
                        spending_desc = "conservative spenders"
                    elif spending_mean < 65:
                        spending_desc = "moderate spenders"
                    else:
                        spending_desc = "high spenders"
                    
                    insight += f"- {age_desc} with {income_desc} who are {spending_desc}\n"
                    insight += f"- Average age: {age_mean:.1f}, Income: ${income_mean:.1f}k, Spending: {spending_mean:.1f}/100\n"
                    
                    # Marketing recommendations
                    if spending_mean >= 65 and income_mean >= 60:
                        insight += "- Recommendation: Premium products, luxury marketing, loyalty programs\n"
                    elif spending_mean < 35:
                        insight += "- Recommendation: Value products, discount campaigns, budget-friendly options\n"
                    elif age_mean < 30:
                        insight += "- Recommendation: Trendy products, social media marketing, experience-based offers\n"
                    else:
                        insight += "- Recommendation: Balanced product mix, targeted promotions, personalized offers\n"
                    
                    cluster_size = profiles.get('cluster_sizes', {}).get(cluster, 0)
                    cluster_pct = profiles.get('cluster_percentages', {}).get(cluster, 0)
                    insight += f"- Segment size: {cluster_size} customers ({cluster_pct}% of total)\n"
                    
                    insights[f"cluster_{cluster}"] = insight
        
        logger.info(f"Business insights generated for {len(insights)} clusters")
        return insights
    
    def compare_clustering_methods(self, X: np.ndarray) -> pd.DataFrame:
        """
        Compare different clustering methods and their performance.
        
        Args:
            X (np.ndarray): Feature matrix
        
        Returns:
            pd.DataFrame: Comparison of clustering methods
        """
        logger.info("Comparing clustering methods...")
        
        comparison_results = []
        
        # K-Means
        if 'kmeans' not in self.models:
            self.perform_kmeans_clustering(X)
        kmeans_results = self.models['kmeans']
        
        comparison_results.append({
            'Method': 'K-Means',
            'Number of Clusters': kmeans_results['n_clusters'],
            'Silhouette Score': kmeans_results['silhouette_score'],
            'Calinski-Harabasz Score': kmeans_results['calinski_harabasz_score'],
            'Davies-Bouldin Score': kmeans_results['davies_bouldin_score'],
            'Special Notes': f"Inertia: {kmeans_results['inertia']:.2f}"
        })
        
        # DBSCAN
        if 'dbscan' not in self.models:
            # Optimize parameters first
            optimization = self.optimize_dbscan_parameters(X)
            if optimization['best_params']:
                self.perform_dbscan_clustering(X, **optimization['best_params'])
            else:
                self.perform_dbscan_clustering(X)
        
        dbscan_results = self.models['dbscan']
        comparison_results.append({
            'Method': 'DBSCAN',
            'Number of Clusters': dbscan_results['n_clusters'],
            'Silhouette Score': dbscan_results['silhouette_score'],
            'Calinski-Harabasz Score': dbscan_results['calinski_harabasz_score'],
            'Davies-Bouldin Score': dbscan_results['davies_bouldin_score'],
            'Special Notes': f"Noise points: {dbscan_results['n_noise']}"
        })
        
        # Hierarchical
        if 'hierarchical' not in self.models:
            # Use same number of clusters as K-Means for fair comparison
            self.perform_hierarchical_clustering(X, n_clusters=kmeans_results['n_clusters'])
        
        hierarchical_results = self.models['hierarchical']
        comparison_results.append({
            'Method': 'Hierarchical',
            'Number of Clusters': hierarchical_results['n_clusters'],
            'Silhouette Score': hierarchical_results['silhouette_score'],
            'Calinski-Harabasz Score': hierarchical_results['calinski_harabasz_score'],
            'Davies-Bouldin Score': hierarchical_results['davies_bouldin_score'],
            'Special Notes': f"Linkage: {hierarchical_results['linkage']}"
        })
        
        comparison_df = pd.DataFrame(comparison_results)
        logger.info("Clustering methods comparison completed")
        
        return comparison_df


if __name__ == "__main__":
    # Test the clustering analyzer
    from data_loader import load_sample_data
    from preprocessor import quick_preprocess
    
    # Load and preprocess data
    df = load_sample_data()
    df_processed, X = quick_preprocess(df)
    
    # Initialize analyzer
    analyzer = CustomerClusteringAnalyzer()
    
    # Perform clustering analysis
    kmeans_results = analyzer.perform_kmeans_clustering(X)
    dbscan_results = analyzer.perform_dbscan_clustering(X)
    
    # Compare methods
    comparison = analyzer.compare_clustering_methods(X)
    print("Clustering Methods Comparison:")
    print(comparison)
