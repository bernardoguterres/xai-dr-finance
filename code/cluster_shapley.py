"""
ClusterShapley Implementation
Explaining t-SNE and UMAP dimensionality reduction using cluster-based SHAP.

Based on: Marcílio-Jr & Eler (2021) "Explaining dimensionality reduction 
results using Shapley values" - Expert Systems with Applications

Author: Bernardo Guterres
Date: December 2025

Instead of explaining embedding coordinates directly (which is
mathematically intractable for t-SNE/UMAP), explain CLUSTER MEMBERSHIP.
This reveals which features drive the visual separation in embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available. Install with: pip install hdbscan")

np.random.seed(42)

# ============================================================================
# SECTION 1: ClusterShapley Core Implementation
# ============================================================================

class ClusterShapley:
    """
    Explain dimensionality reduction results using cluster-based SHAP values.
    
    Pipeline:
    1. Apply DR (t-SNE, UMAP) to get 2D embedding
    2. Cluster the embedded points (HDBSCAN, KMeans)
    3. Train classifier: original features → cluster labels
    4. Apply SHAP to explain cluster membership
    
    This reveals which original features drive the visual groupings.
    """
    
    def __init__(self, 
                 dr_method='tsne', 
                 cluster_method='kmeans',
                 n_clusters=5,
                 classifier='xgboost'):
        """
        Args:
            dr_method: 'tsne' or 'umap'
            cluster_method: 'kmeans', 'hdbscan', or 'dbscan'
            n_clusters: Number of clusters (ignored for HDBSCAN/DBSCAN)
            classifier: 'xgboost', 'rf', or 'gbm'
        """
        self.dr_method = dr_method.lower()
        self.cluster_method = cluster_method.lower()
        self.n_clusters = n_clusters
        self.classifier_type = classifier.lower()
        
        self.dr_model = None
        self.clusterer = None
        self.classifier = None
        self.embedding = None
        self.cluster_labels = None
        self.explainer = None
        self.feature_names = None
        
    def fit(self, X, feature_names=None, dr_params=None, cluster_params=None):
        """
        Fit the complete ClusterShapley pipeline.
        
        Args:
            X: Input features (n_samples, n_features)
            feature_names: List of feature names
            dr_params: Dict of parameters for DR method
            cluster_params: Dict of parameters for clustering
        """
        self.feature_names = feature_names if feature_names is not None else \
                            [f'Feature_{i}' for i in range(X.shape[1])]
        self._X = X
        
        # Step 1: Dimensionality Reduction
        print("Step 1: Applying dimensionality reduction...")
        self._fit_dr(X, dr_params)
        
        # Step 2: Clustering
        print("Step 2: Clustering in embedding space...")
        self._fit_clustering(cluster_params)
        
        # Step 3: Train classifier
        print("Step 3: Training classifier (features → clusters)...")
        self._fit_classifier(X)
        
        # Step 4: Create SHAP explainer
        print("Step 4: Creating SHAP explainer...")
        self._create_explainer(X)
        
        print("ClusterShapley pipeline fitted successfully!")
        return self
    
    def _fit_dr(self, X, params=None):
        """Fit dimensionality reduction."""
        params = params or {}
        
        if self.dr_method == 'tsne':
            default_params = {
                'n_components': 2,
                'perplexity': 30,
                'learning_rate': 200,
                'n_iter': 1000,
                'random_state': 42
            }
            default_params.update(params)
            self.dr_model = TSNE(**default_params)
            
        elif self.dr_method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed")
            default_params = {
                'n_components': 2,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'random_state': 42
            }
            default_params.update(params)
            self.dr_model = umap.UMAP(**default_params)
        else:
            raise ValueError(f"Unknown DR method: {self.dr_method}")
        
        self.embedding = self.dr_model.fit_transform(X)
        print(f"  Embedding shape: {self.embedding.shape}")
    
    def _fit_clustering(self, params=None):
        """Fit clustering on embedding."""
        params = params or {}
        
        if self.cluster_method == 'kmeans':
            default_params = {'n_clusters': self.n_clusters, 'random_state': 42}
            default_params.update(params)
            self.clusterer = KMeans(**default_params)
            
        elif self.cluster_method == 'hdbscan':
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN not installed")
            default_params = {'min_cluster_size': 50, 'min_samples': 10}
            default_params.update(params)
            self.clusterer = HDBSCAN(**default_params)
            
        elif self.cluster_method == 'dbscan':
            default_params = {'eps': 0.5, 'min_samples': 10}
            default_params.update(params)
            self.clusterer = DBSCAN(**default_params)
        else:
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")
        
        self.cluster_labels = self.clusterer.fit_predict(self.embedding)
        
        # Handle noise points in HDBSCAN/DBSCAN
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    
    def _fit_classifier(self, X):
        """Train classifier to predict cluster labels from original features."""
        # Remove noise points for classification
        valid_mask = self.cluster_labels != -1
        X_valid = X[valid_mask]
        y_valid = self.cluster_labels[valid_mask]
        
        if self.classifier_type == 'xgboost':
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.classifier_type == 'gbm':
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_type}")
        
        self.classifier.fit(X_valid, y_valid)
        
        # Evaluate classifier
        cv_scores = cross_val_score(self.classifier, X_valid, y_valid, cv=5)
        self.classifier_accuracy = cv_scores.mean()
        print(f"  Classifier CV accuracy: {self.classifier_accuracy:.3f} (+/- {cv_scores.std():.3f})")
    
    def _create_explainer(self, X):
        """Create TreeSHAP explainer."""
        # Use subset as background
        background_idx = np.random.choice(len(X), min(100, len(X)), replace=False)
        background = X[background_idx]
        
        if self.classifier_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.classifier)
        else:
            self.explainer = shap.TreeExplainer(self.classifier, data=background)
    
    def explain_clusters(self, X=None, sample_size=None):
        """
        Get SHAP explanations for cluster membership.
        
        Returns:
            shap_values: Array of shape (n_samples, n_features, n_clusters)
            For binary: (n_samples, n_features)
        """
        if X is None:
            X = self._X
        
        if sample_size is not None and sample_size < len(X):
            idx = np.random.choice(len(X), sample_size, replace=False)
            X = X[idx]
        
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def get_cluster_drivers(self, cluster_id, top_n=10):
        """
        Get top features driving membership in a specific cluster.
        
        Args:
            cluster_id: Which cluster to explain
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        # Get samples in this cluster
        cluster_mask = self.cluster_labels == cluster_id
        X_cluster = self._X[cluster_mask]
        
        if len(X_cluster) == 0:
            return pd.DataFrame()
        
        # Compute SHAP values
        shap_values = self.explain_clusters(X_cluster)
        
        # Handle multi-class vs binary
        if isinstance(shap_values, list):
            # Multi-class: get SHAP for this cluster
            cluster_shap = shap_values[cluster_id]
        else:
            cluster_shap = shap_values
        
        # Mean absolute SHAP
        mean_shap = np.abs(cluster_shap).mean(axis=0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_shap,
            'Mean_SHAP': cluster_shap.mean(axis=0)  # Signed
        })
        
        return results.sort_values('Importance', ascending=False).head(top_n)
    
    def get_global_feature_importance(self, top_n=None):
        """
        Get global feature importance across all clusters.
        """
        # Compute SHAP for all samples
        shap_values = self.explain_clusters(self._X, sample_size=500)
        
        # Aggregate across classes
        if isinstance(shap_values, list):
            # Sum absolute SHAP across all classes
            total_shap = np.zeros_like(shap_values[0])
            for sv in shap_values:
                total_shap += np.abs(sv)
            mean_shap = total_shap.mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        results = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False)
        
        if top_n:
            results = results.head(top_n)
        
        return results
    
    def explain_sample(self, sample_idx):
        """
        Explain why a specific sample belongs to its cluster.
        
        Returns:
            Dict with cluster assignment and top contributing features
        """
        sample = self._X[sample_idx:sample_idx+1]
        cluster = self.cluster_labels[sample_idx]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(sample)
        
        if isinstance(shap_values, list):
            sample_shap = shap_values[cluster][0]
        else:
            sample_shap = shap_values[0]
        
        # Get top features
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
        
        explanation = {
            'sample_idx': sample_idx,
            'cluster': cluster,
            'embedding': self.embedding[sample_idx],
            'top_features': []
        }
        
        for i in sorted_idx[:5]:
            explanation['top_features'].append({
                'feature': self.feature_names[i],
                'shap_value': sample_shap[i],
                'direction': 'increases' if sample_shap[i] > 0 else 'decreases'
            })
        
        return explanation
    
    def generate_cluster_rules(self, threshold_std=0.5):
        """
        Generate human-readable rules for each cluster.
        
        Returns:
            Dict mapping cluster_id → rule string
        """
        rules = {}
        
        unique_clusters = sorted(set(self.cluster_labels) - {-1})
        
        for cluster_id in unique_clusters:
            drivers = self.get_cluster_drivers(cluster_id, top_n=5)
            
            if len(drivers) == 0:
                continue
            
            rule_parts = []
            for _, row in drivers.iterrows():
                if row['Mean_SHAP'] > threshold_std:
                    rule_parts.append(f"HIGH {row['Feature']}")
                elif row['Mean_SHAP'] < -threshold_std:
                    rule_parts.append(f"LOW {row['Feature']}")
            
            if rule_parts:
                rules[cluster_id] = f"Cluster {cluster_id}: " + " AND ".join(rule_parts[:3])
            else:
                rules[cluster_id] = f"Cluster {cluster_id}: (no strong drivers)"
        
        return rules
    
    def visualize(self, save_path='cluster_shapley.png'):
        """
        Create comprehensive visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Embedding with cluster colors
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            self.embedding[:, 0], 
            self.embedding[:, 1],
            c=self.cluster_labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        ax1.set_xlabel(f'{self.dr_method.upper()} 1')
        ax1.set_ylabel(f'{self.dr_method.upper()} 2')
        ax1.set_title(f'{self.dr_method.upper()} Embedding with Clusters')
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # Plot 2: Global feature importance
        ax2 = axes[0, 1]
        global_imp = self.get_global_feature_importance(top_n=10)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(global_imp)))
        ax2.barh(global_imp['Feature'], global_imp['Importance'], color=colors)
        ax2.set_xlabel('Mean |SHAP|')
        ax2.set_title('Global Feature Importance for Clustering')
        ax2.invert_yaxis()
        
        # Plot 3: Per-cluster feature importance heatmap
        ax3 = axes[1, 0]
        unique_clusters = sorted(set(self.cluster_labels) - {-1})[:6]  # Limit to 6 clusters
        
        importance_matrix = []
        for cluster_id in unique_clusters:
            drivers = self.get_cluster_drivers(cluster_id, top_n=10)
            if len(drivers) > 0:
                imp_dict = dict(zip(drivers['Feature'], drivers['Mean_SHAP']))
                importance_matrix.append([imp_dict.get(f, 0) for f in self.feature_names[:10]])
        
        if importance_matrix:
            sns.heatmap(
                np.array(importance_matrix),
                xticklabels=self.feature_names[:10],
                yticklabels=[f'Cluster {c}' for c in unique_clusters[:len(importance_matrix)]],
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                ax=ax3
            )
            ax3.set_title('Cluster-Specific Feature Effects')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Cluster rules
        ax4 = axes[1, 1]
        ax4.axis('off')
        rules = self.generate_cluster_rules()
        
        rule_text = "CLUSTER INTERPRETATION RULES\n" + "="*40 + "\n\n"
        for cluster_id, rule in rules.items():
            rule_text += f"{rule}\n\n"
        
        ax4.text(0.1, 0.9, rule_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax4.set_title('Logical Rules for Clusters')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()
        
        return fig

# ============================================================================
# SECTION 2: Financial Data Generator
# ============================================================================

def generate_financial_clusters(n_samples=2000, n_features=15):
    """
    Generate synthetic financial data with distinct cluster structure.

    Clusters represent:
    - Cluster 0: Growth stocks (high revenue growth, high PE, low dividend)
    - Cluster 1: Value stocks (low PE, high dividend, moderate debt)
    - Cluster 2: Quality stocks (high ROE, low debt, stable earnings)
    - Cluster 3: Momentum stocks (high price momentum, high volume)
    - Cluster 4: Distressed stocks (high debt, negative earnings)

    UNREALISTIC CLUSTER SEPARATION WARNING: These clusters are generated with
    well-separated multivariate normals (distinct means, small within-cluster
    variance). Real financial data does NOT form such clean clusters — stock
    characteristics overlap heavily across styles (e.g., many growth stocks also
    have high quality metrics). The small within-cluster standard deviations
    (scale parameters) relative to the between-cluster mean differences produce
    near-perfect cluster separability, leading to:
    - Classifier CV accuracy > 0.95 (vs. 0.6–0.75 typical on real data)
    - ARI scores > 0.9 (vs. 0.3–0.5 on real stock data)
    - ClusterShapley feature importance rankings that are trivially correct
    These results demonstrate the method works on clean data but do not reflect
    performance on real, noisy, overlapping financial data distributions.
    """
    np.random.seed(42)
    
    n_per_cluster = n_samples // 5
    
    feature_names = [
        'Revenue_Growth', 'EPS_Growth', 'PE_Ratio', 'PB_Ratio', 'Dividend_Yield',
        'ROE', 'ROA', 'Debt_to_Equity', 'Current_Ratio', 'Operating_Margin',
        'Price_Momentum_3M', 'Price_Momentum_12M', 'Volume_Trend', 'Volatility', 'Beta'
    ]
    
    clusters = []
    labels = []
    
    # Cluster 0: Growth stocks
    growth = np.random.normal(loc=[0.25, 0.30, 35, 6, 0.005, 0.15, 0.08, 0.8, 1.5, 0.15, 0.15, 0.25, 0.1, 0.35, 1.3],
                              scale=[0.1, 0.15, 10, 2, 0.005, 0.05, 0.03, 0.3, 0.3, 0.05, 0.1, 0.15, 0.05, 0.1, 0.2],
                              size=(n_per_cluster, n_features))
    clusters.append(growth)
    labels.extend([0] * n_per_cluster)
    
    # Cluster 1: Value stocks  
    value = np.random.normal(loc=[0.03, 0.05, 12, 1.2, 0.04, 0.10, 0.06, 0.5, 2.0, 0.12, 0.02, 0.05, -0.02, 0.20, 0.8],
                             scale=[0.03, 0.05, 3, 0.3, 0.015, 0.03, 0.02, 0.2, 0.4, 0.03, 0.05, 0.08, 0.03, 0.05, 0.15],
                             size=(n_per_cluster, n_features))
    clusters.append(value)
    labels.extend([1] * n_per_cluster)
    
    # Cluster 2: Quality stocks
    quality = np.random.normal(loc=[0.08, 0.10, 22, 4, 0.02, 0.25, 0.15, 0.3, 2.5, 0.22, 0.08, 0.12, 0.03, 0.18, 0.9],
                               scale=[0.04, 0.05, 5, 1, 0.01, 0.05, 0.04, 0.1, 0.5, 0.05, 0.05, 0.08, 0.02, 0.04, 0.1],
                               size=(n_per_cluster, n_features))
    clusters.append(quality)
    labels.extend([2] * n_per_cluster)
    
    # Cluster 3: Momentum stocks
    momentum = np.random.normal(loc=[0.12, 0.15, 28, 5, 0.01, 0.12, 0.07, 0.6, 1.8, 0.14, 0.35, 0.50, 0.25, 0.40, 1.5],
                                scale=[0.08, 0.1, 8, 1.5, 0.008, 0.04, 0.03, 0.25, 0.35, 0.04, 0.12, 0.15, 0.1, 0.1, 0.25],
                                size=(n_per_cluster, n_features))
    clusters.append(momentum)
    labels.extend([3] * n_per_cluster)
    
    # Cluster 4: Distressed stocks
    distressed = np.random.normal(loc=[-0.05, -0.15, 8, 0.6, 0.0, -0.05, -0.02, 2.5, 0.8, 0.02, -0.15, -0.25, -0.1, 0.50, 1.8],
                                   scale=[0.08, 0.1, 5, 0.3, 0.005, 0.08, 0.05, 0.8, 0.3, 0.04, 0.12, 0.15, 0.08, 0.12, 0.3],
                                   size=(n_per_cluster, n_features))
    clusters.append(distressed)
    labels.extend([4] * n_per_cluster)
    
    X = np.vstack(clusters)
    y = np.array(labels)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    cluster_names = {
        0: 'Growth',
        1: 'Value', 
        2: 'Quality',
        3: 'Momentum',
        4: 'Distressed'
    }
    
    return df, y, feature_names, cluster_names

# ============================================================================
# SECTION 3: Advanced Analysis
# ============================================================================

def compare_dr_methods_clustering(X, feature_names, true_labels):
    """
    Compare how well different DR methods preserve cluster structure.
    """
    results = []
    
    methods = [('tsne', {}), ('umap', {})] if UMAP_AVAILABLE else [('tsne', {})]
    cluster_methods = ['kmeans']
    if HDBSCAN_AVAILABLE:
        cluster_methods.append('hdbscan')
    
    for dr_method, dr_params in methods:
        for cluster_method in cluster_methods:
            print(f"\nTesting {dr_method.upper()} + {cluster_method.upper()}...")
            
            try:
                cs = ClusterShapley(
                    dr_method=dr_method,
                    cluster_method=cluster_method,
                    n_clusters=5
                )
                cs.fit(X, feature_names)
                
                # Measure cluster quality
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                
                # Compare discovered clusters to true labels
                valid_mask = cs.cluster_labels != -1
                ari = adjusted_rand_score(true_labels[valid_mask], cs.cluster_labels[valid_mask])
                nmi = normalized_mutual_info_score(true_labels[valid_mask], cs.cluster_labels[valid_mask])
                
                # Get global importance
                global_imp = cs.get_global_feature_importance(top_n=5)
                
                results.append({
                    'DR_Method': dr_method.upper(),
                    'Cluster_Method': cluster_method.upper(),
                    'Classifier_Accuracy': cs.classifier_accuracy,
                    'ARI_vs_True': ari,
                    'NMI_vs_True': nmi,
                    'Top_Features': list(global_imp['Feature'])
                })
                
            except Exception as e:
                print(f"Error: {e}")
    
    return pd.DataFrame(results)

def stability_analysis(X, feature_names, n_runs=5, dr_method='tsne'):
    """
    Analyze stability of ClusterShapley explanations across runs.
    """
    feature_rankings = []
    
    for seed in range(n_runs):
        np.random.seed(seed)
        
        cs = ClusterShapley(dr_method=dr_method, n_clusters=5)
        cs.fit(X, feature_names)
        
        global_imp = cs.get_global_feature_importance()
        ranking = {row['Feature']: i for i, row in global_imp.iterrows()}
        feature_rankings.append(ranking)
    
    # Compute rank correlation between runs
    from scipy.stats import spearmanr
    
    correlations = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            ranks_i = [feature_rankings[i].get(f, len(feature_names)) for f in feature_names]
            ranks_j = [feature_rankings[j].get(f, len(feature_names)) for f in feature_names]
            corr, _ = spearmanr(ranks_i, ranks_j)
            correlations.append(corr)
    
    return {
        'mean_rank_correlation': np.mean(correlations),
        'std_rank_correlation': np.std(correlations),
        'min_rank_correlation': np.min(correlations),
        'all_correlations': correlations
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CLUSTERSHAPLEY - EXPLAINING t-SNE AND UMAP")
    print("=" * 70)
    
    # Generate synthetic financial data
    print("\n1. Generating synthetic financial data with cluster structure...")
    df, true_labels, feature_names, cluster_names = generate_financial_clusters(n_samples=2000)
    
    print(f"Data shape: {df.shape}")
    print(f"True clusters: {cluster_names}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    # Run ClusterShapley with t-SNE
    print("\n2. Running ClusterShapley with t-SNE...")
    cs_tsne = ClusterShapley(dr_method='tsne', cluster_method='kmeans', n_clusters=5)
    cs_tsne.fit(X, feature_names)
    
    # Get global feature importance
    print("\n3. Global feature importance for clustering:")
    global_imp = cs_tsne.get_global_feature_importance(top_n=10)
    print(global_imp.to_string(index=False))
    
    # Get cluster-specific drivers
    print("\n4. Cluster-specific feature drivers:")
    for cluster_id in range(5):
        print(f"\nCluster {cluster_id} ({cluster_names.get(cluster_id, 'Unknown')}):")
        drivers = cs_tsne.get_cluster_drivers(cluster_id, top_n=5)
        if len(drivers) > 0:
            for _, row in drivers.iterrows():
                direction = "↑" if row['Mean_SHAP'] > 0 else "↓"
                print(f"  {direction} {row['Feature']}: {row['Mean_SHAP']:.3f}")
    
    # Generate logical rules
    print("\n5. Generated logical rules:")
    rules = cs_tsne.generate_cluster_rules()
    for cluster_id, rule in rules.items():
        print(f"  {rule}")
    
    # Explain individual sample
    print("\n6. Sample explanation:")
    sample_exp = cs_tsne.explain_sample(sample_idx=42)
    print(f"  Sample 42 belongs to Cluster {sample_exp['cluster']}")
    print(f"  Top features driving this assignment:")
    for feat in sample_exp['top_features'][:3]:
        print(f"    • {feat['feature']} {feat['direction']} cluster membership ({feat['shap_value']:.3f})")
    
    # Visualize
    print("\n7. Creating visualization...")
    cs_tsne.visualize(save_path='/home/claude/cluster_shapley_tsne.png')
    
    # Run with UMAP if available
    if UMAP_AVAILABLE:
        print("\n8. Running ClusterShapley with UMAP...")
        cs_umap = ClusterShapley(dr_method='umap', cluster_method='kmeans', n_clusters=5)
        cs_umap.fit(X, feature_names)
        cs_umap.visualize(save_path='/home/claude/cluster_shapley_umap.png')
    
    # Stability analysis
    print("\n9. Stability analysis (may take a minute)...")
    stability = stability_analysis(X, feature_names, n_runs=3, dr_method='tsne')
    print(f"  Mean rank correlation: {stability['mean_rank_correlation']:.3f}")
    print(f"  Std rank correlation: {stability['std_rank_correlation']:.3f}")
    
    # Compare methods
    print("\n10. Comparing DR methods...")
    comparison = compare_dr_methods_clustering(X, feature_names, true_labels)
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison.to_csv('/home/claude/cluster_shapley_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("CLUSTERSHAPLEY ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  • cluster_shapley_tsne.png")
    if UMAP_AVAILABLE:
        print("  • cluster_shapley_umap.png")
    print("  • cluster_shapley_comparison.csv")
    
    print("\nKey findings:")
    print(f"  • Classifier accuracy: {cs_tsne.classifier_accuracy:.3f}")
    print(f"  • Top 3 features: {list(global_imp['Feature'][:3])}")
    print(f"  • Explanation stability: {stability['mean_rank_correlation']:.3f}")
