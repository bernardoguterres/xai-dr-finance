"""
Probing Classifiers for Embedding Interpretation
Transferred from NLP (BERTology) and genomics domains.

Author: Bernardo Guterres
Date: December 2025

Probing classifiers test what information is encoded in embeddings
by training lightweight classifiers on frozen embeddings to predict specific
properties. If a linear probe achieves high accuracy predicting a property,
that property is explicitly encoded in the embedding.

- Rogers et al. (2020): "A Primer in BERTology"
- Voita & Titov (2020): "Information-Theoretic Probing with MDL"
- Hewitt & Liang (2019): "Designing and Interpreting Probes"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try importing UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

np.random.seed(42)

# ============================================================================
# SECTION 1: Probing Classifier Framework
# ============================================================================

class ProbingClassifier:
    """
    Framework for probing what information is encoded in embeddings.
    
    Core idea: If a simple classifier can predict property P from embedding E,
    then P is encoded in E.
    """
    
    PROBE_TYPES = {
        'linear': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'mlp': lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
        'svm': lambda: LinearSVC(max_iter=5000, random_state=42),
        'rf': lambda: RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }
    
    def __init__(self, probe_type='linear'):
        """
        Args:
            probe_type: 'linear', 'mlp', 'svm', or 'rf'
                       Linear probes are preferred as they test if info is
                       linearly accessible (truly "encoded" vs just "present")
        """
        self.probe_type = probe_type
        self.probe = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        
    def fit(self, embeddings, labels):
        """
        Train probe to predict labels from embeddings.
        
        Args:
            embeddings: (n_samples, embedding_dim) array
            labels: (n_samples,) array of labels (categorical or continuous)
        """
        # Scale embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Encode labels if categorical
        if not np.issubdtype(type(labels[0]), np.floating):
            self.label_encoder = LabelEncoder()
            labels_encoded = self.label_encoder.fit_transform(labels)
        else:
            labels_encoded = labels
        
        # Create and fit probe
        self.probe = self.PROBE_TYPES[self.probe_type]()
        self.probe.fit(embeddings_scaled, labels_encoded)
        
        return self
    
    def evaluate(self, embeddings, labels, cv=5):
        """
        Evaluate probe with cross-validation.
        
        Returns:
            Dict with accuracy, f1, and baseline metrics
        """
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        if not np.issubdtype(type(labels[0]), np.floating):
            le = LabelEncoder()
            labels_encoded = le.fit_transform(labels)
        else:
            labels_encoded = labels
        
        # Cross-validation
        probe = self.PROBE_TYPES[self.probe_type]()
        
        cv_scores = cross_val_score(
            probe, embeddings_scaled, labels_encoded,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # Random baseline
        n_classes = len(np.unique(labels_encoded))
        random_baseline = 1.0 / n_classes
        
        # Majority baseline
        _, counts = np.unique(labels_encoded, return_counts=True)
        majority_baseline = counts.max() / counts.sum()
        
        return {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'random_baseline': random_baseline,
            'majority_baseline': majority_baseline,
            'selectivity': cv_scores.mean() - majority_baseline,  # How much better than baseline
            'cv_scores': cv_scores
        }
    
    def predict(self, embeddings):
        """Predict labels from embeddings."""
        embeddings_scaled = self.scaler.transform(embeddings)
        predictions = self.probe.predict(embeddings_scaled)
        
        if self.label_encoder:
            return self.label_encoder.inverse_transform(predictions)
        return predictions
    
    def get_importance(self, feature_names=None):
        """
        Get importance of each embedding dimension for prediction.
        Only available for linear and RF probes.
        """
        if self.probe_type == 'linear':
            importances = np.abs(self.probe.coef_).mean(axis=0)
        elif self.probe_type == 'rf':
            importances = self.probe.feature_importances_
        else:
            return None
        
        if feature_names:
            return pd.Series(importances, index=feature_names)
        return importances

class ProbeRegressor:
    """Regression version for continuous properties."""
    
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
    
    def evaluate(self, embeddings, targets, cv=5):
        """Evaluate regression probe."""
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        cv_scores = cross_val_score(
            Ridge(alpha=1.0), embeddings_scaled, targets, cv=cv, scoring='r2'
        )
        
        return {
            'r2_mean': cv_scores.mean(),
            'r2_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

# ============================================================================
# SECTION 2: Comprehensive Probing Suite
# ============================================================================

class EmbeddingProber:
    """
    Comprehensive probing analysis for embeddings.
    Tests multiple properties and probe types.
    """
    
    def __init__(self, embeddings, embedding_name='Embedding'):
        """
        Args:
            embeddings: (n_samples, embedding_dim) array
            embedding_name: Name for reporting
        """
        self.embeddings = embeddings
        self.embedding_name = embedding_name
        self.results = {}
    
    def probe_property(self, labels, property_name, probe_types=['linear', 'mlp']):
        """
        Probe for a specific property using multiple probe types.
        
        Args:
            labels: Labels to predict
            property_name: Name of property being probed
            probe_types: List of probe types to try
        """
        results = {'property': property_name}
        
        for pt in probe_types:
            probe = ProbingClassifier(probe_type=pt)
            eval_results = probe.evaluate(self.embeddings, labels)
            
            results[f'{pt}_accuracy'] = eval_results['accuracy_mean']
            results[f'{pt}_selectivity'] = eval_results['selectivity']
        
        results['baseline'] = eval_results['majority_baseline']
        
        self.results[property_name] = results
        return results
    
    def probe_regression(self, targets, property_name):
        """Probe for continuous property."""
        probe = ProbeRegressor()
        eval_results = probe.evaluate(self.embeddings, targets)
        
        self.results[property_name] = {
            'property': property_name,
            'r2': eval_results['r2_mean'],
            'r2_std': eval_results['r2_std'],
            'type': 'regression'
        }
        
        return self.results[property_name]
    
    def run_financial_probes(self, data_df, true_factors=None):
        """
        Run standard financial property probes.
        
        Args:
            data_df: DataFrame with original features
            true_factors: Dict mapping factor names to continuous values
        """
        print(f"\nProbing {self.embedding_name}...")
        print("=" * 60)
        
        # Market regime probe (based on momentum)
        if 'Price_Momentum_1M' in data_df.columns or 'Price_Mom_1M' in data_df.columns:
            mom_col = 'Price_Momentum_1M' if 'Price_Momentum_1M' in data_df.columns else 'Price_Mom_1M'
            regime_labels = pd.cut(
                data_df[mom_col], 
                bins=3, 
                labels=['Bear', 'Sideways', 'Bull']
            ).values
            self.probe_property(regime_labels, 'Market_Regime')
            print(f"  Market Regime: {self.results['Market_Regime']['linear_accuracy']:.3f} accuracy")
        
        # Quality tier (based on ROE)
        if 'ROE' in data_df.columns:
            quality_labels = pd.cut(
                data_df['ROE'],
                bins=3,
                labels=['Low', 'Medium', 'High']
            ).values
            self.probe_property(quality_labels, 'Quality_Tier')
            print(f"  Quality Tier: {self.results['Quality_Tier']['linear_accuracy']:.3f} accuracy")
        
        # Value classification (based on PE)
        if 'PE_Ratio' in data_df.columns:
            value_labels = pd.cut(
                data_df['PE_Ratio'],
                bins=3,
                labels=['Value', 'Fair', 'Growth']
            ).values
            self.probe_property(value_labels, 'Value_Style')
            print(f"  Value Style: {self.results['Value_Style']['linear_accuracy']:.3f} accuracy")
        
        # Risk level (based on volatility or beta)
        vol_col = None
        if 'Volatility' in data_df.columns:
            vol_col = 'Volatility'
        elif 'Beta' in data_df.columns:
            vol_col = 'Beta'
        
        if vol_col:
            risk_labels = pd.cut(
                data_df[vol_col],
                bins=3,
                labels=['Low', 'Medium', 'High']
            ).values
            self.probe_property(risk_labels, 'Risk_Level')
            print(f"  Risk Level: {self.results['Risk_Level']['linear_accuracy']:.3f} accuracy")
        
        # Continuous factor regression
        if true_factors:
            print("\n  Continuous factor R² scores:")
            for factor_name, factor_values in true_factors.items():
                if isinstance(factor_values, np.ndarray):
                    self.probe_regression(factor_values, f'Factor_{factor_name}')
                    print(f"    {factor_name}: R² = {self.results[f'Factor_{factor_name}']['r2']:.3f}")
        
        return self.results
    
    def get_summary_df(self):
        """Get results as DataFrame."""
        rows = []
        for prop_name, result in self.results.items():
            if 'linear_accuracy' in result:
                rows.append({
                    'Property': prop_name,
                    'Linear_Acc': result.get('linear_accuracy', np.nan),
                    'MLP_Acc': result.get('mlp_accuracy', np.nan),
                    'Selectivity': result.get('linear_selectivity', np.nan),
                    'Baseline': result.get('baseline', np.nan)
                })
            elif 'r2' in result:
                rows.append({
                    'Property': prop_name,
                    'R2': result['r2'],
                    'Type': 'regression'
                })
        
        return pd.DataFrame(rows)
    
    def visualize(self, save_path='probing_results.png'):
        """Visualize probing results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Classification results
        ax1 = axes[0]
        class_results = [(k, v) for k, v in self.results.items() if 'linear_accuracy' in v]
        
        if class_results:
            props = [r[0] for r in class_results]
            linear_acc = [r[1]['linear_accuracy'] for r in class_results]
            baselines = [r[1]['baseline'] for r in class_results]
            
            x = np.arange(len(props))
            width = 0.35
            
            ax1.bar(x - width/2, linear_acc, width, label='Linear Probe', color='steelblue')
            ax1.bar(x + width/2, baselines, width, label='Majority Baseline', color='lightgray')
            
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'{self.embedding_name}: Classification Probes')
            ax1.set_xticks(x)
            ax1.set_xticklabels(props, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(alpha=0.3, axis='y')
        
        # Regression results
        ax2 = axes[1]
        reg_results = [(k, v) for k, v in self.results.items() if 'r2' in v]
        
        if reg_results:
            props = [r[0].replace('Factor_', '') for r in reg_results]
            r2_scores = [r[1]['r2'] for r in reg_results]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(props)))
            ax2.barh(props, r2_scores, color=colors)
            ax2.set_xlabel('R² Score')
            ax2.set_title(f'{self.embedding_name}: Regression Probes')
            ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
            ax2.grid(alpha=0.3, axis='x')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()

# ============================================================================
# SECTION 3: Comparative Probing Across DR Methods
# ============================================================================

class ComparativeProbingAnalysis:
    """
    Compare information preservation across different DR methods.
    """
    
    def __init__(self, X_original, feature_names, dr_methods=None):
        """
        Args:
            X_original: Original feature matrix
            feature_names: List of feature names
            dr_methods: Dict of DR method name → embedding matrix
        """
        self.X_original = X_original
        self.feature_names = feature_names
        self.dr_methods = dr_methods or {}
        self.comparison_results = {}
    
    def add_dr_method(self, name, embeddings):
        """Add a DR method's embeddings."""
        self.dr_methods[name] = embeddings
    
    def run_comparison(self, labels_dict, true_factors=None):
        """
        Run probing comparison across all DR methods.
        
        Args:
            labels_dict: Dict mapping property name → labels array
            true_factors: Dict mapping factor name → continuous values
        """
        all_results = {}
        
        for dr_name, embeddings in self.dr_methods.items():
            print(f"\n{'='*60}")
            print(f"Probing: {dr_name}")
            print('='*60)
            
            prober = EmbeddingProber(embeddings, embedding_name=dr_name)
            
            # Classification probes
            for prop_name, labels in labels_dict.items():
                prober.probe_property(labels, prop_name)
            
            # Regression probes
            if true_factors:
                for factor_name, factor_values in true_factors.items():
                    if isinstance(factor_values, np.ndarray):
                        prober.probe_regression(factor_values, factor_name)
            
            all_results[dr_name] = prober.get_summary_df()
            all_results[dr_name]['DR_Method'] = dr_name
        
        self.comparison_results = pd.concat(all_results.values(), ignore_index=True)
        return self.comparison_results
    
    def visualize_comparison(self, save_path='probing_comparison.png'):
        """Create comparison visualization."""
        if self.comparison_results.empty:
            print("No results to visualize. Run run_comparison first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Classification comparison
        ax1 = axes[0]
        class_df = self.comparison_results[self.comparison_results['Linear_Acc'].notna()]
        
        if not class_df.empty:
            pivot = class_df.pivot(index='Property', columns='DR_Method', values='Linear_Acc')
            pivot.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_ylabel('Linear Probe Accuracy')
            ax1.set_title('Classification Probes by DR Method')
            ax1.legend(title='DR Method')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(alpha=0.3, axis='y')
        
        # Regression comparison
        ax2 = axes[1]
        reg_df = self.comparison_results[self.comparison_results['R2'].notna()]
        
        if not reg_df.empty:
            pivot = reg_df.pivot(index='Property', columns='DR_Method', values='R2')
            pivot.plot(kind='bar', ax=ax2, width=0.8)
            ax2.set_ylabel('R² Score')
            ax2.set_title('Regression Probes by DR Method')
            ax2.legend(title='DR Method')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
        plt.close()

# ============================================================================
# SECTION 4: Financial Data Generation
# ============================================================================

def generate_financial_data(n_samples=3000):
    """Generate synthetic financial data with known factor structure.

    CIRCULARITY WARNING: The probing labels used in __main__ (e.g., Market_Regime
    from Price_Momentum_1M, Quality_Tier from ROE) are derived from the same
    features that exist in X. Since PCA embeddings are linear combinations of ALL
    input features, a linear probe on PCA embeddings will trivially predict labels
    derived from any single feature — this measures reconstruction ability, not
    whether the embedding has learned meaningful latent concepts. High probing
    accuracy (> 0.8) on this data is expected by construction and does not
    indicate that the embedding encodes genuine financial semantics. To make
    probing meaningful, labels should come from an external source not derivable
    from the input features (e.g., future returns, analyst ratings, credit events).

    Additionally, the factor loadings here (0.7–0.8 signal, 0.2–0.3 noise) are
    unrealistically high — see xai_dr_comparison.py docstring for discussion.
    """
    np.random.seed(42)
    
    # True latent factors
    growth = np.random.normal(0, 1, n_samples)
    value = np.random.normal(0, 1, n_samples)
    quality = np.random.normal(0, 1, n_samples)
    momentum = np.random.normal(0, 1, n_samples)
    
    features = {}
    
    # Growth features
    features['Revenue_Growth'] = 0.8 * growth + 0.2 * np.random.randn(n_samples)
    features['EPS_Growth'] = 0.75 * growth + 0.25 * np.random.randn(n_samples)
    features['Sales_Growth'] = 0.7 * growth + 0.3 * np.random.randn(n_samples)
    
    # Value features
    features['PE_Ratio'] = 25 + 10 * value + 3 * np.random.randn(n_samples)
    features['PB_Ratio'] = 3 + value + 0.5 * np.random.randn(n_samples)
    features['Dividend_Yield'] = 0.02 + 0.015 * (-value) + 0.005 * np.random.randn(n_samples)
    
    # Quality features
    features['ROE'] = 0.15 + 0.08 * quality + 0.02 * np.random.randn(n_samples)
    features['ROA'] = 0.08 + 0.04 * quality + 0.01 * np.random.randn(n_samples)
    features['Debt_Ratio'] = 0.4 - 0.2 * quality + 0.1 * np.random.randn(n_samples)
    
    # Momentum features
    features['Price_Momentum_1M'] = 0.05 * momentum + 0.03 * np.random.randn(n_samples)
    features['Price_Momentum_6M'] = 0.1 * momentum + 0.05 * np.random.randn(n_samples)
    features['Volatility'] = 0.25 + 0.1 * np.abs(momentum) + 0.05 * np.random.randn(n_samples)
    
    df = pd.DataFrame(features)
    
    true_factors = {
        'Growth': growth,
        'Value': value,
        'Quality': quality,
        'Momentum': momentum
    }
    
    return df, true_factors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROBING CLASSIFIERS FOR EMBEDDING INTERPRETATION")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating synthetic financial data...")
    df, true_factors = generate_financial_data(n_samples=3000)
    feature_names = df.columns.tolist()
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    # Create embeddings using different DR methods
    print("\n2. Creating embeddings with different DR methods...")
    
    # PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)
    print(f"  PCA: {X_pca.shape}")
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f"  t-SNE: {X_tsne.shape}")
    
    # UMAP
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=4, random_state=42)
        X_umap = reducer.fit_transform(X)
        print(f"  UMAP: {X_umap.shape}")
    
    # Create property labels for probing
    print("\n3. Creating property labels...")
    
    labels_dict = {
        'Market_Regime': pd.cut(df['Price_Momentum_1M'], bins=3, labels=['Bear', 'Sideways', 'Bull']).values,
        'Quality_Tier': pd.cut(df['ROE'], bins=3, labels=['Low', 'Medium', 'High']).values,
        'Value_Style': pd.cut(df['PE_Ratio'], bins=3, labels=['Value', 'Fair', 'Growth']).values,
        'Risk_Level': pd.cut(df['Volatility'], bins=3, labels=['Low', 'Medium', 'High']).values
    }
    
    # Run comparative probing
    print("\n4. Running comparative probing analysis...")
    
    comparison = ComparativeProbingAnalysis(X, feature_names)
    comparison.add_dr_method('PCA', X_pca)
    comparison.add_dr_method('t-SNE', X_tsne)
    if UMAP_AVAILABLE:
        comparison.add_dr_method('UMAP', X_umap)
    comparison.add_dr_method('Original', X)  # Baseline
    
    results_df = comparison.run_comparison(labels_dict, true_factors)
    
    # Display results
    print("\n" + "=" * 70)
    print("PROBING RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Visualize
    print("\n5. Creating visualizations...")
    comparison.visualize_comparison(save_path='/home/claude/probing_comparison.png')
    
    # Detailed probing for best method
    print("\n6. Detailed probing analysis for PCA...")
    
    pca_prober = EmbeddingProber(X_pca, embedding_name='PCA')
    pca_prober.run_financial_probes(df, true_factors)
    pca_prober.visualize(save_path='/home/claude/pca_probing_details.png')
    
    # Probe importance analysis
    print("\n7. Which embedding dimensions encode which properties?")
    
    for prop_name, labels in labels_dict.items():
        probe = ProbingClassifier(probe_type='linear')
        probe.fit(X_pca, labels)
        
        importances = probe.get_importance([f'PC{i+1}' for i in range(X_pca.shape[1])])
        top_dim = importances.idxmax()
        
        print(f"  {prop_name}: Most important dimension = {top_dim}")
    
    # Save results
    results_df.to_csv('/home/claude/probing_results.csv', index=False)
    
    print("\n" + "=" * 70)
    print("PROBING ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\nKey findings:")
    
    # Best method per property
    for prop in labels_dict.keys():
        prop_results = results_df[results_df['Property'] == prop]
        if 'Linear_Acc' in prop_results.columns:
            best_idx = prop_results['Linear_Acc'].idxmax()
            best_method = prop_results.loc[best_idx, 'DR_Method']
            best_acc = prop_results.loc[best_idx, 'Linear_Acc']
            print(f"  {prop}: Best method = {best_method} ({best_acc:.3f})")
    
    print("\nOutput files:")
    print("  • probing_comparison.png - Method comparison")
    print("  • pca_probing_details.png - Detailed PCA probing")
    print("  • probing_results.csv - Full results table")
    
    print("\nInterpretation guide:")
    print("  • High linear probe accuracy → Property is linearly encoded")
    print("  • High selectivity → Encoding is meaningful (above baseline)")
    print("  • Compare Original vs. DR → Information preservation")
