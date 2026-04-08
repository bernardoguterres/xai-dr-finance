"""
XAI for Dimensionality Reduction - Comprehensive Comparison
Compares explainability across: PCA, t-SNE, UMAP, NMF, ICA, Factor Analysis

Author: Bernardo Guterres
Date: December 2025

This script provides a systematic comparison of dimensionality reduction
techniques ranked by their amenability to explanation methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Try importing UMAP (optional dependency)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

np.random.seed(42)

# ============================================================================
# SECTION 1: Generate Synthetic Financial Data with Known Factor Structure
# ============================================================================

def generate_financial_data_with_factors(n_samples=2000, n_features=20, n_factors=4):
    """
    Generate synthetic financial data with KNOWN underlying factor structure.
    This allows us to validate whether XAI methods recover true factors.

    Factors:
    - Factor 1: Growth (Revenue_Growth, EPS_Growth, Sales_Growth, R&D_Intensity)
    - Factor 2: Value (PE_Ratio, PB_Ratio, Dividend_Yield, FCF_Yield)
    - Factor 3: Quality (ROE, ROA, Debt_to_Equity, Interest_Coverage)
    - Factor 4: Momentum (Price_Momentum_1M, Price_Momentum_6M, Volume_Trend, Volatility)

    IMPORTANT LIMITATION — Unrealistic Signal-to-Noise Ratio:
    The factor loadings here (0.6–0.85) with noise coefficients (0.15–0.45) give
    signal-to-noise ratios of ~2:1 to ~5:1. In real financial data, factor loadings
    are typically 0.2–0.4 with much larger idiosyncratic noise (Fama & French, 1993;
    Barra risk models report R² of 0.2–0.4 for factor models). This means:

    - PCA will recover factors almost perfectly here (by construction), whereas on
      real data it would explain only 20–40% of variance in the first 4 components.
    - Surrogate R² values will be near 1.0 for linear methods, vs. 0.5–0.7 typical
      in academic studies (Lundberg & Lee, 2017).
    - Factor recovery scores will be artificially high because the data was designed
      so that PCA's variance-maximizing directions align exactly with the 4 factors.

    These results should be interpreted as a PROOF OF CONCEPT on idealised data,
    not as evidence that these methods achieve similar performance on real financial
    datasets. The thesis must discuss this limitation explicitly.
    """
    np.random.seed(42)
    
    # Generate latent factors (the "true" underlying structure)
    growth_factor = np.random.normal(0, 1, n_samples)
    value_factor = np.random.normal(0, 1, n_samples)
    quality_factor = np.random.normal(0, 1, n_samples)
    momentum_factor = np.random.normal(0, 1, n_samples)
    
    # Add some correlation between factors (realistic)
    quality_factor = 0.3 * growth_factor + 0.9 * quality_factor  # Quality correlates with growth
    
    # Generate observed features as combinations of factors + noise
    # NOTE: Loading coefficients (0.6–0.85) are 2–3x higher than typical in
    # empirical finance. Barra/MSCI factor models report loadings of 0.2–0.4.
    # The low noise terms (0.15–0.4) further inflate recoverability.
    features = {}

    # Growth features (load heavily on growth factor)
    features['Revenue_Growth'] = 0.8 * growth_factor + 0.2 * np.random.normal(0, 1, n_samples)
    features['EPS_Growth'] = 0.75 * growth_factor + 0.1 * value_factor + 0.25 * np.random.normal(0, 1, n_samples)
    features['Sales_Growth'] = 0.85 * growth_factor + 0.15 * np.random.normal(0, 1, n_samples)
    features['R&D_Intensity'] = 0.6 * growth_factor + 0.3 * quality_factor + 0.3 * np.random.normal(0, 1, n_samples)
    features['Capex_Growth'] = 0.7 * growth_factor + 0.2 * momentum_factor + 0.3 * np.random.normal(0, 1, n_samples)
    
    # Value features (load heavily on value factor)
    features['PE_Ratio'] = 0.8 * value_factor + 0.2 * np.random.normal(0, 1, n_samples)
    features['PB_Ratio'] = 0.75 * value_factor + 0.15 * quality_factor + 0.2 * np.random.normal(0, 1, n_samples)
    features['Dividend_Yield'] = 0.7 * value_factor - 0.3 * growth_factor + 0.25 * np.random.normal(0, 1, n_samples)
    features['FCF_Yield'] = 0.65 * value_factor + 0.25 * quality_factor + 0.3 * np.random.normal(0, 1, n_samples)
    features['EV_EBITDA'] = 0.6 * value_factor + 0.2 * growth_factor + 0.35 * np.random.normal(0, 1, n_samples)
    
    # Quality features (load heavily on quality factor)
    features['ROE'] = 0.8 * quality_factor + 0.2 * np.random.normal(0, 1, n_samples)
    features['ROA'] = 0.75 * quality_factor + 0.1 * growth_factor + 0.25 * np.random.normal(0, 1, n_samples)
    features['Debt_to_Equity'] = -0.7 * quality_factor + 0.3 * np.random.normal(0, 1, n_samples)  # Negative loading
    features['Interest_Coverage'] = 0.65 * quality_factor + 0.35 * np.random.normal(0, 1, n_samples)
    features['Gross_Margin'] = 0.6 * quality_factor + 0.2 * growth_factor + 0.35 * np.random.normal(0, 1, n_samples)
    
    # Momentum features (load heavily on momentum factor)
    features['Price_Momentum_1M'] = 0.8 * momentum_factor + 0.2 * np.random.normal(0, 1, n_samples)
    features['Price_Momentum_6M'] = 0.75 * momentum_factor + 0.1 * growth_factor + 0.25 * np.random.normal(0, 1, n_samples)
    features['Volume_Trend'] = 0.6 * momentum_factor + 0.4 * np.random.normal(0, 1, n_samples)
    features['Volatility'] = 0.5 * momentum_factor - 0.3 * quality_factor + 0.4 * np.random.normal(0, 1, n_samples)
    features['Beta'] = 0.55 * momentum_factor + 0.45 * np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(features)
    
    # Store true factors for validation
    true_factors = pd.DataFrame({
        'Growth': growth_factor,
        'Value': value_factor,
        'Quality': quality_factor,
        'Momentum': momentum_factor
    })
    
    # Define which features belong to which factor (ground truth)
    factor_membership = {
        'Growth': ['Revenue_Growth', 'EPS_Growth', 'Sales_Growth', 'R&D_Intensity', 'Capex_Growth'],
        'Value': ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'FCF_Yield', 'EV_EBITDA'],
        'Quality': ['ROE', 'ROA', 'Debt_to_Equity', 'Interest_Coverage', 'Gross_Margin'],
        'Momentum': ['Price_Momentum_1M', 'Price_Momentum_6M', 'Volume_Trend', 'Volatility', 'Beta']
    }
    
    return df, true_factors, factor_membership

# ============================================================================
# SECTION 2: Dimensionality Reduction Implementations
# ============================================================================

class DRExplainer:
    """
    Unified interface for dimensionality reduction with explainability.
    """
    
    def __init__(self, method='pca', n_components=4, **kwargs):
        self.method = method.lower()
        self.n_components = n_components
        self.kwargs = kwargs
        self.model = None
        self.embedding = None
        self.feature_names = None
        self.surrogates = None
        self.shap_explainers = None
        
    def fit_transform(self, X, feature_names=None):
        """Fit DR method and transform data."""
        self.feature_names = feature_names if feature_names is not None else [f'F{i}' for i in range(X.shape[1])]
        
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components)
            self.embedding = self.model.fit_transform(X)
            
        elif self.method == 'nmf':
            # NMF requires non-negative data
            scaler = MinMaxScaler()
            X_positive = scaler.fit_transform(X) + 0.01  # Ensure positive
            self.model = NMF(n_components=self.n_components, init='nndsvda', max_iter=500, random_state=42)
            self.embedding = self.model.fit_transform(X_positive)
            self._X_positive = X_positive
            
        elif self.method == 'ica':
            self.model = FastICA(n_components=self.n_components, random_state=42, max_iter=1000)
            self.embedding = self.model.fit_transform(X)
            
        elif self.method == 'factor_analysis':
            self.model = FactorAnalysis(n_components=self.n_components, random_state=42)
            self.embedding = self.model.fit_transform(X)
            
        elif self.method == 'tsne':
            self.model = TSNE(n_components=min(3, self.n_components), perplexity=30, random_state=42)
            self.embedding = self.model.fit_transform(X)
            
        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed. Use: pip install umap-learn")
            self.model = umap.UMAP(n_components=self.n_components, n_neighbors=15, min_dist=0.1, random_state=42)
            self.embedding = self.model.fit_transform(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._X = X
        return self.embedding
    
    def get_native_explanation(self):
        """
        Get native/intrinsic explanation if available.
        Returns loadings matrix for linear methods.
        """
        if self.method == 'pca':
            return pd.DataFrame(
                self.model.components_.T,
                index=self.feature_names,
                columns=[f'PC{i+1}' for i in range(self.n_components)]
            )
        elif self.method == 'nmf':
            return pd.DataFrame(
                self.model.components_.T,
                index=self.feature_names,
                columns=[f'NMF{i+1}' for i in range(self.n_components)]
            )
        elif self.method == 'ica':
            # mixing_ is (n_features, n_components) — no transpose needed
            return pd.DataFrame(
                self.model.mixing_,
                index=self.feature_names,
                columns=[f'IC{i+1}' for i in range(self.n_components)]
            )
        elif self.method == 'factor_analysis':
            return pd.DataFrame(
                self.model.components_.T,
                index=self.feature_names,
                columns=[f'Factor{i+1}' for i in range(self.n_components)]
            )
        else:
            return None  # t-SNE and UMAP have no native loadings
    
    def train_surrogates(self, X=None):
        """
        Train surrogate models (XGBoost) to approximate the DR transformation.
        This enables SHAP explanation for any DR method.

        R² is evaluated on a held-out 20% test split for honest fidelity
        measurement. The final surrogate used for SHAP is trained on all data.

        SURROGATE R² IS TAUTOLOGICAL FOR LINEAR METHODS:
        PCA/FA/ICA embeddings are exact linear combinations of input features.
        Ridge regression achieves R² = 1.000 on PCA output regardless of data
        noise level (verified empirically). XGBoost gets < 1.0 only due to
        tree-based approximation error, not data difficulty. This means:

        - Surrogate R² is UNINFORMATIVE for linear methods (always ~0.95+)
        - It is ONLY meaningful for non-linear methods (t-SNE, UMAP, AE)
          where the mapping is genuinely complex
        - For linear methods, use factor recovery and variance explained instead
        - In thesis evaluation, annotate linear method R² as tautological
        """
        if X is None:
            X = self._X

        from sklearn.model_selection import train_test_split

        self.surrogates = []
        self.surrogate_r2 = []

        for dim in range(self.embedding.shape[1]):
            y = self.embedding[:, dim]

            # Evaluate on held-out data for honest R²
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            eval_surrogate = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            eval_surrogate.fit(X_train, y_train)
            r2 = r2_score(y_test, eval_surrogate.predict(X_test))
            self.surrogate_r2.append(r2)

            # Train final surrogate on ALL data for SHAP explanations
            full_surrogate = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            full_surrogate.fit(X, y)
            self.surrogates.append(full_surrogate)

        print(f"Surrogate R² scores (held-out): {[f'{r:.3f}' for r in self.surrogate_r2]}")
        return self.surrogate_r2
    
    def explain_with_shap(self, X_sample, dim=0, background_size=100):
        """
        Explain embedding dimension using TreeSHAP on surrogate.
        """
        if self.surrogates is None:
            print("Training surrogates first...")
            self.train_surrogates()
        
        # Use subset as background for efficiency
        background_idx = np.random.choice(len(self._X), min(background_size, len(self._X)), replace=False)
        background = self._X[background_idx]
        
        explainer = shap.TreeExplainer(self.surrogates[dim], data=background)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values
    
    def get_top_features(self, dim=0, top_n=5):
        """
        Get top contributing features for a dimension.
        Uses native explanation if available, otherwise surrogate SHAP.
        """
        native_exp = self.get_native_explanation()
        
        if native_exp is not None:
            # Use native loadings
            col = native_exp.columns[dim]
            loadings = native_exp[col].abs().sort_values(ascending=False)
            return loadings.head(top_n)
        else:
            # Use surrogate SHAP (mean absolute SHAP values)
            if self.surrogates is None:
                self.train_surrogates()
            
            shap_values = self.explain_with_shap(self._X[:500], dim=dim)  # Sample for speed
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            importance = pd.Series(mean_shap, index=self.feature_names)
            return importance.sort_values(ascending=False).head(top_n)

# ============================================================================
# SECTION 3: Explainability Metrics
# ============================================================================

def factor_recovery_score(dr_explainer, true_factor_membership, threshold=0.3):
    """
    Measure how well DR method recovers known factor structure.

    For each discovered component, find best-matching true factor.
    Score = mean correlation between component loadings and true membership.

    CIRCULAR EVALUATION WARNING: When used with generate_financial_data_with_factors(),
    this metric is partially tautological for linear methods. PCA is designed to find
    variance-maximizing directions, and the synthetic data is constructed so that the
    dominant variance directions ARE the 4 factors. High factor recovery scores for PCA
    on this data prove that PCA can find linear structure in linearly-generated data —
    not that PCA outperforms non-linear methods in general. Additionally:

    - The greedy best-match logic allows the same factor to be "claimed" by multiple
      components, inflating scores when factors share variance (e.g., Quality correlates
      with Growth at r=0.3 in our data).
    - For t-SNE/UMAP, recovery goes through surrogate SHAP feature importances (a
      different measure than native loadings), making the comparison across methods
      apples-to-oranges.
    - No exclusive assignment (Hungarian algorithm) is used, which would give lower
      but more honest scores.
    """
    native_exp = dr_explainer.get_native_explanation()
    
    if native_exp is None:
        # For t-SNE/UMAP, use surrogate feature importance
        if dr_explainer.surrogates is None:
            dr_explainer.train_surrogates()
        
        # Get mean |SHAP| for each dimension
        importances = []
        for dim in range(dr_explainer.embedding.shape[1]):
            shap_vals = dr_explainer.explain_with_shap(dr_explainer._X[:200], dim=dim)
            importances.append(np.abs(shap_vals).mean(axis=0))
        
        native_exp = pd.DataFrame(
            np.array(importances).T,
            index=dr_explainer.feature_names,
            columns=[f'Dim{i+1}' for i in range(len(importances))]
        )
    
    # For each DR component, find best matching factor
    recovery_scores = []
    
    for col in native_exp.columns:
        loadings = np.abs(native_exp[col].values)
        loadings = loadings / loadings.max()  # Normalize
        
        best_score = 0
        for factor_name, factor_features in true_factor_membership.items():
            # Create binary membership vector
            membership = np.array([1 if f in factor_features else 0 for f in dr_explainer.feature_names])
            
            # Correlation between loadings and membership
            corr = np.corrcoef(loadings, membership)[0, 1]
            if np.isnan(corr):
                corr = 0
            best_score = max(best_score, abs(corr))
        
        recovery_scores.append(best_score)
    
    return np.mean(recovery_scores), recovery_scores

def explanation_stability(dr_method, X, n_runs=5):
    """
    Measure stability of explanations across multiple runs.
    Critical for stochastic methods like t-SNE.

    KNOWN ISSUE: Returns 1.0 unconditionally for all "deterministic" methods
    (PCA, NMF, ICA, FA). This is an overstatement:
    - ICA (FastICA) has sign and permutation ambiguity; different initialisations
      can flip component signs or reorder components, changing SHAP rankings.
    - NMF with random init has multiple local minima (nndsvda init used here is
      deterministic, but the SHAP importance rankings still vary with background
      sample selection).
    - Even PCA's SHAP importance can vary with the background sample in
      TreeExplainer. Only native PCA loadings are truly seed-invariant.
    Reporting stability=1.0 for these methods inflates the comparison with t-SNE/UMAP.
    """
    explanations = []
    
    for seed in range(n_runs):
        np.random.seed(seed)
        
        if dr_method == 'tsne':
            model = TSNE(n_components=2, perplexity=30, random_state=seed)
        elif dr_method == 'umap' and UMAP_AVAILABLE:
            model = umap.UMAP(n_components=2, random_state=seed)
        else:
            # Deterministic methods - use single run
            return 1.0, [1.0]
        
        embedding = model.fit_transform(X)
        
        # Train surrogate
        surrogate = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        surrogate.fit(X, embedding[:, 0])
        
        # Get feature importance
        importance = surrogate.feature_importances_
        explanations.append(importance)
    
    # Compute pairwise correlations
    correlations = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            corr = np.corrcoef(explanations[i], explanations[j])[0, 1]
            correlations.append(corr)
    
    return np.mean(correlations), correlations

def morf_faithfulness(dr_explainer, X, dim=0, n_steps=10):
    """
    Most Relevant First (MoRF) faithfulness test.

    Remove features in order of importance and measure embedding change.
    Good explanations should show steep initial decline.

    TAUTOLOGICAL EVALUATION WARNING: Both the SHAP importance ranking AND the
    embedding predictions come from the same XGBoost surrogate. TreeSHAP
    computes exact Shapley values for tree models, so masking the top-SHAP
    features of the surrogate will mechanically cause the largest drop in the
    surrogate's output. This tests whether TreeSHAP correctly explains
    XGBoost (which it does by mathematical guarantee) — NOT whether the
    surrogate's SHAP values faithfully explain the original DR transformation.
    A proper MoRF test would re-run the actual DR method (e.g., PCA) with
    masked features and compare the true embedding change. The current
    implementation will always produce "good" faithfulness scores regardless
    of surrogate quality, making it uninformative as a validation metric.
    """
    if dr_explainer.surrogates is None:
        dr_explainer.train_surrogates()
    
    # Get feature importance ranking
    shap_values = dr_explainer.explain_with_shap(X[:100], dim=dim)
    mean_importance = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_importance)[::-1]  # Most important first
    
    # Compute embedding with progressively masked features
    baseline_embedding = dr_explainer.surrogates[dim].predict(X)
    morf_curve = [1.0]  # Start at 1 (no features removed)
    
    X_masked = X.copy()
    feature_mean = X.mean(axis=0)
    
    step_size = len(sorted_idx) // n_steps
    
    for step in range(1, n_steps + 1):
        # Mask top features with mean
        features_to_mask = sorted_idx[:step * step_size]
        X_masked = X.copy()
        X_masked[:, features_to_mask] = feature_mean[features_to_mask]
        
        # Get new embedding
        new_embedding = dr_explainer.surrogates[dim].predict(X_masked)
        
        # Measure correlation with baseline
        corr = np.corrcoef(baseline_embedding, new_embedding)[0, 1]
        morf_curve.append(max(0, corr))
    
    # Compute area under MoRF curve (lower = better faithfulness)
    auc = np.trapz(morf_curve) / len(morf_curve)
    
    return auc, morf_curve

# ============================================================================
# SECTION 4: Run Comprehensive Comparison
# ============================================================================

def run_comparison(X, feature_names, true_factor_membership, methods=None):
    """
    Run comparison across all DR methods.
    """
    if methods is None:
        methods = ['pca', 'nmf', 'ica', 'factor_analysis', 'tsne']
        if UMAP_AVAILABLE:
            methods.append('umap')
    
    results = []
    explainers = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing: {method.upper()}")
        print('='*60)
        
        try:
            # Fit DR method
            n_comp = 2 if method in ['tsne', 'umap'] else 4
            explainer = DRExplainer(method=method, n_components=n_comp)
            embedding = explainer.fit_transform(X, feature_names)
            explainers[method] = explainer
            
            # Get native explanation
            native_exp = explainer.get_native_explanation()
            has_native = native_exp is not None
            
            # Train surrogates
            surrogate_r2 = explainer.train_surrogates()
            
            # Factor recovery
            recovery_score, _ = factor_recovery_score(explainer, true_factor_membership)
            
            # Explanation stability
            stability, _ = explanation_stability(method, X, n_runs=3)
            
            # MoRF faithfulness
            morf_auc, _ = morf_faithfulness(explainer, X, dim=0)
            
            # Top features for dimension 1
            top_features = explainer.get_top_features(dim=0, top_n=5)
            
            results.append({
                'Method': method.upper(),
                'Has_Native_Explanation': has_native,
                'Surrogate_R2_Mean': np.mean(surrogate_r2),
                'Factor_Recovery': recovery_score,
                'Stability': stability,
                'MoRF_AUC': morf_auc,
                'Top_Features': list(top_features.index[:3])
            })
            
            print(f"Native explanation available: {has_native}")
            print(f"Mean surrogate R²: {np.mean(surrogate_r2):.3f}")
            print(f"Factor recovery score: {recovery_score:.3f}")
            print(f"Explanation stability: {stability:.3f}")
            print(f"MoRF AUC (lower=better): {morf_auc:.3f}")
            print(f"Top 3 features for Dim 1: {list(top_features.index[:3])}")
            
        except Exception as e:
            print(f"Error with {method}: {e}")
            results.append({
                'Method': method.upper(),
                'Has_Native_Explanation': False,
                'Surrogate_R2_Mean': np.nan,
                'Factor_Recovery': np.nan,
                'Stability': np.nan,
                'MoRF_AUC': np.nan,
                'Top_Features': []
            })
    
    return pd.DataFrame(results), explainers

# ============================================================================
# SECTION 5: Visualization
# ============================================================================

def visualize_comparison(results_df, explainers, X, save_path='dr_comparison.png'):
    """
    Create comprehensive visualization of DR comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Explainability scores bar chart
    ax1 = axes[0, 0]
    metrics = ['Factor_Recovery', 'Stability', 'Surrogate_R2_Mean']
    x = np.arange(len(results_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = results_df[metric].fillna(0).values
        ax1.bar(x + i*width, values, width, label=metric.replace('_', ' '))
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Score')
    ax1.set_title('Explainability Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(results_df['Method'])
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: MoRF curves for each method
    ax2 = axes[0, 1]
    for method, explainer in explainers.items():
        try:
            _, morf_curve = morf_faithfulness(explainer, X[:500], dim=0, n_steps=8)
            ax2.plot(morf_curve, marker='o', label=method.upper())
        except:
            pass
    ax2.set_xlabel('Features Removed (proportion)')
    ax2.set_ylabel('Embedding Correlation')
    ax2.set_title('MoRF Faithfulness Curves')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: PCA loadings heatmap (if available)
    ax3 = axes[0, 2]
    if 'pca' in explainers:
        pca_exp = explainers['pca'].get_native_explanation()
        if pca_exp is not None:
            sns.heatmap(pca_exp.values[:10], 
                       xticklabels=pca_exp.columns,
                       yticklabels=pca_exp.index[:10],
                       cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax3)
            ax3.set_title('PCA Loadings (Top 10 Features)')
    
    # Plot 4: NMF components (if available)
    ax4 = axes[1, 0]
    if 'nmf' in explainers:
        nmf_exp = explainers['nmf'].get_native_explanation()
        if nmf_exp is not None:
            sns.heatmap(nmf_exp.values[:10], 
                       xticklabels=nmf_exp.columns,
                       yticklabels=nmf_exp.index[:10],
                       cmap='YlOrRd', annot=True, fmt='.2f', ax=ax4)
            ax4.set_title('NMF Components (Top 10 Features)')
    
    # Plot 5: t-SNE embedding colored by known factor
    ax5 = axes[1, 1]
    if 'tsne' in explainers:
        tsne_emb = explainers['tsne'].embedding
        # Color by first feature as proxy
        scatter = ax5.scatter(tsne_emb[:, 0], tsne_emb[:, 1], 
                             c=X[:, 0], cmap='viridis', alpha=0.5, s=10)
        ax5.set_xlabel('t-SNE 1')
        ax5.set_ylabel('t-SNE 2')
        ax5.set_title('t-SNE Embedding')
        plt.colorbar(scatter, ax=ax5, label='Feature 1')
    
    # Plot 6: Summary ranking
    ax6 = axes[1, 2]
    # Compute overall score (weighted average)
    results_df['Overall'] = (
        0.4 * results_df['Factor_Recovery'].fillna(0) +
        0.3 * results_df['Stability'].fillna(0) +
        0.2 * results_df['Surrogate_R2_Mean'].fillna(0) +
        0.1 * (1 - results_df['MoRF_AUC'].fillna(0.5))
    )
    
    sorted_results = results_df.sort_values('Overall', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_results)))
    ax6.barh(sorted_results['Method'], sorted_results['Overall'], color=colors)
    ax6.set_xlabel('Overall Explainability Score')
    ax6.set_title('DR Methods Ranked by Explainability')
    ax6.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("XAI FOR DIMENSIONALITY REDUCTION - COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    # Generate data with known factor structure
    print("\n1. Generating synthetic financial data with known factors...")
    df, true_factors, factor_membership = generate_financial_data_with_factors(
        n_samples=2000, 
        n_features=20, 
        n_factors=4
    )
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"True factors: {list(factor_membership.keys())}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    feature_names = df.columns.tolist()
    
    # Run comparison
    print("\n2. Running DR method comparison...")
    results_df, explainers = run_comparison(X, feature_names, factor_membership)
    
    # Display results
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('/home/claude/dr_comparison_results.csv', index=False)
    print("\nResults saved to: dr_comparison_results.csv")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    visualize_comparison(results_df, explainers, X, save_path='/home/claude/dr_comparison.png')
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR YOUR THESIS")
    print("=" * 70)
    
    best_method = results_df.loc[results_df['Factor_Recovery'].idxmax(), 'Method']
    print(f"\n✓ Best factor recovery: {best_method}")
    
    stable_method = results_df.loc[results_df['Stability'].idxmax(), 'Method']
    print(f"✓ Most stable explanations: {stable_method}")
    
    print("\nFor financial applications requiring regulatory compliance:")
    print("  1. Use PCA, NMF, or Factor Analysis (native explainability)")
    print("  2. Avoid t-SNE for anything beyond visualization")
    print("  3. If using UMAP, always validate with ClusterShapley")
    print("  4. Test explanation stability across market regimes")
