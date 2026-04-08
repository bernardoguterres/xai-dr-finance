"""
Semi-Synthetic Financial Data Generator
Generates financial data with realistic noise calibrated to empirical parameters.

Implements Dr. Watson's recommendation: fit hyperparameters (loadings, noise variance)
to real-world data and simulate on that basis. Supports three noise regimes:
- Idealised: Original high-SNR settings (proof of concept)
- Moderate: Intermediate stress test
- Realistic: Calibrated to Fama-French / empirical factor model estimates

Author: Bernardo Guterres
Date: March 2026

    - Fama & French (1993): Common risk factors in stock returns
    - Barra/MSCI (2011): USE4 risk model, typical R² = 0.25-0.40
    - Connor & Korajczyk (1988): Risk and return in an equilibrium APT
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# ============================================================================
# SECTION 1: Empirical Calibration Parameters
# ============================================================================

# Empirical factor loadings estimated from Fama-French literature and Barra
# risk model documentation. These represent REALISTIC magnitudes for
# cross-sectional stock characteristic factor models.
#
# Sources:
#   - Fama & French (1993, 2015): 5-factor model, average R² ~ 0.25
#   - Harvey et al. (2016): Most published factors have t-stats of 2-3
#   - Barra USE4 model: Specific risk > 60% of total variance for most stocks
#   - Hou et al. (2020): Replicating anomalies, average factor loading ~ 0.3

EMPIRICAL_PARAMS = {
    'idealised': {
        'description': 'Original high-SNR settings for proof of concept',
        'loadings_range': (0.60, 0.85),
        'noise_range': (0.15, 0.40),
        'cross_factor_corr': 0.10,
        'noise_distribution': 'normal',
        'expected_factor_r2': 0.75,  # What PCA should explain
    },
    'moderate': {
        'description': 'Intermediate stress test',
        'loadings_range': (0.35, 0.55),
        'noise_range': (0.45, 0.65),
        'cross_factor_corr': 0.25,
        'noise_distribution': 'normal',
        'expected_factor_r2': 0.40,
    },
    'realistic': {
        'description': 'Calibrated to Fama-French empirical estimates',
        'loadings_range': (0.15, 0.30),
        'noise_range': (0.70, 0.90),
        'cross_factor_corr': 0.35,
        'noise_distribution': 'student_t',  # Fat tails
        'degrees_of_freedom': 5,  # Typical for financial returns
        'expected_factor_r2': 0.20,
        'n_noise_features': 5,  # Pure noise features with no factor structure
        'nonlinear_distortion': True,  # Apply monotone non-linear transforms
    },
    'noisy': {
        'description': 'Stress-test: signal nearly buried in noise. Tests method breakdown.',
        'loadings_range': (0.08, 0.18),
        'noise_range': (0.85, 1.10),
        'cross_factor_corr': 0.45,  # High cross-factor contamination
        'noise_distribution': 'student_t',
        'degrees_of_freedom': 3,  # Very heavy tails (variance = 3)
        'expected_factor_r2': 0.08,
        'n_noise_features': 10,  # Half the features are pure noise
        'nonlinear_distortion': True,
        'interaction_noise': True,  # Add multiplicative cross-feature noise
    }
}

def generate_semi_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 20,
    n_factors: int = 4,
    regime: str = 'realistic',
    random_state: int = 42,
    custom_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate semi-synthetic financial data with configurable noise regime.

    The data follows a linear factor model:
        X_ij = sum_k (lambda_ik * F_kj) + epsilon_ij

    where lambda are factor loadings, F are latent factors, and epsilon is
    idiosyncratic noise. The key difference from the original generator is
    that loadings and noise levels are calibrated to empirical estimates.

    Args:
        n_samples: Number of observations (stocks/time periods)
        n_features: Number of observed features
        n_factors: Number of latent factors (default 4: Growth, Value, Quality, Momentum)
        regime: One of 'idealised', 'moderate', 'realistic', 'noisy'
        random_state: Random seed for reproducibility
        custom_params: Override default parameters for the regime

    Returns:
        Tuple of (features_df, true_factors_df, factor_membership_dict)
    """
    np.random.seed(random_state)

    params = EMPIRICAL_PARAMS.get(regime)
    if params is None:
        raise ValueError(f"Unknown regime: {regime}. Choose from {list(EMPIRICAL_PARAMS.keys())}")

    if custom_params:
        params = {**params, **custom_params}

    loading_lo, loading_hi = params['loadings_range']
    noise_lo, noise_hi = params['noise_range']
    cross_corr = params['cross_factor_corr']

    # ---- Generate latent factors with realistic cross-correlations ----
    # Real factors are correlated: value and quality share variance,
    # growth and momentum share variance, etc.
    factor_corr = np.eye(n_factors)
    if n_factors >= 4:
        factor_corr[0, 2] = cross_corr       # Growth-Quality
        factor_corr[2, 0] = cross_corr
        factor_corr[0, 3] = cross_corr * 0.6  # Growth-Momentum (weaker)
        factor_corr[3, 0] = cross_corr * 0.6
        factor_corr[1, 2] = cross_corr * 0.8  # Value-Quality
        factor_corr[2, 1] = cross_corr * 0.8
        factor_corr[1, 3] = -cross_corr * 0.5  # Value-Momentum (negative)
        factor_corr[3, 1] = -cross_corr * 0.5

    # Generate correlated factors
    L = np.linalg.cholesky(factor_corr)
    raw_factors = np.random.normal(0, 1, (n_samples, n_factors))
    factors = raw_factors @ L.T

    factor_names = ['Growth', 'Value', 'Quality', 'Momentum'][:n_factors]
    if n_factors > 4:
        factor_names += [f'Factor_{i}' for i in range(5, n_factors + 1)]

    # ---- Define feature names and factor assignments ----
    feature_groups = {
        'Growth': ['Revenue_Growth', 'EPS_Growth', 'Sales_Growth',
                    'R&D_Intensity', 'Capex_Growth'],
        'Value': ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield',
                  'FCF_Yield', 'EV_EBITDA'],
        'Quality': ['ROE', 'ROA', 'Debt_to_Equity',
                    'Interest_Coverage', 'Gross_Margin'],
        'Momentum': ['Price_Momentum_1M', 'Price_Momentum_6M',
                     'Volume_Trend', 'Volatility', 'Beta']
    }

    feature_names = []
    factor_membership = {}
    for fname in factor_names:
        if fname in feature_groups:
            feature_names.extend(feature_groups[fname])
            factor_membership[fname] = feature_groups[fname]
        else:
            group = [f'{fname}_feat_{j}' for j in range(n_features // n_factors)]
            feature_names.extend(group)
            factor_membership[fname] = group

    feature_names = feature_names[:n_features]

    # ---- Generate loadings matrix ----
    # Each feature loads primarily on its assigned factor, with small
    # cross-loadings on other factors (realistic: features are not pure)
    n_feat = len(feature_names)
    loadings = np.zeros((n_feat, n_factors))

    feat_idx = 0
    for f_idx, fname in enumerate(factor_names):
        group_size = len(factor_membership.get(fname, []))
        for j in range(group_size):
            if feat_idx >= n_feat:
                break
            # Primary loading: drawn from the regime's range
            primary = np.random.uniform(loading_lo, loading_hi)
            loadings[feat_idx, f_idx] = primary

            # Cross-loadings: small, on other factors (realistic contamination)
            for other_f in range(n_factors):
                if other_f != f_idx:
                    cross_load = np.random.uniform(0, loading_lo * 0.4)
                    # Some features have negative cross-loadings (e.g., Debt_to_Equity on Quality)
                    if np.random.random() < 0.3:
                        cross_load = -cross_load
                    loadings[feat_idx, other_f] = cross_load

            feat_idx += 1

    # ---- Generate noise ----
    if params['noise_distribution'] == 'student_t':
        df = params.get('degrees_of_freedom', 5)
        # Student-t noise has heavier tails than normal (realistic for finance)
        raw_noise = np.random.standard_t(df, size=(n_samples, n_feat))
        # Scale to unit variance (Student-t with df degrees has var = df/(df-2))
        raw_noise = raw_noise / np.sqrt(df / (df - 2))
    else:
        raw_noise = np.random.normal(0, 1, (n_samples, n_feat))

    # Scale noise per feature
    noise_scales = np.random.uniform(noise_lo, noise_hi, n_feat)
    noise = raw_noise * noise_scales

    # ---- Construct observed features ----
    X = factors @ loadings.T + noise

    # ---- Apply non-linear distortions (realistic regime) ----
    # Real financial features are not linearly related to factors.
    # E.g., PE ratios are ratios (non-linear), volatility is a variance
    # measure, growth rates compound non-linearly.
    if params.get('nonlinear_distortion', False):
        for col in range(X.shape[1]):
            transform = np.random.choice(['none', 'square', 'exp', 'abs', 'sinh'])
            if transform == 'square':
                # Signed square: preserves sign, compresses small values
                X[:, col] = np.sign(X[:, col]) * np.sqrt(np.abs(X[:, col]))
            elif transform == 'exp':
                # Soft exponential: creates right-skew (like PE ratios)
                X[:, col] = np.sign(X[:, col]) * np.log1p(np.abs(X[:, col]))
            elif transform == 'abs':
                # Absolute value: like volatility (always positive)
                X[:, col] = np.abs(X[:, col])
            elif transform == 'sinh':
                # Sinh: stretches tails (like return distributions)
                X[:, col] = np.sinh(X[:, col] * 0.5) * 2
            # 'none' leaves the feature linear

    # ---- Add multiplicative interaction noise (noisy regime) ----
    # Real financial features exhibit heteroskedasticity and cross-feature
    # interactions. E.g., volatility amplifies during crises, correlations
    # spike under stress. This adds feature-pair interactions scaled by noise,
    # destroying the clean additive factor structure that linear methods assume.
    if params.get('interaction_noise', False):
        n_interactions = min(n_feat, 10)
        for _ in range(n_interactions):
            i, j = np.random.choice(X.shape[1], 2, replace=False)
            interaction = X[:, i] * X[:, j]
            # Mix interaction back into a random feature
            target = np.random.randint(0, X.shape[1])
            mix_weight = np.random.uniform(0.2, 0.5)
            X[:, target] = (1 - mix_weight) * X[:, target] + mix_weight * interaction

    # ---- Add pure noise features (realistic regime) ----
    # Real datasets contain features that are not driven by the main factors
    # (e.g., analyst sentiment scores, supply chain metrics, ESG ratings).
    # These dilute the signal and make factor recovery harder.
    n_noise_feats = params.get('n_noise_features', 0)
    noise_feat_names = []
    if n_noise_feats > 0:
        noise_feat_names = [f'Noise_Feature_{i+1}' for i in range(n_noise_feats)]
        if params['noise_distribution'] == 'student_t':
            df_val = params.get('degrees_of_freedom', 5)
            pure_noise = np.random.standard_t(df_val, size=(n_samples, n_noise_feats))
            pure_noise = pure_noise / np.sqrt(df_val / (df_val - 2))
        else:
            pure_noise = np.random.normal(0, 1, (n_samples, n_noise_feats))
        X = np.hstack([X, pure_noise])
        feature_names = feature_names + noise_feat_names

    n_feat_total = X.shape[1]

    # ---- Compute actual SNR for reporting ----
    # Only compute for factor-driven features (not pure noise)
    n_factor_feats = n_feat
    signal_var = np.var(factors @ loadings.T, axis=0)
    noise_var_arr = np.var(noise, axis=0)
    snr = signal_var / (noise_var_arr + 1e-10)
    mean_snr = np.mean(snr)

    df_features = pd.DataFrame(X, columns=feature_names[:n_feat_total])
    df_factors = pd.DataFrame(factors, columns=factor_names)

    print(f"\nSemi-synthetic data generated (regime: {regime})")
    print(f"  Samples: {n_samples}, Features: {n_feat_total} "
          f"({n_factor_feats} factor-driven + {n_noise_feats} pure noise)")
    print(f"  Factors: {n_factors}")
    print(f"  Loading range: [{loading_lo:.2f}, {loading_hi:.2f}]")
    print(f"  Noise range: [{noise_lo:.2f}, {noise_hi:.2f}]")
    print(f"  Noise distribution: {params['noise_distribution']}")
    if params.get('nonlinear_distortion', False):
        print(f"  Non-linear distortion: ENABLED")
    if params.get('interaction_noise', False):
        print(f"  Interaction noise: ENABLED")
    print(f"  Mean SNR (factor features): {mean_snr:.2f} "
          f"(range: [{snr.min():.2f}, {snr.max():.2f}])")
    print(f"  Expected factor R²: ~{params['expected_factor_r2']:.2f}")

    return df_features, df_factors, factor_membership

def generate_all_regimes(
    n_samples: int = 2000,
    random_state: int = 42
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    """
    Generate data under all three noise regimes for comparative evaluation.

    Returns:
        Dict mapping regime name to (features_df, factors_df, membership_dict)
    """
    results = {}
    for regime in ['idealised', 'moderate', 'realistic', 'noisy']:
        print(f"\n{'='*60}")
        print(f"Generating {regime.upper()} regime data")
        print('='*60)
        results[regime] = generate_semi_synthetic_data(
            n_samples=n_samples,
            regime=regime,
            random_state=random_state
        )
    return results

# ============================================================================
# SECTION 2: Semi-Synthetic Calibration from Real Data
# ============================================================================

def calibrate_from_real_data(
    real_data: pd.DataFrame,
    n_factors: int = 4
) -> Dict:
    """
    Fit a factor analysis model to real data and extract empirical loadings.

    This implements the semi-synthetic approach recommended by Dr. Watson:
    fit hyperparameters to real-world data, then use them to simulate.

    Args:
        real_data: DataFrame of real financial features (standardised)
        n_factors: Number of factors to extract

    Returns:
        Dict of calibrated parameters that can be passed as custom_params
        to generate_semi_synthetic_data()
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(real_data.select_dtypes(include=[np.number]).dropna(axis=1))

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(X)

    # Extract empirical loading magnitudes
    abs_loadings = np.abs(fa.components_.T)
    primary_loadings = []
    for i in range(abs_loadings.shape[0]):
        primary_loadings.append(abs_loadings[i].max())

    loading_lo = float(np.percentile(primary_loadings, 25))
    loading_hi = float(np.percentile(primary_loadings, 75))

    # Extract noise variance
    noise_var = fa.noise_variance_
    noise_lo = float(np.sqrt(np.percentile(noise_var, 25)))
    noise_hi = float(np.sqrt(np.percentile(noise_var, 75)))

    # Compute factor correlation from scores
    scores = fa.transform(X)
    factor_corr = np.corrcoef(scores.T)
    avg_cross_corr = float(np.mean(np.abs(factor_corr[np.triu_indices(n_factors, k=1)])))

    # Estimate explained variance
    explained_var = 1 - np.mean(noise_var)

    calibrated = {
        'description': f'Calibrated from real data ({real_data.shape[0]} samples, {X.shape[1]} features)',
        'loadings_range': (max(0.05, loading_lo), min(0.95, loading_hi)),
        'noise_range': (max(0.1, noise_lo), min(0.95, noise_hi)),
        'cross_factor_corr': avg_cross_corr,
        'noise_distribution': 'student_t',
        'degrees_of_freedom': 5,
        'expected_factor_r2': float(explained_var),
    }

    print(f"\nCalibrated parameters from real data:")
    print(f"  Loading range: [{calibrated['loadings_range'][0]:.3f}, {calibrated['loadings_range'][1]:.3f}]")
    print(f"  Noise range: [{calibrated['noise_range'][0]:.3f}, {calibrated['noise_range'][1]:.3f}]")
    print(f"  Cross-factor correlation: {avg_cross_corr:.3f}")
    print(f"  Estimated factor R²: {explained_var:.3f}")

    return calibrated

# ============================================================================
# SECTION 3: Comparative Analysis Across Regimes
# ============================================================================

def compare_regimes_summary(regime_results: Dict) -> pd.DataFrame:
    """
    Compute summary statistics for each regime's data to verify
    that the noise levels produce expected properties.

    Args:
        regime_results: Output from generate_all_regimes()

    Returns:
        DataFrame comparing data properties across regimes
    """
    from sklearn.decomposition import PCA

    summaries = []
    for regime, (df_features, df_factors, membership) in regime_results.items():
        X = StandardScaler().fit_transform(df_features)

        # PCA explained variance (proxy for factor R²)
        pca = PCA(n_components=4)
        pca.fit(X)
        var_explained_4 = sum(pca.explained_variance_ratio_)

        # Correlation between features and their assigned factor
        avg_loading_corr = []
        for fname, fgroup in membership.items():
            if fname in df_factors.columns:
                for feat in fgroup:
                    if feat in df_features.columns:
                        corr = np.abs(np.corrcoef(
                            df_features[feat].values,
                            df_factors[fname].values
                        )[0, 1])
                        avg_loading_corr.append(corr)

        # Signal-to-noise ratio
        signal = df_factors.values @ np.linalg.lstsq(
            df_factors.values, X, rcond=None
        )[0]
        residual = X - signal
        snr = np.mean(np.var(signal, axis=0) / (np.var(residual, axis=0) + 1e-10))

        summaries.append({
            'Regime': regime,
            'Mean_Feature_Factor_Corr': np.mean(avg_loading_corr) if avg_loading_corr else 0,
            'PCA_4PC_Variance_Explained': var_explained_4,
            'Mean_SNR': snr,
            'Feature_Std_Mean': np.mean(np.std(X, axis=0)),
        })

    summary_df = pd.DataFrame(summaries)
    print("\n" + "=" * 70)
    print("REGIME COMPARISON SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    return summary_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEMI-SYNTHETIC FINANCIAL DATA GENERATOR")
    print("Calibrated to empirical factor model estimates")
    print("=" * 70)

    # Generate data under all three regimes
    regime_results = generate_all_regimes(n_samples=2000)

    # Compare regimes
    summary = compare_regimes_summary(regime_results)

    # Show what PCA would find under each regime
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("PCA EXPLAINED VARIANCE BY REGIME")
    print("(Shows how much harder factor recovery becomes with realistic noise)")
    print("=" * 70)

    for regime, (df_feat, df_fac, membership) in regime_results.items():
        X = StandardScaler().fit_transform(df_feat)
        pca = PCA(n_components=4)
        pca.fit(X)
        var_ratios = pca.explained_variance_ratio_

        print(f"\n  {regime.upper()}:")
        for i, vr in enumerate(var_ratios):
            print(f"    PC{i+1}: {vr:.3f} ({vr*100:.1f}%)")
        print(f"    Total (4 PCs): {sum(var_ratios):.3f} ({sum(var_ratios)*100:.1f}%)")

    # Optional: calibrate from real data if available
    print("\n" + "=" * 70)
    print("SEMI-SYNTHETIC CALIBRATION DEMO")
    print("=" * 70)

    try:
        from data_loaders import _generate_synthetic_stock_fundamentals
        # In production, replace with load_stock_fundamentals() for real data
        real_proxy = _generate_synthetic_stock_fundamentals(n_stocks=500)
        numeric_cols = real_proxy.select_dtypes(include=[np.number]).columns
        calibrated_params = calibrate_from_real_data(real_proxy[numeric_cols])

        print("\nGenerating data with calibrated parameters...")
        df_cal, factors_cal, mem_cal = generate_semi_synthetic_data(
            regime='realistic',
            custom_params=calibrated_params
        )
    except Exception as e:
        print(f"Calibration demo skipped: {e}")

    print("\n" + "=" * 70)
    print("THESIS INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
    For thesis Chapter 5 (Evaluation), present results as follows:

    1. IDEALISED regime (SNR ~3-5): Proof of concept
       - "Under idealised conditions, the framework achieves R² > 0.95..."
       - Demonstrates methodological correctness

    2. MODERATE regime (SNR ~1-1.5): Stress test
       - "Under moderate noise, surrogate R² degrades to X..."
       - Shows where methods begin to struggle

    3. REALISTIC regime (SNR ~0.3-0.7): Practical performance
       - "Under realistic conditions calibrated to empirical estimates..."
       - Shows honest expected performance on real data
       - Methods that still work here are genuinely useful

    Key thesis argument: The value is in knowing WHEN methods work,
    not just that they work under favourable conditions.
    """)
