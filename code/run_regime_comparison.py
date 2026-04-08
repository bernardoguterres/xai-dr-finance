"""
Regime Comparison: Run XAI pipeline across idealised, moderate, realistic data.

Compares surrogate R², factor recovery, stability, MoRF faithfulness across
three noise regimes to demonstrate graceful degradation.

Author: Bernardo Guterres
Date: March 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

from semi_synthetic_generator import generate_semi_synthetic_data, EMPIRICAL_PARAMS
from xai_dr_comparison import (
    DRExplainer,
    factor_recovery_score,
    explanation_stability,
    morf_faithfulness
)

def run_single_regime(regime: str, methods=None):
    """Run full DR comparison on one regime."""
    if methods is None:
        methods = ['pca', 'nmf', 'ica', 'factor_analysis', 'tsne']

    df_feat, df_fac, membership = generate_semi_synthetic_data(
        n_samples=2000, regime=regime, random_state=42
    )

    X = StandardScaler().fit_transform(df_feat)
    feature_names = df_feat.columns.tolist()

    results = []

    for method in methods:
        print(f"\n  {method.upper()}...")
        try:
            n_comp = 2 if method == 'tsne' else 4
            explainer = DRExplainer(method=method, n_components=n_comp)
            explainer.fit_transform(X, feature_names)

            surrogate_r2 = explainer.train_surrogates()
            recovery, _ = factor_recovery_score(explainer, membership)
            stability, _ = explanation_stability(method, X, n_runs=3)
            morf_auc, _ = morf_faithfulness(explainer, X, dim=0)

            results.append({
                'Regime': regime,
                'Method': method.upper(),
                'Surrogate_R2': np.mean(surrogate_r2),
                'Factor_Recovery': recovery,
                'Stability': stability,
                'MoRF_AUC': morf_auc,
            })

            print(f"    R²={np.mean(surrogate_r2):.3f}  Recovery={recovery:.3f}  "
                  f"Stability={stability:.3f}  MoRF={morf_auc:.3f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'Regime': regime,
                'Method': method.upper(),
                'Surrogate_R2': np.nan,
                'Factor_Recovery': np.nan,
                'Stability': np.nan,
                'MoRF_AUC': np.nan,
            })

    return pd.DataFrame(results)

def main():
    Path('outputs').mkdir(exist_ok=True)

    all_results = []

    for regime in ['idealised', 'moderate', 'realistic', 'noisy']:
        print(f"\n{'='*70}")
        print(f"  REGIME: {regime.upper()}")
        params = EMPIRICAL_PARAMS[regime]
        print(f"  Loadings: {params['loadings_range']}, Noise: {params['noise_range']}")
        print(f"  Expected factor R²: {params['expected_factor_r2']}")
        print('='*70)

        regime_df = run_single_regime(regime)
        all_results.append(regime_df)

    combined = pd.concat(all_results, ignore_index=True)

    # Save raw results
    combined.to_csv('outputs/regime_comparison_results.csv', index=False)

    # Print comparison tables
    print("\n" + "="*70)
    print("FULL RESULTS TABLE")
    print("="*70)
    print(combined.to_string(index=False, float_format='%.3f'))

    # Pivot tables for each metric
    for metric in ['Surrogate_R2', 'Factor_Recovery', 'Stability', 'MoRF_AUC']:
        print(f"\n--- {metric} by Regime and Method ---")
        pivot = combined.pivot(index='Method', columns='Regime', values=metric)
        pivot = pivot[['idealised', 'moderate', 'realistic']]
        print(pivot.to_string(float_format='%.3f'))

    # Literature comparison
    print("\n" + "="*70)
    print("LITERATURE COMPARISON")
    print("="*70)
    print("""
    Expected ranges from literature:
      Surrogate R² (PCA on real data):    0.50 - 0.75
      Surrogate R² (t-SNE on real data):  0.30 - 0.60
      Factor Recovery (real data):         0.30 - 0.55
      PCA variance explained (4 PCs):     0.25 - 0.40  (Fama-French)

    Our 'realistic' regime targets these ranges.
    If scores are still above these ranges, parameters need further adjustment.
    """)

    # Check if realistic regime is actually realistic
    realistic = combined[combined['Regime'] == 'realistic']
    pca_r2 = realistic[realistic['Method'] == 'PCA']['Surrogate_R2'].values
    pca_recovery = realistic[realistic['Method'] == 'PCA']['Factor_Recovery'].values

    if len(pca_r2) > 0 and pca_r2[0] > 0.75:
        print(f"WARNING: PCA surrogate R² = {pca_r2[0]:.3f} still too high for realistic regime")
        print("  Consider lowering loadings further or increasing noise")
    if len(pca_recovery) > 0 and pca_recovery[0] > 0.55:
        print(f"WARNING: Factor recovery = {pca_recovery[0]:.3f} still too high for realistic regime")
        print("  Consider adding more cross-loadings or reducing primary loadings")

    print(f"\nResults saved to: outputs/regime_comparison_results.csv")
    return combined

if __name__ == "__main__":
    start = time.time()
    results = main()
    print(f"\nTotal time: {time.time() - start:.1f}s")
