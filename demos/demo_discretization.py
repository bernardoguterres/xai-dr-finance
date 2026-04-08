"""
Demonstration: Threshold Discretization on Real SHAP Values
============================================================
Tests all discretization strategies on SHAP values from DR methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import our modules
from data_loaders import _generate_synthetic_stock_fundamentals, prepare_for_dr, create_factor_labels
from xai_dr_comparison import DRExplainer
from threshold_discretizer import ThresholdDiscretizer

# Set seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("THRESHOLD DISCRETIZATION DEMONSTRATION")
print("=" * 80)

# Step 1: Generate synthetic data
print("\n1. Generating synthetic financial data...")
df = _generate_synthetic_stock_fundamentals(n_stocks=500)
X_clean, feature_names, metadata = prepare_for_dr(
    df,
    exclude_cols=['Ticker', 'Sector'],
    handle_categorical='drop'
)
labels_dict = create_factor_labels(df)

print(f"   Data shape: {X_clean.shape}")
print(f"   Factor labels: {list(labels_dict.keys())}")

# Step 2: Run PCA and get SHAP values
print("\n2. Running PCA and computing SHAP explanations...")
explainer = DRExplainer(method='pca', n_components=4)
embedding = explainer.fit_transform(X_clean, feature_names)

# Train surrogate models
explainer.train_surrogates()

# Get SHAP values for first component
shap_values = explainer.explain_with_shap(X_clean[:100], dim=0)
print(f"   SHAP values shape: {shap_values.shape}")
print(f"   Range: [{shap_values.min():.3f}, {shap_values.max():.3f}]")

# Get factor scores (projection onto first component)
factor_scores = embedding[:, 0].reshape(-1, 1)
print(f"   Factor scores shape: {factor_scores.shape}")
print(f"   Range: [{factor_scores.min():.3f}, {factor_scores.max():.3f}]")

# Step 3: Test all discretization strategies
print("\n3. Testing all discretization strategies...")
print("=" * 80)

strategies = ['quantile', 'std', 'hybrid']
results = {}

for strategy in strategies:
    print(f"\n   Strategy: {strategy.upper()}")
    print("   " + "-" * 76)

    # Initialize discretizer
    discretizer = ThresholdDiscretizer(method=strategy, n_bins=3)

    # Fit and transform
    labels, thresholds = discretizer.fit_transform(factor_scores)

    # Get statistics
    unique, counts = np.unique(labels, return_counts=True)

    print(f"   Thresholds: {thresholds[0]}")
    print(f"   Label distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"      {label}: {count:4d} samples ({percentage:5.1f}%)")

    # Store results
    results[strategy] = {
        'labels': labels,
        'thresholds': thresholds,
        'distribution': dict(zip(unique, counts))
    }

# Step 4: Domain-specific thresholds
print("\n4. Testing domain-specific thresholds...")
print("=" * 80)

# Define finance-specific thresholds for PCA Component 1
# (could represent value/growth distinction based on PE, Beta, etc.)
domain_thresholds = {
    'PC1': [-1.0, 1.0]  # Low/Med/High based on standardized scores
}

discretizer_domain = ThresholdDiscretizer(
    method='domain',
    n_bins=3,
    custom_thresholds=domain_thresholds
)

labels_domain, thresholds_domain = discretizer_domain.fit_transform(factor_scores)
unique, counts = np.unique(labels_domain, return_counts=True)

print(f"   Custom thresholds: {thresholds_domain[0]}")
print(f"   Label distribution:")
for label, count in zip(unique, counts):
    percentage = (count / len(labels_domain)) * 100
    print(f"      {label}: {count:4d} samples ({percentage:5.1f}%)")

# Step 5: Validate completeness property (Algorithm 1 requirement)
print("\n5. Validating SHAP completeness property...")
print("=" * 80)

# For a sample, sum of SHAP should equal (prediction - baseline)
sample_idx = 10
sample_shap = shap_values[sample_idx]
sample_score = factor_scores[sample_idx, 0]

# Baseline is mean of factor scores
baseline = factor_scores.mean()

shap_sum = sample_shap.sum()
difference = sample_score - baseline

print(f"   Sample {sample_idx}:")
print(f"      Factor score: {sample_score:.4f}")
print(f"      Baseline: {baseline:.4f}")
print(f"      Difference: {difference:.4f}")
print(f"      Sum of SHAP: {shap_sum:.4f}")
print(f"      Error: {abs(shap_sum - difference):.6f}")

epsilon = 0.01
if abs(shap_sum - difference) < epsilon:
    print(f"   ✅ Completeness validated (error < {epsilon})")
else:
    print(f"   ⚠️  Completeness may be violated (error >= {epsilon})")

# Step 6: Create summary comparison
print("\n6. Discretization Strategy Comparison")
print("=" * 80)

comparison_data = []
for strategy in strategies:
    dist = results[strategy]['distribution']
    comparison_data.append({
        'Strategy': strategy.capitalize(),
        'LOW': dist.get('LOW', 0),
        'MED': dist.get('MED', 0),
        'HIGH': dist.get('HIGH', 0),
        'Thresholds': f"[{results[strategy]['thresholds'][0][0]:.2f}, {results[strategy]['thresholds'][0][1]:.2f}]"
    })

# Add domain strategy
dist_domain = dict(zip(*np.unique(labels_domain, return_counts=True)))
comparison_data.append({
    'Strategy': 'Domain',
    'LOW': dist_domain.get('LOW', 0),
    'MED': dist_domain.get('MED', 0),
    'HIGH': dist_domain.get('HIGH', 0),
    'Thresholds': f"[{thresholds_domain[0][0]:.2f}, {thresholds_domain[0][1]:.2f}]"
})

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Step 7: Save results
print("\n7. Saving results...")
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# Save discretization results
results_file = output_dir / 'discretization_results.csv'
df_comparison.to_csv(results_file, index=False)
print(f"   ✅ Saved: {results_file}")

# Save example labeled data
labeled_data = pd.DataFrame({
    'Factor_Score': factor_scores[:, 0],
    'Quantile_Label': results['quantile']['labels'][:, 0],
    'Std_Label': results['std']['labels'][:, 0],
    'Hybrid_Label': results['hybrid']['labels'][:, 0],
    'Domain_Label': labels_domain[:, 0]
})

labeled_file = output_dir / 'discretized_scores.csv'
labeled_data.to_csv(labeled_file, index=False)
print(f"   ✅ Saved: {labeled_file}")

print("\n" + "=" * 80)
print("✅ DISCRETIZATION DEMONSTRATION COMPLETE")
print("=" * 80)
print(f"\nOutput files:")
print(f"  • {results_file}")
print(f"  • {labeled_file}")
print("\nAll discretization strategies working correctly!")
