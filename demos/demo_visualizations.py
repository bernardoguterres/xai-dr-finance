"""
Comprehensive Visualizations for Simulated Data Results
========================================================
Creates publication-quality visualizations for the thesis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our modules
from data_loaders import _generate_synthetic_stock_fundamentals, prepare_for_dr, create_factor_labels
from xai_dr_comparison import DRExplainer
from threshold_discretizer import ThresholdDiscretizer

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

np.random.seed(42)

print("=" * 80)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Create output directory
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# Step 1: Generate data
print("\n1. Generating synthetic data...")
df = _generate_synthetic_stock_fundamentals(n_stocks=500)
X_clean, feature_names, metadata = prepare_for_dr(
    df,
    exclude_cols=['Ticker', 'Sector'],
    handle_categorical='drop'
)
labels_dict = create_factor_labels(df)

# Step 2: Run multiple DR methods
print("\n2. Running DR methods...")
methods = ['pca', 'nmf', 'factor_analysis', 'tsne']
embeddings = {}
shap_values_dict = {}

for method in methods:
    print(f"   - {method.upper()}")
    explainer = DRExplainer(method=method, n_components=2)
    emb = explainer.fit_transform(X_clean, feature_names)
    embeddings[method] = emb

    # Get SHAP for first component
    explainer.train_surrogates()
    shap_vals = explainer.explain_with_shap(X_clean, dim=0)
    shap_values_dict[method] = shap_vals

# Step 3: Visualization 1 - DR Method Comparison
print("\n3. Creating DR method comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Dimensionality Reduction Methods Comparison', fontsize=16, fontweight='bold')

for idx, (method, ax) in enumerate(zip(methods, axes.flat)):
    emb = embeddings[method]

    # Color by Value_Style
    colors = labels_dict['Value_Style']
    color_map = {'Value': 0, 'Core': 1, 'Growth': 2}
    color_values = [color_map[c] for c in colors]

    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=color_values, cmap='viridis', alpha=0.6, s=30)
    ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)

    # Add legend
    if idx == 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.0), markersize=8, label='Value'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.5), markersize=8, label='Core'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1.0), markersize=8, label='Growth')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
viz1_path = output_dir / 'viz_dr_comparison.png'
plt.savefig(viz1_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {viz1_path}")
plt.close()

# Step 4: Visualization 2 - SHAP Feature Importance
print("\n4. Creating SHAP importance comparison...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Top Feature Importances (SHAP) by DR Method', fontsize=16, fontweight='bold')

for idx, (method, ax) in enumerate(zip(methods, axes.flat)):
    shap_vals = shap_values_dict[method]

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Get top 10 features
    top_indices = np.argsort(mean_abs_shap)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_values, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(f'{method.upper()}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
viz2_path = output_dir / 'viz_shap_importance.png'
plt.savefig(viz2_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {viz2_path}")
plt.close()

# Step 5: Visualization 3 - Discretization Strategies Comparison
print("\n5. Creating discretization comparison...")
explainer = DRExplainer(method='pca', n_components=4)
factor_scores = explainer.fit_transform(X_clean, feature_names)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Threshold Discretization Strategies', fontsize=16, fontweight='bold')

strategies = ['quantile', 'std', 'hybrid']
colors_disc = {'LOW': 'blue', 'MED': 'green', 'HIGH': 'red'}

for idx, (strategy, ax) in enumerate(zip(strategies, axes.flat[:3])):
    discretizer = ThresholdDiscretizer(method=strategy, n_bins=3)
    labels, thresholds = discretizer.fit_transform(factor_scores[:, 0:1])

    # Plot distribution
    for category in ['LOW', 'MED', 'HIGH']:
        mask = labels[:, 0] == category
        scores = factor_scores[mask, 0]
        ax.hist(scores, bins=30, alpha=0.5, label=category, color=colors_disc[category])

    # Add threshold lines
    for threshold in thresholds[0]:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_title(f'{strategy.upper()} Discretization', fontsize=13, fontweight='bold')
    ax.set_xlabel('Factor Score (PC1)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# Remove empty subplot
axes.flat[3].axis('off')

plt.tight_layout()
viz3_path = output_dir / 'viz_discretization.png'
plt.savefig(viz3_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {viz3_path}")
plt.close()

# Step 6: Visualization 4 - Factor Distribution by Labels
print("\n6. Creating factor distribution analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Factor Scores by Investment Characteristics', fontsize=16, fontweight='bold')

label_types = ['Value_Style', 'Quality_Tier', 'Size', 'Risk_Level']

for idx, (label_type, ax) in enumerate(zip(label_types, axes.flat)):
    # Get labels
    labels_data = labels_dict[label_type]
    unique_labels = np.unique(labels_data)

    # Box plot
    data_to_plot = []
    for label in unique_labels:
        mask = labels_data == label
        data_to_plot.append(factor_scores[mask, 0])

    bp = ax.boxplot(data_to_plot, labels=unique_labels, patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_title(f'PC1 Distribution by {label_type}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Factor Score (PC1)', fontsize=11)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
viz4_path = output_dir / 'viz_factor_distributions.png'
plt.savefig(viz4_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {viz4_path}")
plt.close()

# Step 7: Visualization 5 - Feature Correlation Heatmap
print("\n7. Creating feature correlation heatmap...")
fig, ax = plt.subplots(figsize=(14, 12))

# Calculate correlation matrix
corr_matrix = pd.DataFrame(X_clean, columns=feature_names).corr()

# Create heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)

ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
viz5_path = output_dir / 'viz_correlation_heatmap.png'
plt.savefig(viz5_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {viz5_path}")
plt.close()

# Step 8: Visualization 6 - Explained Variance
print("\n8. Creating explained variance plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('PCA Explained Variance Analysis', fontsize=16, fontweight='bold')

# Get PCA variance
explainer_pca = DRExplainer(method='pca', n_components=10)
explainer_pca.fit_transform(X_clean, feature_names)
var_explained = explainer_pca.model.explained_variance_ratio_

# Scree plot
ax1.plot(range(1, 11), var_explained, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('Scree Plot', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xticks(range(1, 11))

# Cumulative variance
cumulative_var = np.cumsum(var_explained)
ax2.plot(range(1, 11), cumulative_var, 'ro-', linewidth=2, markersize=8)
ax2.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='90% threshold')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xticks(range(1, 11))

plt.tight_layout()
viz6_path = output_dir / 'viz_variance_explained.png'
plt.savefig(viz6_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {viz6_path}")
plt.close()

print("\n" + "=" * 80)
print("✅ VISUALIZATION SUITE COMPLETE")
print("=" * 80)
print(f"\nGenerated 6 comprehensive visualizations:")
print(f"  1. {viz1_path} - DR method comparison")
print(f"  2. {viz2_path} - SHAP feature importance")
print(f"  3. {viz3_path} - Discretization strategies")
print(f"  4. {viz4_path} - Factor distributions by labels")
print(f"  5. {viz5_path} - Feature correlation heatmap")
print(f"  6. {viz6_path} - Explained variance analysis")
print("\nAll visualizations ready for thesis!")
