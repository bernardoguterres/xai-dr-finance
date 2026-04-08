"""
Demonstration: IF-THEN Rule Generation from DR Explanations
============================================================
Demonstrates BSPR Algorithm 1: Converting continuous XAI outputs to discrete rules.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import our modules
from data_loaders import _generate_synthetic_stock_fundamentals, prepare_for_dr
from xai_dr_comparison import DRExplainer
from rule_generator import RuleGenerator
from threshold_discretizer import ThresholdDiscretizer

# Set seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("IF-THEN RULE GENERATION DEMONSTRATION")
print("=" * 80)

# Step 1: Generate synthetic data
print("\n1. Generating synthetic financial data...")
df = _generate_synthetic_stock_fundamentals(n_stocks=500)
X_clean, feature_names, metadata = prepare_for_dr(
    df,
    exclude_cols=['Ticker', 'Sector'],
    handle_categorical='drop'
)

print(f"   Data shape: {X_clean.shape}")
print(f"   Features: {len(feature_names)}")

# Step 2: Run PCA and get SHAP explanations
print("\n2. Running PCA and computing SHAP values...")
explainer = DRExplainer(method='pca', n_components=4)
factor_scores = explainer.fit_transform(X_clean, feature_names)

# Train surrogate models for SHAP
explainer.train_surrogates()

# Get SHAP values for each dimension
print("   Computing SHAP values for each component...")
shap_list = []
for dim in range(4):
    shap_vals = explainer.explain_with_shap(X_clean, dim=dim)
    shap_list.append(shap_vals)
    print(f"   Component {dim+1}: shape {shap_vals.shape}")

# Stack into (n_samples, n_components, n_features) format
shap_values = np.stack(shap_list, axis=1)
print(f"   Final SHAP shape: {shap_values.shape}")

# Step 3: Generate rules with different strategies
print("\n3. Generating IF-THEN rules...")
print("=" * 80)

strategies = ['quantile', 'std', 'domain']
all_rules = {}

for strategy in strategies:
    print(f"\n   Strategy: {strategy.upper()}")
    print("   " + "-" * 76)

    # Initialize rule generator
    if strategy == 'domain':
        # Use finance-specific thresholds
        custom_thresholds = {
            f'PC{i+1}': [-1.0, 1.0] for i in range(4)
        }
        discretizer = ThresholdDiscretizer(
            method=strategy,
            n_bins=3,
            custom_thresholds=custom_thresholds
        )
        rule_gen = RuleGenerator(discretizer=discretizer)
    else:
        discretizer = ThresholdDiscretizer(method=strategy, n_bins=3)
        rule_gen = RuleGenerator(discretizer=discretizer)

    # Generate rules (Algorithm 1)
    result = rule_gen.generate_rules(
        factor_scores=factor_scores,
        shap_values=shap_values,
        feature_names=feature_names,
        factor_names=[f'PC{i+1}' for i in range(4)]
    )
    rules = result['rules']

    all_rules[strategy] = rules

    # Print some example rules
    print(f"   Generated {len(rules)} rules total")
    print(f"\n   Sample rules:")
    for rule in rules[:3]:  # Show first 3 rules
        print(f"      • {rule.rule_text}")

# Step 4: Display detailed rule analysis
print("\n4. Detailed Rule Analysis (Quantile Strategy)")
print("=" * 80)

quantile_rules = all_rules['quantile']

# Group rules by category
rules_by_category = {}
for rule in quantile_rules:
    key = f"{rule.factor}_{rule.category}"
    rules_by_category[key] = rule

# Show one detailed example for each category
print("\nExample: PC1 (First Principal Component)")
print("-" * 80)

for category in ['LOW', 'MED', 'HIGH']:
    key = f'PC1_{category}'
    if key in rules_by_category:
        rule = rules_by_category[key]
        print(f"\nCategory: {category}")
        print(f"Rule: {rule.rule_text}")
        print(f"Threshold range: [{rule.threshold_bounds[0]:.3f}, {rule.threshold_bounds[1]:.3f}]")
        print(f"Top features and SHAP contributions:")
        for feat, shap_val in zip(rule.features, rule.shap_values):
            print(f"  • {feat:20s}: {shap_val:+.4f}")

# Step 5: Export rules to structured formats
print("\n5. Exporting rules...")
print("=" * 80)

output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# Export as JSON (for programmatic use)
for strategy, rules in all_rules.items():
    rules_json = {
        'discretization_method': strategy,
        'n_rules': len(rules),
        'rules': [rule.to_dict() for rule in rules]
    }

    json_file = output_dir / f'rules_{strategy}.json'
    import json
    with open(json_file, 'w') as f:
        json.dump(rules_json, f, indent=2)
    print(f"   ✅ Saved: {json_file}")

# Export as CSV (for spreadsheet/analysis)
rules_data = []
for strategy, rules in all_rules.items():
    for rule in rules:
        rules_data.append({
            'Strategy': strategy,
            'Factor': rule.factor,
            'Category': rule.category,
            'Lower_Bound': rule.threshold_bounds[0],
            'Upper_Bound': rule.threshold_bounds[1],
            'Top_Feature_1': rule.features[0] if len(rule.features) > 0 else '',
            'SHAP_1': rule.shap_values[0] if len(rule.shap_values) > 0 else 0,
            'Top_Feature_2': rule.features[1] if len(rule.features) > 1 else '',
            'SHAP_2': rule.shap_values[1] if len(rule.shap_values) > 1 else 0,
            'Rule_Text': rule.rule_text
        })

df_rules = pd.DataFrame(rules_data)
csv_file = output_dir / 'all_rules.csv'
df_rules.to_csv(csv_file, index=False)
print(f"   ✅ Saved: {csv_file}")

# Export human-readable text file
txt_file = output_dir / 'rules_human_readable.txt'
with open(txt_file, 'w') as f:
    f.write("INTERPRETABLE IF-THEN RULES FOR DIMENSIONALITY REDUCTION\n")
    f.write("=" * 80 + "\n\n")

    for strategy, rules in all_rules.items():
        f.write(f"\nDiscretization Strategy: {strategy.upper()}\n")
        f.write("-" * 80 + "\n\n")

        for rule in rules:
            f.write(f"{rule.rule_text}\n")
            f.write(f"  Threshold: [{rule.threshold_bounds[0]:.3f}, {rule.threshold_bounds[1]:.3f}]\n")
            f.write(f"  Key drivers:\n")
            for feat, shap in zip(rule.features[:3], rule.shap_values[:3]):
                f.write(f"    - {feat}: {shap:+.4f}\n")
            f.write("\n")

print(f"   ✅ Saved: {txt_file}")

# Step 6: Rule quality metrics
print("\n6. Rule Quality Metrics")
print("=" * 80)

# Coverage: percentage of samples assigned to each category
for strategy, rules in all_rules.items():
    print(f"\n   Strategy: {strategy.upper()}")

    # Count rules per category
    category_counts = {}
    for rule in rules:
        if rule.category not in category_counts:
            category_counts[rule.category] = 0
        category_counts[rule.category] += 1

    print(f"   Rules per category: {category_counts}")
    print(f"   Total rules: {len(rules)}")

# Step 7: Demonstrate rule usage
print("\n7. Example: Using Rules for Classification")
print("=" * 80)

# Pick a test sample
test_idx = 42
test_score_pc1 = factor_scores[test_idx, 0]

print(f"\nTest sample #{test_idx}")
print(f"PC1 score: {test_score_pc1:.3f}")

# Find which rule applies (using quantile strategy)
quantile_rules = all_rules['quantile']
for rule in quantile_rules:
    if rule.factor == 'PC1':
        lower, upper = rule.threshold_bounds
        if lower <= test_score_pc1 < upper or (rule.category == 'HIGH' and test_score_pc1 >= lower):
            print(f"\nMatched rule: {rule.rule_text}")
            print(f"Classification: {rule.category}")
            print(f"Key features driving this classification:")
            for feat, shap in zip(rule.features[:3], rule.shap_values[:3]):
                print(f"  • {feat:20s}: {shap:+.4f}")
            break

print("\n" + "=" * 80)
print("✅ RULE GENERATION DEMONSTRATION COMPLETE")
print("=" * 80)
print(f"\nGenerated rules using {len(strategies)} discretization strategies")
print(f"Output files:")
print(f"  • rules_*.json (programmatic use)")
print(f"  • all_rules.csv (analysis)")
print(f"  • rules_human_readable.txt (documentation)")
print("\nThese rules convert continuous factor scores into actionable insights!")
