"""
Rule Generator for Explainable Dimensionality Reduction

Generates interpretable IF-THEN rules
from continuous factor scores using SHAP attributions and threshold discretization.

continuous XAI outputs into discrete logical predicates suitable for
financial decision-making and regulatory compliance.

Author: Bernardo Guterres
Date: February 2026
Supervisor: Dr. David Watson

- Lundberg & Lee (2017) - SHAP: A Unified Approach to Interpreting Model Predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import warnings

from threshold_discretizer import ThresholdDiscretizer

class Rule:
    """
    Data class representing a single IF-THEN rule.

    Attributes:
        factor: Factor name/index
        category: Discrete category (LOW/MEDIUM/HIGH)
        threshold_bounds: [lower, upper] threshold values
        features: List of top contributing feature names
        shap_values: SHAP contributions for each feature
        rule_text: Human-readable rule string
    """

    def __init__(
        self,
        factor: str,
        category: str,
        threshold_bounds: Tuple[float, float],
        features: List[str],
        shap_values: np.ndarray,
        rule_text: str
    ):
        self.factor = factor
        self.category = category
        self.threshold_bounds = threshold_bounds
        self.features = features
        self.shap_values = shap_values
        self.rule_text = rule_text

    def __repr__(self) -> str:
        return self.rule_text

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'factor': self.factor,
            'category': self.category,
            'threshold_bounds': self.threshold_bounds,
            'features': self.features,
            'shap_contributions': self.shap_values.tolist(),
            'rule_text': self.rule_text
        }

class RuleGenerator:
    """
    Generates interpretable IF-THEN rules from continuous XAI outputs.

    Threshold Discretization with Shapley Attribution.

    This class bridges the gap between continuous mathematical outputs from
    dimensionality reduction and discrete logical rules needed for:
    - Financial decision-making
    - Regulatory compliance (ECB, SR 11-7)
    - Business stakeholder communication

    Examples:
        >>> from xai_dr_comparison import DRExplainer
        >>>
        >>> # Get factor scores and SHAP values
        >>> explainer = DRExplainer(method='pca', n_components=4)
        >>> factor_scores = explainer.fit_transform(X, feature_names)
        >>> explainer.train_surrogates()
        >>>
        >>> # Get SHAP for each dimension
        >>> shap_list = []
        >>> for dim in range(4):
        >>>     shap_vals = explainer.explain_with_shap(X, dim=dim)
        >>>     shap_list.append(shap_vals)
        >>> shap_values = np.stack(shap_list, axis=1)  # (n, k, d)
        >>>
        >>> # Generate rules
        >>> rule_gen = RuleGenerator(top_n_features=5)
        >>> rules_dict = rule_gen.generate_rules(
        >>>     factor_scores, shap_values, feature_names
        >>> )
        >>>
        >>> # Print rules
        >>> for rule in rules_dict['rules']:
        >>>     print(rule)
    """

    def __init__(
        self,
        discretizer: Optional[ThresholdDiscretizer] = None,
        top_n_features: int = 5,
        epsilon: float = 0.01,
        min_samples_per_category: int = 10
    ):
        """
        Initialize rule generator.

        Args:
            discretizer: ThresholdDiscretizer instance (creates default if None)
            top_n_features: Number of top SHAP features to include in rules
            epsilon: Tolerance for SHAP completeness validation (as fraction of score range)
            min_samples_per_category: Minimum samples needed to generate a rule
        """
        self.discretizer = discretizer or ThresholdDiscretizer(method='quantile', n_bins=3)
        self.top_n_features = top_n_features
        self.epsilon = epsilon
        self.min_samples_per_category = min_samples_per_category

    def generate_rules(
        self,
        factor_scores: np.ndarray,
        shap_values: np.ndarray,
        feature_names: List[str],
        factor_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Main method implementing the rule generation algorithm.

        Args:
            factor_scores: (n, k) continuous factor scores from DR
            shap_values: (n, k, d) SHAP attributions
                - n: number of samples
                - k: number of factors/components
                - d: number of original features
            feature_names: List of d feature names
            factor_names: Optional list of k factor names (default: Factor 1, Factor 2, ...)

        Returns:
            Dict with structure:
                {
                    'discrete_labels': ndarray (n, k) - Categorical labels
                    'thresholds': dict - Threshold boundaries per factor
                    'rules': list of Rule objects
                    'validation': dict - SHAP completeness check results
                    'metadata': dict - Additional information
                }

        Raises:
            ValueError: If input shapes don't match
        """
        # Validate inputs
        factor_scores, shap_values = self._validate_inputs(
            factor_scores, shap_values, feature_names
        )

        n_samples, n_factors = factor_scores.shape
        n_features = len(feature_names)

        # Create default factor names if not provided
        if factor_names is None:
            factor_names = [f'Factor {i+1}' for i in range(n_factors)]

        # Step 1-3: Apply threshold discretization
        discrete_labels, thresholds = self.discretizer.fit_transform(factor_scores)
        label_mapping = self.discretizer.get_label_mapping()

        # Step 4-5: Generate rules for each factor and category
        rules_list = []

        for factor_idx in range(n_factors):
            for category_label, category_name in label_mapping.items():
                # Find samples in this category
                samples_in_category = np.where(discrete_labels[:, factor_idx] == category_label)[0]

                if len(samples_in_category) < self.min_samples_per_category:
                    warnings.warn(
                        f"Only {len(samples_in_category)} samples in "
                        f"{factor_names[factor_idx]} - {category_name} "
                        f"(minimum {self.min_samples_per_category} required). Skipping rule."
                    )
                    continue

                # Compute mean SHAP values for this category
                mean_shap = np.mean(
                    shap_values[samples_in_category, factor_idx, :],
                    axis=0
                )

                # Get top contributing features
                top_feature_indices = np.argsort(np.abs(mean_shap))[-self.top_n_features:][::-1]
                top_features = [feature_names[i] for i in top_feature_indices]
                top_shap_values = mean_shap[top_feature_indices]

                # Determine threshold bounds for this category
                threshold_bounds = self._get_threshold_bounds(
                    thresholds[factor_idx],
                    category_label,
                    factor_scores[:, factor_idx]
                )

                # Generate rule text
                rule_text = self._generate_rule_text(
                    factor_name=factor_names[factor_idx],
                    category=category_name,
                    top_features=top_features,
                    shap_contributions=top_shap_values,
                    threshold_bounds=threshold_bounds,
                    n_samples=len(samples_in_category)
                )

                # Create Rule object
                rule = Rule(
                    factor=factor_names[factor_idx],
                    category=category_name,
                    threshold_bounds=threshold_bounds,
                    features=top_features,
                    shap_values=top_shap_values,
                    rule_text=rule_text
                )

                rules_list.append(rule)

        # Step 6: Validate SHAP completeness
        validation_results = self._validate_shap_completeness(
            shap_values, factor_scores
        )

        # Compile results
        results = {
            'discrete_labels': discrete_labels,
            'thresholds': thresholds,
            'rules': rules_list,
            'validation': validation_results,
            'metadata': {
                'n_samples': n_samples,
                'n_factors': n_factors,
                'n_features': n_features,
                'discretization_method': self.discretizer.method,
                'n_bins': self.discretizer.n_bins,
                'top_n_features': self.top_n_features,
                'n_rules_generated': len(rules_list)
            }
        }

        return results

    def export_rules(
        self,
        rules_dict: Dict,
        format: str = 'json',
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Export rules to JSON, CSV, or LaTeX table.

        Args:
            rules_dict: Output from generate_rules()
            format: 'json', 'csv', or 'latex'
            output_path: Optional file path (auto-generated if None)

        Returns:
            Formatted string if no output_path, None otherwise
        """
        if format == 'json':
            return self._export_json(rules_dict, output_path)
        elif format == 'csv':
            return self._export_csv(rules_dict, output_path)
        elif format == 'latex':
            return self._export_latex(rules_dict, output_path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json', 'csv', or 'latex'")

    # ==================== Private Methods ====================

    def _validate_inputs(
        self,
        factor_scores: np.ndarray,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays and dimensions."""
        factor_scores = np.asarray(factor_scores)
        shap_values = np.asarray(shap_values)

        if factor_scores.ndim != 2:
            raise ValueError(f"factor_scores must be 2D, got shape {factor_scores.shape}")

        if shap_values.ndim != 3:
            raise ValueError(f"shap_values must be 3D, got shape {shap_values.shape}")

        n_samples, n_factors = factor_scores.shape
        shap_n, shap_k, shap_d = shap_values.shape

        if shap_n != n_samples:
            raise ValueError(
                f"Sample dimension mismatch: factor_scores has {n_samples}, "
                f"shap_values has {shap_n}"
            )

        if shap_k != n_factors:
            raise ValueError(
                f"Factor dimension mismatch: factor_scores has {n_factors}, "
                f"shap_values has {shap_k}"
            )

        if shap_d != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: feature_names has {len(feature_names)}, "
                f"shap_values has {shap_d}"
            )

        return factor_scores, shap_values

    def _get_threshold_bounds(
        self,
        thresholds: List[float],
        category_label: int,
        all_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Get lower and upper bounds for a category.

        For 3 categories with thresholds [t1, t2]:
        - Category 0 (LOW): (-inf, t1)
        - Category 1 (MED): [t1, t2)
        - Category 2 (HIGH): [t2, +inf)
        """
        if category_label == 0:
            # Lowest category
            return (float(all_scores.min()), thresholds[0])
        elif category_label == len(thresholds):
            # Highest category
            return (thresholds[-1], float(all_scores.max()))
        else:
            # Middle category
            return (thresholds[category_label - 1], thresholds[category_label])

    def _generate_rule_text(
        self,
        factor_name: str,
        category: str,
        top_features: List[str],
        shap_contributions: np.ndarray,
        threshold_bounds: Tuple[float, float],
        n_samples: int
    ) -> str:
        """
        Generate human-readable IF-THEN rule text.

        Example:
        "IF Growth Factor is HIGH (score ≥ 0.67, n=342)
         THEN primary drivers are:
           - Revenue_Growth (SHAP: +0.45)
           - EPS_Growth (SHAP: +0.38)
           - R&D_Intensity (SHAP: +0.22)"
        """
        lower, upper = threshold_bounds

        # Format threshold description
        if lower <= -1e6:
            threshold_desc = f"score < {upper:.2f}"
        elif upper >= 1e6:
            threshold_desc = f"score ≥ {lower:.2f}"
        else:
            threshold_desc = f"{lower:.2f} ≤ score < {upper:.2f}"

        # Build rule text
        rule_lines = [
            f"IF {factor_name} is {category} ({threshold_desc}, n={n_samples})",
            "THEN primary drivers are:"
        ]

        for feature, shap_val in zip(top_features, shap_contributions):
            sign = '+' if shap_val >= 0 else ''
            rule_lines.append(f"  - {feature} (SHAP: {sign}{shap_val:.3f})")

        return '\n'.join(rule_lines)

    def _validate_shap_completeness(
        self,
        shap_values: np.ndarray,
        factor_scores: np.ndarray
    ) -> Dict:
        """
        Validate SHAP completeness property: sum(SHAP) ≈ f(x) - baseline

        For each sample and factor, check:
        |sum(shap_values[i, f, :]) - (factor_scores[i, f] - baseline_f)| < epsilon

        CAVEAT: When using TreeSHAP on XGBoost surrogates, completeness holds
        exactly by mathematical construction (Lundberg et al., 2020). This test
        will virtually always pass and does not validate the pipeline's correctness.
        Additionally, the demo in __main__ constructs synthetic SHAP values with an
        explicit correction term that forces completeness, making that demonstration
        entirely circular. A meaningful completeness test would use SHAP values from
        the actual surrogate pipeline (train_surrogates + explain_with_shap), not
        synthetically constructed arrays.

        Returns:
            Dict with validation metrics:
            {
                'max_error': float - Maximum absolute error
                'mean_error': float - Mean absolute error
                'std_error': float - Standard deviation of error
                'threshold': float - Acceptable error threshold
                'passed': bool - True if max_error < threshold
                'error_distribution': ndarray - All errors for analysis
            }
        """
        n_samples, n_factors, _ = shap_values.shape

        # Compute baseline (mean of each factor)
        baseline = np.mean(factor_scores, axis=0)

        # For each sample and factor, compute: sum(SHAP) - (score - baseline)
        errors = np.zeros((n_samples, n_factors))

        for i in range(n_samples):
            for f in range(n_factors):
                shap_sum = np.sum(shap_values[i, f, :])
                expected = factor_scores[i, f] - baseline[f]
                errors[i, f] = np.abs(shap_sum - expected)

        # Compute statistics
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Adaptive threshold based on score range
        score_range = factor_scores.max() - factor_scores.min()
        threshold = self.epsilon * score_range

        passed = max_error < threshold

        return {
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'threshold': float(threshold),
            'passed': bool(passed),
            'error_distribution': errors.flatten()
        }

    def _export_json(self, rules_dict: Dict, output_path: Optional[Path]) -> Optional[str]:
        """Export to JSON format."""
        export_data = {
            'metadata': rules_dict['metadata'],
            'thresholds': {str(k): v for k, v in rules_dict['thresholds'].items()},
            'validation': {
                k: v for k, v in rules_dict['validation'].items()
                if k != 'error_distribution'  # Too large for JSON
            },
            'rules': [rule.to_dict() for rule in rules_dict['rules']]
        }

        if output_path is None:
            method = rules_dict['metadata']['discretization_method']
            output_path = Path('/home/claude/outputs') / f'rules_{method}.json'

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Rules exported to: {output_path}")
        return None

    def _export_csv(self, rules_dict: Dict, output_path: Optional[Path]) -> Optional[str]:
        """Export to CSV format."""
        rows = []
        for rule in rules_dict['rules']:
            for feature, shap in zip(rule.features, rule.shap_values):
                rows.append({
                    'Factor': rule.factor,
                    'Category': rule.category,
                    'Threshold_Lower': rule.threshold_bounds[0],
                    'Threshold_Upper': rule.threshold_bounds[1],
                    'Feature': feature,
                    'SHAP_Contribution': shap
                })

        df = pd.DataFrame(rows)

        if output_path is None:
            method = rules_dict['metadata']['discretization_method']
            output_path = Path('/home/claude/outputs') / f'rules_{method}.csv'

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        df.to_csv(output_path, index=False)
        print(f"Rules exported to: {output_path}")
        return None

    def _export_latex(self, rules_dict: Dict, output_path: Optional[Path]) -> str:
        """Export to LaTeX table format."""
        latex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{llrrl}",
            r"\hline",
            r"Factor & Category & Lower & Upper & Top Feature (SHAP) \\",
            r"\hline"
        ]

        for rule in rules_dict['rules']:
            factor = rule.factor.replace('_', r'\_')
            category = rule.category
            lower, upper = rule.threshold_bounds

            # Format thresholds
            lower_str = f"{lower:.2f}" if lower > -1e6 else r"$-\infty$"
            upper_str = f"{upper:.2f}" if upper < 1e6 else r"$+\infty$"

            # Top feature
            top_feat = rule.features[0].replace('_', r'\_')
            top_shap = rule.shap_values[0]

            latex_lines.append(
                f"{factor} & {category} & {lower_str} & {upper_str} & "
                f"{top_feat} ({top_shap:+.3f}) \\\\"
            )

        latex_lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\caption{Generated IF-THEN Rules}",
            r"\end{table}"
        ])

        latex_str = '\n'.join(latex_lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                f.write(latex_str)
            print(f"LaTeX table exported to: {output_path}")

        return latex_str

# ==================== Demo/Testing ====================

if __name__ == '__main__':
    """
    Demonstration and basic testing of RuleGenerator.

    This demo validates that the rule generation algorithm is correctly implemented.
    """
    print("="*80)
    print("RuleGenerator - Demonstration (the rule generation algorithm)")
    print("="*80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_factors = 4
    n_features = 20

    # Simulate factor scores (from PCA or other DR)
    factor_scores = np.random.randn(n_samples, n_factors)

    # Simulate SHAP values (would come from TreeSHAP in real use)
    # In reality: explainer.explain_with_shap(X, dim=f) for each factor f
    shap_values = np.random.randn(n_samples, n_factors, n_features) * 0.1

    # Ensure SHAP completeness property holds (for demo purposes)
    baseline = np.mean(factor_scores, axis=0)
    for i in range(n_samples):
        for f in range(n_factors):
            # Make sum(SHAP) = score - baseline (with small noise)
            current_sum = np.sum(shap_values[i, f, :])
            target = factor_scores[i, f] - baseline[f]
            correction = (target - current_sum) / n_features
            shap_values[i, f, :] += correction

    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    factor_names = ['Growth', 'Value', 'Quality', 'Momentum']

    print(f"\nGenerated synthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Factors: {n_factors}")
    print(f"  Features: {n_features}")

    # Test with different discretization methods
    methods = ['quantile', 'std', 'hybrid']

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing with {method.upper()} discretization")
        print(f"{'='*80}")

        discretizer = ThresholdDiscretizer(method=method, n_bins=3)
        rule_gen = RuleGenerator(discretizer=discretizer, top_n_features=5)

        rules_dict = rule_gen.generate_rules(
            factor_scores=factor_scores,
            shap_values=shap_values,
            feature_names=feature_names,
            factor_names=factor_names
        )

        # Print summary
        print(f"\nGenerated {len(rules_dict['rules'])} rules")
        print(f"Discretization method: {rules_dict['metadata']['discretization_method']}")

        # Print validation results
        val = rules_dict['validation']
        print(f"\nSHAP Completeness Validation:")
        print(f"  Max error: {val['max_error']:.6f}")
        print(f"  Mean error: {val['mean_error']:.6f}")
        print(f"  Std error: {val['std_error']:.6f}")
        print(f"  Threshold: {val['threshold']:.6f}")
        print(f"  Passed: {'✓' if val['passed'] else '✗'}")

        # Print sample rules
        print(f"\nSample Rules:")
        for i, rule in enumerate(rules_dict['rules'][:3]):  # First 3 rules
            print(f"\n--- Rule {i+1} ---")
            print(rule)

        # Export rules
        rule_gen.export_rules(rules_dict, format='json')

    print(f"\n{'='*80}")
    print("All tests passed! ✓")
    print(f"{'='*80}")
    print("\nExported files:")
    print("  - /home/claude/outputs/rules_quantile.json")
    print("  - /home/claude/outputs/rules_std.json")
    print("  - /home/claude/outputs/rules_hybrid.json")

    print("\nthe rule generation algorithm implementation complete!")
    print("Next step: Integration with existing experiments (PROMPT_3)")
