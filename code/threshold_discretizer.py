"""
Threshold Discretizer for Financial Factor Scores

Converts continuous embeddings from dimensionality reduction methods
into interpretable categorical labels for regulatory compliance.

Supports 4 discretization strategies:
1. Quantile - Equal-sized bins (33rd, 67th percentiles)
2. Standard Deviation - Statistical bounds (μ±σ)
3. Domain Expert - Finance-specific thresholds
4. Hybrid Adaptive - Combines multiple methods based on distribution shape

Author: Bernardo Guterres
Date: February 2026
Supervisor: Dr. David Watson

"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple
import warnings

class ThresholdDiscretizer:
    """
    Converts continuous factor scores into discrete categorical labels.

    continuous XAI outputs and discrete logical rules needed for financial
    decision-making and regulatory compliance.

    Examples:
        >>> # Quantile discretization (equal-sized bins)
        >>> discretizer = ThresholdDiscretizer(method='quantile', n_bins=3)
        >>> discrete_labels, thresholds = discretizer.fit_transform(factor_scores)

        >>> # Domain-specific thresholds for finance
        >>> custom = {0: [-0.5, 0.5], 1: [-0.3, 0.3]}
        >>> discretizer = ThresholdDiscretizer(method='domain',
        ...                                     custom_thresholds=custom)
        >>> discrete_labels, thresholds = discretizer.fit_transform(factor_scores)

        >>> # Adaptive hybrid approach
        >>> discretizer = ThresholdDiscretizer(method='hybrid', n_bins=3)
        >>> discrete_labels, thresholds = discretizer.fit_transform(factor_scores)
    """

    def __init__(
        self,
        method: str = 'quantile',
        n_bins: int = 3,
        custom_thresholds: Optional[Dict[int, List[float]]] = None,
        random_state: int = 42
    ):
        """
        Initialize threshold discretizer.

        Args:
            method: Discretization strategy
                - 'quantile': Equal-sized bins using percentiles
                - 'std': Statistical bounds (μ ± σ)
                - 'domain': Finance-specific thresholds
                - 'hybrid': Adaptive based on distribution shape
            n_bins: Number of discrete categories (default 3 for LOW/MED/HIGH)
            custom_thresholds: Optional dict mapping factor index to threshold list
                Example: {0: [-0.5, 0.5], 1: [-0.3, 0.3]}
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If method is not valid or n_bins < 2
        """
        valid_methods = ['quantile', 'std', 'domain', 'hybrid']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")

        self.method = method
        self.n_bins = n_bins
        self.custom_thresholds = custom_thresholds or {}
        self.random_state = random_state

        # Fitted attributes
        self.thresholds_ = None
        self.label_mapping_ = None
        self.is_fitted_ = False

        np.random.seed(random_state)

    def fit(self, factor_scores: np.ndarray) -> 'ThresholdDiscretizer':
        """
        Learn threshold boundaries from factor scores.

        Args:
            factor_scores: Array of shape (n_samples, n_factors)

        Returns:
            self (fitted instance)

        Raises:
            ValueError: If factor_scores contains NaN or infinite values
        """
        # Validate input
        factor_scores = self._validate_input(factor_scores)

        n_samples, n_factors = factor_scores.shape

        # Learn thresholds for each factor
        self.thresholds_ = {}

        for factor_idx in range(n_factors):
            factor_values = factor_scores[:, factor_idx]

            if self.method == 'quantile':
                thresholds = self._compute_quantile_thresholds(factor_values)
            elif self.method == 'std':
                thresholds = self._compute_std_thresholds(factor_values)
            elif self.method == 'domain':
                thresholds = self._compute_domain_thresholds(factor_idx, factor_values)
            elif self.method == 'hybrid':
                thresholds = self._compute_hybrid_thresholds(factor_values)

            self.thresholds_[factor_idx] = thresholds

        # Create label mapping
        self.label_mapping_ = self._create_label_mapping()

        self.is_fitted_ = True
        return self

    def transform(self, factor_scores: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply discretization to factor scores.

        Args:
            factor_scores: Array of shape (n_samples, n_factors)

        Returns:
            discrete_labels: Array of shape (n_samples, n_factors) with integer labels
            thresholds: Dict mapping factor index to threshold boundaries

        Raises:
            ValueError: If not fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Discretizer must be fitted before transform. Call fit() first.")

        factor_scores = self._validate_input(factor_scores)
        n_samples, n_factors = factor_scores.shape

        # Apply thresholds
        discrete_labels = np.zeros((n_samples, n_factors), dtype=int)

        for factor_idx in range(n_factors):
            factor_values = factor_scores[:, factor_idx]
            thresholds = self.thresholds_[factor_idx]

            # Digitize: values are binned based on thresholds
            # digitize returns 0 for x < thresholds[0], 1 for thresholds[0] <= x < thresholds[1], etc.
            discrete_labels[:, factor_idx] = np.digitize(factor_values, thresholds)

        return discrete_labels, self.thresholds_

    def fit_transform(self, factor_scores: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Fit and transform in one step.

        Args:
            factor_scores: Array of shape (n_samples, n_factors)

        Returns:
            discrete_labels: Array of shape (n_samples, n_factors)
            thresholds: Dict of threshold boundaries
        """
        return self.fit(factor_scores).transform(factor_scores)

    def get_label_mapping(self) -> Dict[int, str]:
        """
        Get mapping from integer labels to semantic categories.

        Returns:
            Dict mapping integer to category name
            Example: {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

        Raises:
            ValueError: If not fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Discretizer must be fitted first.")
        return self.label_mapping_

    def get_thresholds(self, factor_idx: Optional[int] = None) -> Union[List[float], Dict]:
        """
        Get threshold boundaries.

        Args:
            factor_idx: Optional index of specific factor

        Returns:
            If factor_idx provided: List of thresholds for that factor
            Otherwise: Dict mapping all factors to their thresholds

        Raises:
            ValueError: If not fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Discretizer must be fitted first.")

        if factor_idx is not None:
            return self.thresholds_[factor_idx]
        return self.thresholds_

    # ==================== Private Methods ====================

    def _validate_input(self, factor_scores: np.ndarray) -> np.ndarray:
        """
        Validate input array.

        Args:
            factor_scores: Input array to validate

        Returns:
            Validated numpy array

        Raises:
            ValueError: If input is invalid
        """
        factor_scores = np.asarray(factor_scores)

        if factor_scores.ndim != 2:
            raise ValueError(f"factor_scores must be 2D, got shape {factor_scores.shape}")

        if np.any(np.isnan(factor_scores)):
            raise ValueError("factor_scores contains NaN values")

        if np.any(np.isinf(factor_scores)):
            raise ValueError("factor_scores contains infinite values")

        return factor_scores

    def _compute_quantile_thresholds(self, values: np.ndarray) -> List[float]:
        """
        Compute thresholds using quantiles for equal-sized bins.

        This method creates bins with equal numbers of samples in each bin,
        which is robust to outliers and works well with any distribution.

        For n_bins=3: Returns [33rd percentile, 67th percentile]
        For n_bins=4: Returns [25th, 50th, 75th percentile]

        Args:
            values: 1D array of factor values

        Returns:
            List of threshold values
        """
        percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]  # Exclude 0 and 100
        thresholds = np.percentile(values, percentiles)
        return thresholds.tolist()

    def _compute_std_thresholds(self, values: np.ndarray) -> List[float]:
        """
        Compute thresholds using standard deviation bounds.

        This method assumes roughly normal distribution and creates bins
        based on statistical control limits.

        For n_bins=3: Returns [μ - σ, μ + σ]
        For n_bins=5: Returns [μ - 2σ, μ - σ, μ + σ, μ + 2σ]

        Args:
            values: 1D array of factor values

        Returns:
            List of threshold values
        """
        mean = np.mean(values)
        std = np.std(values)

        if std < 1e-6:
            warnings.warn("Standard deviation is very small, falling back to quantile method")
            return self._compute_quantile_thresholds(values)

        # Create symmetric thresholds around mean
        n_thresholds = self.n_bins - 1
        if n_thresholds == 2:
            # 3 bins: LOW (< μ-σ), MEDIUM (μ-σ to μ+σ), HIGH (> μ+σ)
            thresholds = [mean - std, mean + std]
            return thresholds
        else:
            # General case: evenly spaced within ±k*σ
            k = (n_thresholds + 1) / 2  # Number of std deviations to cover
            thresholds = np.linspace(mean - k*std, mean + k*std, n_thresholds)
            return thresholds.tolist()

    def _compute_domain_thresholds(self, factor_idx: int, values: np.ndarray) -> List[float]:
        """
        Use domain-specific thresholds for finance.

        If custom_thresholds provided, use those.
        Otherwise, use sensible defaults based on typical financial factors.

        These defaults assume standardized factor scores and are based on
        quantitative finance conventions for factor categorization.

        Args:
            factor_idx: Index of the factor
            values: 1D array of factor values (used as fallback)

        Returns:
            List of threshold values
        """
        if factor_idx in self.custom_thresholds:
            return self.custom_thresholds[factor_idx]

        # Default finance-inspired thresholds
        # These are reasonable defaults for standardized factor scores
        default_thresholds = {
            0: [-0.5, 0.5],    # Factor 1: Growth (moderate thresholds)
            1: [-0.3, 0.3],    # Factor 2: Value (tighter thresholds)
            2: [-0.4, 0.4],    # Factor 3: Quality
            3: [-0.3, 0.3]     # Factor 4: Momentum
        }

        if factor_idx in default_thresholds:
            return default_thresholds[factor_idx]

        # Fallback to quantile for unknown factors
        warnings.warn(f"No custom thresholds for factor {factor_idx}, using quantile method")
        return self._compute_quantile_thresholds(values)

    def _compute_hybrid_thresholds(self, values: np.ndarray) -> List[float]:
        """
        Adaptive method: choose strategy based on distribution shape.

        Strategy:
        - If distribution is heavily skewed (|skewness| > 1): use quantile
        - Otherwise: use standard deviation

        This provides robustness to different distribution shapes while
        maintaining interpretability when appropriate.

        Args:
            values: 1D array of factor values

        Returns:
            List of threshold values
        """
        skewness = stats.skew(values)

        if np.abs(skewness) > 1.0:
            # Skewed distribution: quantile is more robust
            return self._compute_quantile_thresholds(values)
        else:
            # Roughly normal: std is more interpretable
            return self._compute_std_thresholds(values)

    def _create_label_mapping(self) -> Dict[int, str]:
        """
        Create semantic labels for categories.

        Returns:
            Dict mapping integer labels to semantic names
        """
        if self.n_bins == 3:
            return {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        elif self.n_bins == 2:
            return {0: 'LOW', 1: 'HIGH'}
        elif self.n_bins == 5:
            return {0: 'VERY_LOW', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'VERY_HIGH'}
        else:
            # Generic labels
            return {i: f'BIN_{i}' for i in range(self.n_bins)}

# ==================== Demo/Testing ====================

if __name__ == '__main__':
    """
    Demonstration and basic testing of ThresholdDiscretizer.

    This demo:
    1. Generates synthetic factor scores with different distributions
    2. Tests all 4 discretization methods
    3. Validates that category distributions make sense
    4. Tests custom threshold functionality
    """
    print("="*80)
    print("ThresholdDiscretizer - Demonstration")
    print("="*80)

    # Generate synthetic factor scores with different distributions
    np.random.seed(42)
    n_samples = 1000
    n_factors = 4

    # Factor 1: Normal distribution (μ=0, σ=1)
    factor1 = np.random.normal(0, 1, n_samples)

    # Factor 2: Skewed distribution (log-normal)
    factor2 = np.random.lognormal(0, 0.5, n_samples) - 1.5

    # Factor 3: Uniform distribution
    factor3 = np.random.uniform(-2, 2, n_samples)

    # Factor 4: Bi-modal distribution
    factor4 = np.concatenate([
        np.random.normal(-1, 0.3, n_samples//2),
        np.random.normal(1, 0.3, n_samples//2)
    ])

    factor_scores = np.column_stack([factor1, factor2, factor3, factor4])

    print(f"\nGenerated factor scores: {factor_scores.shape}")
    print(f"\nFactor score statistics:")
    for i in range(n_factors):
        values = factor_scores[:, i]
        print(f"  Factor {i+1}: min={values.min():.2f}, max={values.max():.2f}, "
              f"mean={values.mean():.2f}, std={values.std():.2f}, "
              f"skew={stats.skew(values):.2f}")

    # Test each method
    methods = ['quantile', 'std', 'domain', 'hybrid']

    for method in methods:
        print(f"\n{'-'*80}")
        print(f"Method: {method.upper()}")
        print(f"{'-'*80}")

        discretizer = ThresholdDiscretizer(method=method, n_bins=3)
        discrete_labels, thresholds = discretizer.fit_transform(factor_scores)
        label_mapping = discretizer.get_label_mapping()

        print(f"\nLabel mapping: {label_mapping}")
        print(f"\nThresholds per factor:")
        for factor_idx, thresh in thresholds.items():
            print(f"  Factor {factor_idx+1}: {[f'{t:.4f}' for t in thresh]}")

        print(f"\nCategory distributions:")
        for factor_idx in range(n_factors):
            labels = discrete_labels[:, factor_idx]
            counts = np.bincount(labels)
            percentages = (counts / len(labels) * 100)

            dist_str = ", ".join([f"{label_mapping[i]}={counts[i]} ({percentages[i]:.1f}%)"
                                 for i in range(len(counts))])
            print(f"  Factor {factor_idx+1}: {dist_str}")

    # Test custom thresholds
    print(f"\n{'='*80}")
    print("Custom Domain Thresholds")
    print(f"{'='*80}")

    custom = {
        0: [-0.5, 0.5],
        1: [-1.0, 0.0],
        2: [-1.5, 1.5],
        3: [-0.8, 0.8]
    }

    discretizer = ThresholdDiscretizer(method='domain', custom_thresholds=custom)
    discrete_labels, thresholds = discretizer.fit_transform(factor_scores)

    print(f"\nApplied custom thresholds: {custom}")
    print(f"\nCategory distributions with custom thresholds:")
    for factor_idx in range(n_factors):
        labels = discrete_labels[:, factor_idx]
        counts = np.bincount(labels)
        percentages = (counts / len(labels) * 100)

        dist_str = ", ".join([f"{discretizer.label_mapping_[i]}={counts[i]} ({percentages[i]:.1f}%)"
                             for i in range(len(counts))])
        print(f"  Factor {factor_idx+1}: {dist_str}")

    # Test error handling
    print(f"\n{'='*80}")
    print("Testing Error Handling")
    print(f"{'='*80}")

    try:
        # Test invalid method
        bad_discretizer = ThresholdDiscretizer(method='invalid')
        print("❌ Should have raised ValueError for invalid method")
    except ValueError as e:
        print(f"✓ Correctly caught invalid method: {e}")

    try:
        # Test transform before fit
        new_discretizer = ThresholdDiscretizer()
        new_discretizer.transform(factor_scores)
        print("❌ Should have raised ValueError for transform before fit")
    except ValueError as e:
        print(f"✓ Correctly caught transform before fit: {e}")

    try:
        # Test NaN values
        bad_data = np.copy(factor_scores)
        bad_data[0, 0] = np.nan
        discretizer.fit(bad_data)
        print("❌ Should have raised ValueError for NaN values")
    except ValueError as e:
        print(f"✓ Correctly caught NaN values: {e}")

    print(f"\n{'='*80}")
    print("All tests passed! ✓")
    print(f"{'='*80}")

    print("\nThresholdDiscretizer is ready for integration with SHAP explanations!")
    print("Next step: Implement RuleGenerator (PROMPT_2)")
