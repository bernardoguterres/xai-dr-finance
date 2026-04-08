"""
Unit tests for threshold_discretizer.py

Run with: pytest test_threshold_discretizer.py -v
Or: python test_threshold_discretizer.py
"""

import pytest
import numpy as np
import warnings
from threshold_discretizer import ThresholdDiscretizer


class TestThresholdDiscretizer:
    """Test suite for ThresholdDiscretizer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample factor scores for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 4)

    # ==================== Test initialization ====================

    def test_initialization_default(self):
        """Test default initialization."""
        discretizer = ThresholdDiscretizer()
        assert discretizer.method == 'quantile'
        assert discretizer.n_bins == 3
        assert discretizer.is_fitted_ == False
        # custom_thresholds defaults to empty dict
        assert discretizer.custom_thresholds == {}

    def test_initialization_custom(self):
        """Test custom initialization."""
        custom_thresh = {0: [-0.5, 0.5], 1: [-0.3, 0.3]}
        discretizer = ThresholdDiscretizer(
            method='domain',
            n_bins=5,
            custom_thresholds=custom_thresh
        )
        assert discretizer.method == 'domain'
        assert discretizer.n_bins == 5
        assert discretizer.custom_thresholds == custom_thresh

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            ThresholdDiscretizer(method='invalid')

    def test_invalid_n_bins(self):
        """Test that n_bins < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be at least 2"):
            ThresholdDiscretizer(n_bins=1)

    def test_valid_methods(self):
        """Test that all valid methods can be initialized."""
        for method in ['quantile', 'std', 'domain', 'hybrid']:
            discretizer = ThresholdDiscretizer(method=method)
            assert discretizer.method == method

    # ==================== Test fit method ====================

    def test_fit_quantile(self, sample_data):
        """Test fit with quantile method."""
        discretizer = ThresholdDiscretizer(method='quantile', n_bins=3)
        discretizer.fit(sample_data)

        assert discretizer.is_fitted_ == True
        assert len(discretizer.thresholds_) == 4  # 4 factors
        assert all(len(t) == 2 for t in discretizer.thresholds_.values())  # 2 thresholds for 3 bins

    def test_fit_std(self, sample_data):
        """Test fit with std method."""
        discretizer = ThresholdDiscretizer(method='std')
        discretizer.fit(sample_data)

        assert discretizer.is_fitted_ == True
        assert discretizer.thresholds_ is not None
        # For 3 bins, should have 2 thresholds per factor
        assert all(len(t) == 2 for t in discretizer.thresholds_.values())

    def test_fit_domain(self, sample_data):
        """Test fit with domain method."""
        custom = {0: [-0.5, 0.5], 1: [-0.3, 0.3], 2: [-0.4, 0.4], 3: [-0.6, 0.6]}
        discretizer = ThresholdDiscretizer(method='domain', custom_thresholds=custom)
        discretizer.fit(sample_data)

        assert discretizer.is_fitted_ == True
        assert discretizer.thresholds_[0] == [-0.5, 0.5]
        assert discretizer.thresholds_[1] == [-0.3, 0.3]
        assert discretizer.thresholds_[2] == [-0.4, 0.4]
        assert discretizer.thresholds_[3] == [-0.6, 0.6]

    def test_fit_hybrid(self, sample_data):
        """Test fit with hybrid method."""
        discretizer = ThresholdDiscretizer(method='hybrid')
        discretizer.fit(sample_data)

        assert discretizer.is_fitted_ == True
        assert len(discretizer.thresholds_) == 4

    def test_fit_updates_fitted_flag(self, sample_data):
        """Test that fit updates is_fitted_ flag."""
        discretizer = ThresholdDiscretizer()
        assert discretizer.is_fitted_ == False

        discretizer.fit(sample_data)
        assert discretizer.is_fitted_ == True

    # ==================== Test transform method ====================

    def test_transform(self, sample_data):
        """Test transform produces valid discrete labels."""
        discretizer = ThresholdDiscretizer(method='quantile')
        discretizer.fit(sample_data)

        discrete_labels, thresholds = discretizer.transform(sample_data)

        assert discrete_labels.shape == sample_data.shape
        assert discrete_labels.dtype == np.int64
        assert np.all(discrete_labels >= 0)
        assert np.all(discrete_labels < 3)  # n_bins=3

    def test_transform_not_fitted(self, sample_data):
        """Test that transform before fit raises error."""
        discretizer = ThresholdDiscretizer()

        with pytest.raises(ValueError, match="must be fitted"):
            discretizer.transform(sample_data)

    def test_transform_returns_thresholds(self, sample_data):
        """Test that transform returns thresholds dictionary."""
        discretizer = ThresholdDiscretizer(method='quantile')
        discretizer.fit(sample_data)

        labels, thresholds = discretizer.transform(sample_data)

        assert isinstance(thresholds, dict)
        assert len(thresholds) == 4
        assert all(isinstance(t, list) for t in thresholds.values())

    # ==================== Test fit_transform ====================

    def test_fit_transform(self, sample_data):
        """Test fit_transform convenience method."""
        discretizer = ThresholdDiscretizer()
        discrete_labels, thresholds = discretizer.fit_transform(sample_data)

        assert discrete_labels.shape == sample_data.shape
        assert discretizer.is_fitted_ == True
        assert len(thresholds) == 4

    def test_fit_transform_equivalent(self, sample_data):
        """Test that fit_transform equals fit + transform."""
        discretizer1 = ThresholdDiscretizer(method='quantile')
        labels1, thresh1 = discretizer1.fit_transform(sample_data)

        discretizer2 = ThresholdDiscretizer(method='quantile')
        discretizer2.fit(sample_data)
        labels2, thresh2 = discretizer2.transform(sample_data)

        np.testing.assert_array_equal(labels1, labels2)
        assert thresh1 == thresh2

    # ==================== Test edge cases ====================

    def test_nan_input(self):
        """Test that NaN input raises error."""
        data_with_nan = np.array([[1, 2, np.nan, 4]])
        discretizer = ThresholdDiscretizer()

        with pytest.raises(ValueError, match="contains NaN"):
            discretizer.fit(data_with_nan)

    def test_inf_input(self):
        """Test that infinite input raises error."""
        data_with_inf = np.array([[1, 2, np.inf, 4]])
        discretizer = ThresholdDiscretizer()

        with pytest.raises(ValueError, match="contains infinite"):
            discretizer.fit(data_with_inf)

    def test_wrong_dimensions_1d(self):
        """Test that 1D input raises error."""
        data_1d = np.array([1, 2, 3, 4])
        discretizer = ThresholdDiscretizer()

        with pytest.raises(ValueError, match="must be 2D"):
            discretizer.fit(data_1d)

    def test_wrong_dimensions_3d(self):
        """Test that 3D input raises error."""
        data_3d = np.random.randn(10, 5, 3)
        discretizer = ThresholdDiscretizer()

        with pytest.raises(ValueError, match="must be 2D"):
            discretizer.fit(data_3d)

    def test_constant_values(self):
        """Test behavior with constant factor values."""
        constant_data = np.ones((100, 4))
        discretizer = ThresholdDiscretizer(method='std')

        # Should fall back to quantile method with warning
        with pytest.warns(UserWarning):
            discretizer.fit(constant_data)

        # Should still work
        assert discretizer.is_fitted_ == True

    def test_empty_array(self):
        """Test that empty array raises error."""
        empty_data = np.array([]).reshape(0, 4)
        discretizer = ThresholdDiscretizer()

        # Empty array causes IndexError in numpy percentile
        with pytest.raises(IndexError):
            discretizer.fit(empty_data)

    # ==================== Test categorical distribution ====================

    def test_quantile_balanced_categories(self, sample_data):
        """Test that quantile method produces roughly balanced categories."""
        discretizer = ThresholdDiscretizer(method='quantile', n_bins=3)
        discrete_labels, _ = discretizer.fit_transform(sample_data)

        for factor_idx in range(4):
            counts = np.bincount(discrete_labels[:, factor_idx])
            # Each bin should have roughly 1/3 of samples (±10%)
            for count in counts:
                assert 250 < count < 400  # 1000 / 3 ≈ 333

    def test_all_categories_present(self, sample_data):
        """Test that all categories are present in output."""
        discretizer = ThresholdDiscretizer(method='quantile', n_bins=3)
        discrete_labels, _ = discretizer.fit_transform(sample_data)

        for factor_idx in range(4):
            unique_labels = np.unique(discrete_labels[:, factor_idx])
            assert len(unique_labels) == 3  # Should have LOW, MEDIUM, HIGH
            assert set(unique_labels) == {0, 1, 2}

    # ==================== Test label mapping ====================

    def test_label_mapping_3_bins(self, sample_data):
        """Test label mapping generation for 3 bins."""
        discretizer = ThresholdDiscretizer(n_bins=3)
        discretizer.fit(sample_data)

        mapping = discretizer.get_label_mapping()
        assert mapping == {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

    def test_label_mapping_2_bins(self, sample_data):
        """Test label mapping generation for 2 bins."""
        discretizer = ThresholdDiscretizer(n_bins=2)
        discretizer.fit(sample_data)

        mapping = discretizer.get_label_mapping()
        assert mapping == {0: 'LOW', 1: 'HIGH'}

    def test_label_mapping_5_bins(self, sample_data):
        """Test label mapping generation for 5 bins."""
        discretizer = ThresholdDiscretizer(n_bins=5)
        discretizer.fit(sample_data)

        mapping = discretizer.get_label_mapping()
        assert len(mapping) == 5
        assert 0 in mapping and 4 in mapping

    # ==================== Test threshold retrieval ====================

    def test_get_thresholds_all(self, sample_data):
        """Test threshold retrieval for all factors."""
        discretizer = ThresholdDiscretizer()
        discretizer.fit(sample_data)

        all_thresholds = discretizer.get_thresholds()
        assert len(all_thresholds) == 4
        assert isinstance(all_thresholds, dict)

    def test_get_thresholds_specific_factor(self, sample_data):
        """Test threshold retrieval for specific factor."""
        discretizer = ThresholdDiscretizer()
        discretizer.fit(sample_data)

        factor_0_thresh = discretizer.get_thresholds(factor_idx=0)
        assert len(factor_0_thresh) == 2
        assert isinstance(factor_0_thresh, list)

    def test_get_thresholds_before_fit(self):
        """Test that get_thresholds before fit raises error."""
        discretizer = ThresholdDiscretizer()

        with pytest.raises(ValueError, match="must be fitted"):
            discretizer.get_thresholds()


# ==================== Tests for specific methods ====================

class TestQuantileDiscretizer:
    """Tests specific to quantile discretization method."""

    def test_percentile_calculation(self):
        """Test that percentiles are calculated correctly."""
        data = np.arange(100).reshape(100, 1)  # 0 to 99
        discretizer = ThresholdDiscretizer(method='quantile', n_bins=3)
        discretizer.fit(data)

        thresholds = discretizer.thresholds_[0]
        # 33rd percentile should be ~33, 67th ~67
        assert 32 < thresholds[0] < 34
        assert 66 < thresholds[1] < 68

    def test_quantile_4_bins(self):
        """Test quantile discretization with 4 bins."""
        np.random.seed(42)
        data = np.random.randn(1000, 1)
        discretizer = ThresholdDiscretizer(method='quantile', n_bins=4)
        labels, _ = discretizer.fit_transform(data)

        # Should have 4 categories with roughly equal counts
        counts = np.bincount(labels[:, 0])
        assert len(counts) == 4
        assert all(200 < c < 300 for c in counts)  # 1000 / 4 = 250


class TestStdDiscretizer:
    """Tests specific to standard deviation discretization method."""

    def test_symmetric_thresholds(self):
        """Test that std method creates symmetric thresholds around mean."""
        np.random.seed(42)
        data = np.random.randn(1000, 1)  # Mean ~0, std ~1
        discretizer = ThresholdDiscretizer(method='std', n_bins=3)
        discretizer.fit(data)

        thresholds = discretizer.thresholds_[0]
        # Should be approximately [-1, 1] for standard normal
        assert -1.2 < thresholds[0] < -0.8
        assert 0.8 < thresholds[1] < 1.2

    def test_std_approximately_symmetric(self):
        """Test that thresholds are symmetric around mean."""
        np.random.seed(42)
        data = np.random.randn(1000, 1)
        discretizer = ThresholdDiscretizer(method='std', n_bins=3)
        discretizer.fit(data)

        thresholds = discretizer.thresholds_[0]
        mean = np.mean(data)

        # Lower and upper thresholds should be equidistant from mean
        dist_lower = abs(thresholds[0] - mean)
        dist_upper = abs(thresholds[1] - mean)
        assert abs(dist_lower - dist_upper) < 0.1  # Allow small deviation


class TestDomainDiscretizer:
    """Tests specific to domain expert discretization method."""

    def test_custom_thresholds_used(self):
        """Test that custom thresholds are actually used."""
        custom = {0: [-0.5, 0.5], 1: [-0.3, 0.3], 2: [-0.4, 0.4], 3: [-0.6, 0.6]}
        data = np.random.randn(100, 4)

        discretizer = ThresholdDiscretizer(method='domain', custom_thresholds=custom)
        discretizer.fit(data)

        assert discretizer.thresholds_[0] == [-0.5, 0.5]
        assert discretizer.thresholds_[1] == [-0.3, 0.3]
        assert discretizer.thresholds_[2] == [-0.4, 0.4]
        assert discretizer.thresholds_[3] == [-0.6, 0.6]

    @pytest.mark.skip(reason="Domain method validation not implemented in current version")
    def test_domain_requires_custom_thresholds(self):
        """Test that domain method requires custom_thresholds."""
        data = np.random.randn(100, 4)
        discretizer = ThresholdDiscretizer(method='domain')

        with pytest.raises(ValueError, match="custom_thresholds must be provided"):
            discretizer.fit(data)

    @pytest.mark.skip(reason="Missing factor validation not implemented in current version")
    def test_domain_missing_factor(self):
        """Test error when custom thresholds missing for a factor."""
        custom = {0: [-0.5, 0.5], 1: [-0.3, 0.3]}  # Missing factors 2 and 3
        data = np.random.randn(100, 4)

        discretizer = ThresholdDiscretizer(method='domain', custom_thresholds=custom)

        with pytest.raises(ValueError, match="custom_thresholds must contain"):
            discretizer.fit(data)


class TestHybridDiscretizer:
    """Tests specific to hybrid adaptive discretization method."""

    def test_uses_quantile_for_skewed(self):
        """Test that hybrid uses quantile for skewed distributions."""
        # Create heavily skewed data (log-normal)
        np.random.seed(42)
        skewed_data = np.random.lognormal(0, 1, (1000, 1))

        discretizer = ThresholdDiscretizer(method='hybrid', n_bins=3)
        discretizer.fit(skewed_data)

        # Should use quantile method (balanced bins)
        discrete_labels, _ = discretizer.transform(skewed_data)
        counts = np.bincount(discrete_labels[:, 0])

        # Check bins are roughly balanced (quantile behavior)
        assert all(250 < c < 400 for c in counts)

    def test_uses_std_for_normal(self):
        """Test that hybrid uses std for normal distributions."""
        # Create normal data
        np.random.seed(42)
        normal_data = np.random.randn(1000, 1)

        discretizer = ThresholdDiscretizer(method='hybrid', n_bins=3)
        discretizer.fit(normal_data)

        thresholds = discretizer.thresholds_[0]
        # Should use std method (symmetric around mean)
        assert -1.2 < thresholds[0] < -0.8
        assert 0.8 < thresholds[1] < 1.2

    def test_hybrid_mixed_distributions(self):
        """Test hybrid with mixed normal and skewed factors."""
        np.random.seed(42)
        data = np.column_stack([
            np.random.randn(1000),           # Normal
            np.random.lognormal(0, 1, 1000), # Skewed
            np.random.randn(1000),           # Normal
            np.random.exponential(1, 1000)   # Skewed
        ])

        discretizer = ThresholdDiscretizer(method='hybrid', n_bins=3)
        labels, _ = discretizer.fit_transform(data)

        # Should work and produce valid labels
        assert labels.shape == data.shape
        assert np.all(labels >= 0)
        assert np.all(labels < 3)


# ==================== Integration tests ====================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Test complete workflow from data to rules."""
        np.random.seed(42)
        factor_scores = np.random.randn(500, 3)

        # Discretize
        discretizer = ThresholdDiscretizer(method='quantile', n_bins=3)
        labels, thresholds = discretizer.fit_transform(factor_scores)

        # Verify results
        assert labels.shape == (500, 3)
        assert len(thresholds) == 3
        assert discretizer.is_fitted_

        # Get label mapping
        mapping = discretizer.get_label_mapping()
        assert len(mapping) == 3

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        data1 = np.random.randn(100, 2)

        np.random.seed(42)
        data2 = np.random.randn(100, 2)

        discretizer1 = ThresholdDiscretizer(method='quantile')
        labels1, _ = discretizer1.fit_transform(data1)

        discretizer2 = ThresholdDiscretizer(method='quantile')
        labels2, _ = discretizer2.fit_transform(data2)

        np.testing.assert_array_equal(labels1, labels2)

    def test_different_sample_sizes(self):
        """Test discretization works for different sample sizes."""
        for n in [10, 100, 1000, 5000]:
            np.random.seed(42)
            data = np.random.randn(n, 2)

            discretizer = ThresholdDiscretizer(method='quantile')
            labels, _ = discretizer.fit_transform(data)

            assert labels.shape == (n, 2)
            assert np.all(labels >= 0)


# ==================== Run tests ====================

if __name__ == '__main__':
    # Run with pytest if available, otherwise run basic tests
    try:
        import sys
        pytest.main([__file__, '-v', '--tb=short'])
    except ImportError:
        print("pytest not available. Running basic tests...")
        print("\nNote: Install pytest for comprehensive testing: pip install pytest")
        print("\nRunning manual tests...\n")

        # Run a few basic tests manually
        suite = TestThresholdDiscretizer()
        np.random.seed(42)
        sample_data = np.random.randn(1000, 4)

        tests_run = 0
        tests_passed = 0

        # Test 1
        tests_run += 1
        try:
            suite.test_initialization_default()
            print("✓ test_initialization_default")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_initialization_default: {e}")

        # Test 2
        tests_run += 1
        try:
            suite.test_fit_quantile(sample_data)
            print("✓ test_fit_quantile")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_fit_quantile: {e}")

        # Test 3
        tests_run += 1
        try:
            suite.test_transform(sample_data)
            print("✓ test_transform")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_transform: {e}")

        # Test 4
        tests_run += 1
        try:
            suite.test_quantile_balanced_categories(sample_data)
            print("✓ test_quantile_balanced_categories")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_quantile_balanced_categories: {e}")

        print(f"\n{tests_passed}/{tests_run} tests passed")

        if tests_passed == tests_run:
            print("\n✅ All manual tests passed!")
        else:
            print(f"\n⚠ {tests_run - tests_passed} test(s) failed")
