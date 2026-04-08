# Complete Reproduction Guide

This guide provides step-by-step instructions to reproduce every table, figure, and result reported in the thesis.

## Prerequisites

1. Python 3.8+ installed
2. All dependencies installed: `pip install -r requirements.txt`
3. Approximately 2 GB free disk space for outputs
4. Estimated total runtime: ~30 minutes on a modern CPU

## Complete Reproduction Workflow

### Step 1: Verify Installation

```bash
cd tests/
pytest test_threshold_discretizer.py -v
```

**Expected output:** All 42 tests pass (100% pass rate)

---

## Chapter 5: Evaluation

### Table 5.1 — DR Method Comparison (6 methods × 5 metrics)

**Thesis reference:** Chapter 5, Section 5.2  
**Reproduction command:**

```bash
cd code/
python run_all_experiments.py --method all
```

**Output file:** `outputs/dr_comparison_results.csv`

**Expected results:**
| Method | Surrogate R² | Factor Recovery | Stability | MoRF AUC | Top Features |
|--------|-------------|-----------------|-----------|----------|--------------|
| PCA | 0.865 | 0.912 | 1.000 | 0.876 | Revenue_Growth, Debt_to_Equity |
| NMF | 0.958 | 0.945 | 1.000 | 0.923 | Profit_Margin, Asset_Turnover |
| ICA | 0.891 | 0.867 | 1.000 | 0.889 | Volatility_1Y, Beta |
| FA | 0.887 | 0.898 | 1.000 | 0.891 | ROE, Operating_Margin |
| t-SNE | 0.586 | 0.423 | -0.090 | 0.612 | Price_to_Book, Dividend_Yield |
| UMAP | 0.625 | 0.501 | 0.193 | 0.645 | Market_Cap, Volume |

**Validation:** Compare CSV output against thesis Table 5.1

---

### Table 5.2 — Regime Comparison (4 noise regimes)

**Thesis reference:** Chapter 5, Section 5.6  
**Reproduction command:**

```bash
cd code/
python run_regime_comparison.py
```

**Output file:** `outputs/regime_comparison_results.csv`

**Expected results:**
| Regime | SNR Range | PCA R² | NMF R² | ICA R² | FA R² | t-SNE R² | UMAP R² |
|--------|-----------|--------|--------|--------|-------|----------|---------|
| Idealised | 3.0–5.0 | 0.865 | 0.977 | 0.891 | 0.887 | 0.586 | 0.625 |
| Moderate | 1.0–1.5 | 0.843 | 0.965 | 0.867 | 0.871 | 0.554 | 0.598 |
| Realistic | 0.3–0.7 | 0.821 | 0.932 | 0.843 | 0.856 | 0.523 | 0.571 |
| Noisy | ~0.02 | -3.142 | 0.977 | -1.234 | -0.892 | 0.489 | 0.543 |

**Key finding:** NMF maintains high R² (0.977) even in noisy regime; PCA/ICA/FA collapse (negative R²)

---

### Table 5.3 — Threshold Strategy Comparison

**Thesis reference:** Chapter 5, Section 5.3  
**Reproduction command:**

```bash
cd code/
python -c "
from threshold_discretizer import ThresholdDiscretizer
import numpy as np

np.random.seed(42)
scores = np.random.randn(500, 4)

disc = ThresholdDiscretizer()

# Test robustness to outliers (5% contamination)
strategies = ['quantile', 'std', 'domain', 'hybrid']
for strategy in strategies:
    thresholds_clean = disc.fit_transform(scores, strategy=strategy)
    
    # Inject 5% outliers
    scores_outlier = scores.copy()
    n_outliers = int(0.05 * len(scores))
    outlier_idx = np.random.choice(len(scores), n_outliers, replace=False)
    scores_outlier[outlier_idx] += np.random.uniform(3, 5, (n_outliers, 4))
    
    thresholds_outlier = disc.fit_transform(scores_outlier, strategy=strategy)
    
    # Compute threshold shift
    shift = np.abs(thresholds_outlier - thresholds_clean).mean()
    print(f'{strategy}: threshold_shift = {shift:.3f}')
"
```

**Expected results:**
- Quantile: threshold_shift = 0.03
- Std: threshold_shift = 1.20
- Domain: threshold_shift = 0.00 (fixed thresholds)
- Hybrid: threshold_shift = 0.41

**Conclusion:** Quantile discretization is 40× more robust than standard-deviation method

---

### Table 5.4 — SHAP Completeness Check

**Thesis reference:** Chapter 5, Section 5.4.1  
**Reproduction command:**

```bash
cd code/
python case_study_credit_risk.py
```

**Output file:** `outputs/case_study/model_card.json` (contains completeness statistics)

**Expected results:**
- Total checks: 2,000 (500 samples × 4 factors)
- Passes (ε = 0.01): 1,523 (76.2%)
- Mean error: 0.095
- Max error: 0.821

**Validation:** Open `model_card.json` and check the `shap_completeness` field

---

### Section 5.4 — Credit Risk Case Study

**Thesis reference:** Chapter 5, Section 5.4  
**Reproduction command:**

```bash
cd code/
python case_study_credit_risk.py
```

**Output files:**
- `outputs/case_study/model_card.json` — ECB Guide compliance
- `outputs/case_study/rules_credit_risk.json` — SR 11-7 validation report
- `outputs/case_study/individual_explanations.json` — GDPR Article 22 explanations (5,000 applicants)
- `outputs/case_study/shap_importance.png` — Feature importance visualization
- `outputs/case_study/factor_distributions.png` — Factor score distributions

**Key metrics to verify:**
```bash
cat outputs/case_study/model_card.json | grep -A 5 "surrogate_r2"
```

Expected output:
```json
"surrogate_r2": {
  "PC1": 0.823,
  "PC2": 0.891,
  "PC3": 0.754,
  "PC4": 0.777
}
```

**Validation:** Debt_Burden (PC4) surrogate R² = 0.777 < 0.85 threshold (honestly reported as validation failure)

---

### Figure 5.1 — DR Method Comparison Visualization

**Reproduction command:**

```bash
cd code/
python run_all_experiments.py --method all
```

**Output file:** `outputs/dr_comparison.png`

**Contents:**
- 6 subplots (one per DR method)
- Scatter plots showing first 2 latent dimensions
- Color-coded by true factor membership
- Surrogate R² annotated on each subplot

---

### Figure 5.2 — Regime Robustness

**Reproduction command:**

```bash
cd code/
python run_regime_comparison.py
# Then generate visualization:
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/regime_comparison_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))
for method in ['PCA', 'NMF', 'ICA', 'FA', 't-SNE', 'UMAP']:
    subset = df[df['method'] == method]
    ax.plot(subset['regime'], subset['surrogate_r2'], marker='o', label=method)

ax.set_xlabel('Noise Regime')
ax.set_ylabel('Surrogate R²')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('outputs/regime_robustness.png', dpi=300, bbox_inches='tight')
print('Saved to outputs/regime_robustness.png')
"
```

**Output file:** `outputs/regime_robustness.png`

---

### Table 5.5 — Probing Classifier Accuracy

**Thesis reference:** Chapter 5, Section 5.5  
**Reproduction command:**

```bash
cd code/
python run_all_experiments.py --method probing
```

**Expected output (printed to console):**
```
Probing Classifier Results
==========================
Linear Probe: 0.847
MLP Probe: 0.891
SVM Probe: 0.823
Random Forest Probe: 0.905
```

**Validation:** All probing accuracies should exceed majority-class baseline (~0.34-0.38 for balanced 3-class)

---

## Appendix B: All 36 Generated Rules

**Thesis reference:** Appendix B  
**Reproduction command:**

```bash
cd code/
python run_all_experiments.py --method all
```

**Output files:**
- `outputs/all_rules.csv` — Machine-readable format (36 rows)
- `outputs/rules_quantile.json` — Quantile strategy rules
- `outputs/rules_std.json` — Standard deviation strategy rules
- `outputs/rules_domain.json` — Domain expert strategy rules
- `outputs/rules_human_readable.txt` — Plain English rules

**Verification:**

```bash
wc -l outputs/all_rules.csv
```

Expected: 37 lines (1 header + 36 rules)

```bash
head -20 outputs/rules_human_readable.txt
```

Expected format:
```
IF PC1 is LOW (-6.36 <= score < -0.41, n=167)
THEN primary drivers are:
  - Earnings_Growth (SHAP: -0.177)
  - Debt_to_Equity (SHAP: -0.159)
  - Volatility_1Y (SHAP: -0.145)
  ...
```

---

## Appendix C: Experimental Outputs

**Thesis reference:** Appendix C  
**Files already generated by previous commands:**

- `outputs/discretized_scores.csv` — Discretized factor scores (500 samples × 4 factors)
- `outputs/viz_dr_comparison.png` — DR method comparison visualization
- `outputs/viz_discretization.png` — Discretization strategy comparison
- `outputs/viz_factor_distributions.png` — Factor score distributions
- `outputs/viz_correlation_heatmap.png` — Feature correlation heatmap

**Verification:**

```bash
ls -lh outputs/*.png
```

All PNG files should be present (5 visualizations, ~200-2000 KB each)

---

## Complete Reproduction Checklist

Run all experiments in sequence:

```bash
# 1. Unit tests (verify installation)
cd tests/
pytest test_threshold_discretizer.py -v

# 2. Main DR comparison (Table 5.1)
cd ../code/
python run_all_experiments.py --method all

# 3. Regime comparison (Table 5.2)
python run_regime_comparison.py

# 4. Case study (Section 5.4, SHAP completeness)
python case_study_credit_risk.py

# 5. Verify all outputs
cd ../outputs/
ls -lh *.csv *.png case_study/*
```

**Total runtime:** ~20-30 minutes on a modern CPU (Intel i7/M1 or equivalent)

---

## Verification Against Thesis

After running all experiments, verify key results:

### Critical Numbers to Check

| Metric | Thesis Value | Output File | Location |
|--------|-------------|-------------|----------|
| Case study surrogate R² (PC4) | 0.777 | `case_study/model_card.json` | `surrogate_r2.PC4` |
| SHAP completeness pass rate | 76.2% | `case_study/model_card.json` | `shap_completeness.pass_rate` |
| SHAP completeness max error | 0.821 | `case_study/model_card.json` | `shap_completeness.max_error` |
| Total rules generated | 36 | `all_rules.csv` | Row count - 1 (header) |
| NMF R² (idealised regime) | 0.958 | `dr_comparison_results.csv` | Row for NMF |
| NMF R² (noisy regime) | 0.977 | `regime_comparison_results.csv` | Row for NMF, Noisy |
| PCA R² (noisy regime) | -3.142 | `regime_comparison_results.csv` | Row for PCA, Noisy |
| Quantile threshold shift | 0.03 | Console output | From threshold comparison script |

### Quick Verification Script

```bash
cd outputs/

# Check case study key metrics
echo "=== CASE STUDY METRICS ==="
cat case_study/model_card.json | grep -E "surrogate_r2|shap_completeness" -A 2

# Check DR comparison
echo "=== DR COMPARISON ==="
cat dr_comparison_results.csv | grep -E "Method|NMF|PCA"

# Check regime comparison
echo "=== REGIME COMPARISON ==="
cat regime_comparison_results.csv | grep -E "Regime|Noisy"

# Check rule count
echo "=== RULE COUNT ==="
wc -l all_rules.csv
```

---

## Troubleshooting

### Missing Dependencies

If you get `ModuleNotFoundError`, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Numerical Differences

Small numerical differences (±0.01) are expected due to:
- Random seed variations across NumPy/PyTorch versions
- Floating-point precision differences
- XGBoost version differences

**Acceptable tolerance:** ±2% for R² values, ±5% for SHAP values

### Long Runtime

To speed up experiments:
- Reduce sample size in `run_all_experiments.py` (line 170: `n_samples = 500` → `200`)
- Reduce XGBoost trees (line 95: `n_estimators=100` → `50`)

**Note:** This will change numerical results slightly but preserve qualitative conclusions

### Out of Memory

If you encounter memory errors:
- Close other applications
- Reduce batch size for neural networks (PyTorch models)
- Run experiments individually rather than `--method all`

---

## Contact

If you encounter issues reproducing results, please:

1. Check Python version: `python --version` (must be ≥3.8)
2. Verify dependencies: `pip list | grep -E "numpy|pandas|sklearn|shap|xgboost"`
3. Open a GitHub issue with error output

---

## Reproducibility Statement

All experiments reported in the thesis were run with:
- **Python:** 3.10.8
- **NumPy:** 1.23.5
- **scikit-learn:** 1.2.0
- **XGBoost:** 1.7.3
- **SHAP:** 0.41.0
- **PyTorch:** 1.13.1
- **Random seed:** 42 (fixed throughout)

Results are deterministic and should reproduce exactly on the same environment.
