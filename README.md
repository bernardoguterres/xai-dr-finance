# Explainable AI for Dimensionality Reduction in Finance

**BSc Individual Project, King's College London**  
**Author:** Bernardo Guterres  
**Supervisor:** Dr. David Watson  
**Date:** April 2026

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of a novel framework for converting continuous dimensionality reduction outputs into interpretable, discrete IF-THEN rules suitable for financial decision-making and regulatory compliance.

**Core Contribution:** A threshold discretization framework (Algorithm 1) that bridges the "discrete logic gap" between continuous XAI outputs (SHAP values, factor loadings) and categorical business rules required by financial practitioners and regulators.

**Key Results:**
- 36 interpretable rules generated across 4 PCA factors and 3 discretization strategies
- Case study surrogate R² = 0.777 on synthetic credit risk data
- Quantile discretization achieves 40× better threshold stability than standard-deviation methods
- Validated across 4 noise regimes (SNR: 0.02 to 5.0)

## Repository Structure

```
.
├── code/                           # Source code (17 modules, 8,863 lines)
│   ├── threshold_discretizer.py    # Four-strategy discretization framework
│   ├── rule_generator.py           # Algorithm 1 implementation
│   ├── xai_dr_comparison.py        # Unified DR comparison (6 methods)
│   ├── cluster_shapley.py          # ClusterShapley for t-SNE/UMAP
│   ├── sparse_autoencoder.py       # TopK/L1/Gated sparse autoencoders
│   ├── beta_tcvae.py               # β-TCVAE with disentanglement metrics
│   ├── probing_classifiers.py      # Probing framework (4 probe types)
│   ├── integrated_gradients.py     # IG attribution for autoencoders
│   ├── semi_synthetic_generator.py # 4-regime calibrated data generator
│   ├── data_loaders.py             # Dataset loaders (Fama-French, Yahoo, Lending Club)
│   ├── case_study_credit_risk.py   # Regulatory compliance demonstration
│   ├── run_all_experiments.py      # Main experiment orchestrator
│   └── run_regime_comparison.py    # Noise regime robustness evaluation
│
├── tests/                          # Unit test suite
│   └── test_threshold_discretizer.py  # 42 tests (100% pass rate)
│
├── outputs/                        # Experimental results
│   ├── dr_comparison_results.csv   # Table 5.1 (6 DR methods × 5 metrics)
│   ├── regime_comparison_results.csv  # Table 5.2 (4 noise regimes)
│   ├── all_rules.csv               # All 36 generated rules
│   ├── discretization_results.csv  # Threshold strategy comparison
│   └── case_study/                 # Credit risk case study outputs
│       ├── model_card.json         # ECB-compliant model card
│       ├── rules_credit_risk.json  # SR 11-7 validation report
│       └── individual_explanations.json  # GDPR Article 22 explanations
│
├── thesis/                         # LaTeX thesis (120 pages)
│   └── thesis.pdf                  # Submitted thesis
│
├── README.md                       # This file
├── REPRODUCTION_GUIDE.md           # Step-by-step reproduction instructions
├── requirements.txt                # Python dependencies
└── LICENSE                         # MIT License
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/bernardoguterres/xai-dr-finance.git
cd xai-dr-finance

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Core Experiments

```bash
cd code/

# Full DR comparison (reproduces Table 5.1)
python run_all_experiments.py --method all

# Regime comparison (reproduces Table 5.2)
python run_regime_comparison.py

# Credit risk case study (reproduces Section 5.4)
python case_study_credit_risk.py
```

### 3. Run Unit Tests

```bash
cd tests/
pytest test_threshold_discretizer.py -v
```

## Reproducing Thesis Results

**For complete step-by-step instructions to reproduce every table and figure in the thesis, see [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md).**

Quick reference:

| Thesis Element | Reproduction Command | Output File |
|----------------|----------------------|-------------|
| **Table 5.1** (DR Comparison) | `python run_all_experiments.py --method all` | `outputs/dr_comparison_results.csv` |
| **Table 5.2** (Regime Comparison) | `python run_regime_comparison.py` | `outputs/regime_comparison_results.csv` |
| **Table 5.3** (Threshold Strategies) | `python threshold_discretizer.py` | `outputs/discretization_results.csv` |
| **Section 5.4** (Case Study) | `python case_study_credit_risk.py` | `outputs/case_study/*` |
| **All 36 Rules** | Automatically generated during experiments | `outputs/all_rules.csv` |

## Key Dependencies

- **Python**: ≥ 3.8
- **Core ML**: scikit-learn ≥ 1.0, XGBoost ≥ 1.5, PyTorch ≥ 1.10
- **XAI**: SHAP ≥ 0.40
- **DR**: umap-learn ≥ 0.5, hdbscan ≥ 0.8
- **Data**: numpy ≥ 1.21, pandas ≥ 1.3, scipy ≥ 1.7

Full dependency list in `requirements.txt`.

## Framework Architecture

The framework implements a four-layer architecture:

1. **Data Layer**: Synthetic data generation with 4 calibrated noise regimes
2. **DR Layer**: Unified interface for 6 DR methods (PCA, NMF, ICA, FA, t-SNE, UMAP)
3. **XAI Layer**: 5 explanation techniques (SHAP, ClusterShapley, IG, Sparse AE, β-TCVAE)
4. **Rule Layer**: Threshold discretization + rule generation (Algorithm 1)

### Algorithm 1: Threshold Discretization Framework

The core algorithm (implemented in `rule_generator.py`) operates in 4 steps:

1. **Discretize** factor scores into LOW/MEDIUM/HIGH categories using one of 4 strategies
2. **Attribute** category membership using SHAP values from XGBoost surrogates
3. **Rank** top-5 feature drivers per category by SHAP magnitude
4. **Generate** IF-THEN rules with threshold bounds and sample counts

**Novelty:** First application of systematic threshold discretization to SHAP-attributed DR outputs with regulatory compliance validation.

## Regulatory Compliance Outputs

The case study (`case_study_credit_risk.py`) generates concurrent compliance artifacts:

- **ECB Guide (2018)**: Model card with hyperparameters and variance explained
- **SR 11-7 (Federal Reserve)**: Validation report with surrogate R², MoRF curves, SHAP completeness
- **GDPR Article 22**: Individual explanations for 5,000 credit applicants with per-factor attributions

All outputs are JSON-formatted for machine readability and audit trails.

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@thesis{guterres2026xai,
  title={Explainable AI for Dimensionality Reduction in Finance: 
         Bridging Continuous Outputs and Discrete Business Rules},
  author={Guterres, Bernardo},
  year={2026},
  school={King's College London},
  type={BSc Individual Project}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

All dependencies are released under permissive open-source licenses:
- scikit-learn, PyTorch, umap-learn: BSD-3-Clause
- XGBoost: Apache-2.0
- SHAP: MIT

## Contact

**Bernardo Guterres**  
King's College London  
Email: bernardomloguterres@gmail.com

## Limitations and Future Work

**Known Limitations:**
- Evaluated on synthetic data; real financial data validation pending
- SHAP completeness violations (max error 0.821) due to surrogate approximation
- Static thresholds (no drift detection or recalibration)
- No human evaluation of rule interpretability

**Future Directions:**
- Validation on real Fama-French factor returns and Lending Club loan data
- Dynamic threshold recalibration with distributional drift detection
- Counterfactual explanations to complement SHAP attributions
- Extension to supervised contexts (e.g., SHAP-based credit scoring)

See Chapter 7 of the thesis for detailed discussion.
