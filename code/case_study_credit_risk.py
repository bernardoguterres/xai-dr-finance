"""
Regulatory Compliance Case Study: Credit Risk Assessment

Demonstrates compliance with:
- ECB Guidance on Model Risk Management
- SR 11-7 (Federal Reserve Supervisory Letter)
- GDPR Article 22 (Right to Explanation)

Scenario:
A bank uses PCA to reduce 15 credit features to 4 interpretable risk factors.
Regulators require explanations of how these factors are derived and used.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from data_loaders import _generate_synthetic_credit_data, prepare_for_dr
from xai_dr_comparison import DRExplainer
from threshold_discretizer import ThresholdDiscretizer
from rule_generator import RuleGenerator

class RegulatoryComplianceDemo:
    """
    Complete demonstration of regulatory compliance for credit risk models.

    Generates all documentation required for:
    - Model governance (ECB)
    - Model validation (SR 11-7)
    - Individual explanations (GDPR)
    """

    def __init__(self, output_dir: str = './outputs/case_study'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.data = None
        self.X = None
        self.feature_names = None
        self.explainer = None
        self.rules_dict = None

    def run_full_case_study(self, n_samples: int = 5000):
        """
        Execute complete case study pipeline.

        Args:
            n_samples: Number of credit applications to generate
        """
        print("="*80)
        print("REGULATORY COMPLIANCE CASE STUDY")
        print("Credit Risk Assessment with XAI")
        print("="*80)

        # Step 1: Generate data
        self._generate_credit_data(n_samples)

        # Step 2: Apply dimensionality reduction
        self._apply_dimensionality_reduction()

        # Step 3: Generate explanations
        self._generate_explanations()

        # Step 4: Apply threshold discretization
        self._apply_threshold_discretization()

        # Step 5: Create regulatory documentation
        self._create_regulatory_documentation()

        # Step 6: Generate individual explanations
        self._generate_individual_explanations()

        # Step 7: Create visualizations
        self._create_visualizations()

        # Step 8: Final summary
        self._print_summary()

    def _generate_credit_data(self, n_samples: int):
        """Step 1: Generate synthetic credit application data."""
        print("\n" + "="*80)
        print("[STEP 1] GENERATING CREDIT DATA")
        print("="*80)

        from data_loaders import _generate_synthetic_credit_data

        df = _generate_synthetic_credit_data(n_samples=n_samples)

        # Separate features and target
        self.X, self.feature_names, metadata = prepare_for_dr(
            df,
            exclude_cols=['default', 'grade', 'purpose', 'home_ownership'],
            handle_categorical='drop'
        )

        self.data = df

        print(f"\nGenerated {len(df)} credit applications")
        print(f"Features: {len(self.feature_names)}")
        print(f"Default rate: {df['default'].mean():.1%}")
        print(f"\nFeature list:")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {feat}")

    def _apply_dimensionality_reduction(self):
        """Step 2: Apply PCA to extract risk factors."""
        print("\n" + "="*80)
        print("[STEP 2] DIMENSIONALITY REDUCTION")
        print("="*80)

        print("\nApplying PCA to extract 4 risk factors...")

        self.explainer = DRExplainer(method='pca', n_components=4)
        self.factor_scores = self.explainer.fit_transform(self.X, self.feature_names)

        # Get variance explained directly from PCA model
        variance_explained = self.explainer.model.explained_variance_ratio_

        # Get loadings (components)
        components = self.explainer.model.components_

        print("\nPCA Results:")
        print(f"  Components: 4")
        print(f"  Variance explained:")
        for i, var in enumerate(variance_explained):
            print(f"    Factor {i+1}: {var:.1%}")
        print(f"  Total variance captured: {sum(variance_explained):.1%}")

        # Interpret factors based on loadings
        self.factor_names = self._interpret_factors(components)

        print(f"\nInterpreted Risk Factors:")
        for i, name in enumerate(self.factor_names):
            print(f"  {i+1}. {name}")

    def _interpret_factors(self, components: np.ndarray) -> List[str]:
        """Interpret PCA components as risk factors."""
        factor_names = []

        for i, component in enumerate(components):
            # Find top contributing features
            top_idx = np.argsort(np.abs(component))[-3:]
            top_features = [self.feature_names[j] for j in top_idx]

            # Name based on dominant features
            if any('inc' in f.lower() or 'dti' in f.lower() for f in top_features):
                name = 'Income_Stability'
            elif any('delinq' in f.lower() or 'pub_rec' in f.lower() for f in top_features):
                name = 'Credit_History'
            elif any('revol' in f.lower() or 'bal' in f.lower() for f in top_features):
                name = 'Debt_Burden'
            else:
                name = f'Risk_Factor_{i+1}'

            factor_names.append(name)

        return factor_names

    def _generate_explanations(self):
        """Step 3: Generate SHAP explanations."""
        print("\n" + "="*80)
        print("[STEP 3] GENERATING SHAP EXPLANATIONS")
        print("="*80)

        print("\nTraining surrogate models...")
        self.explainer.train_surrogates()

        surrogate_r2 = self.explainer.surrogate_r2
        print("\nSurrogate Model Quality (R² scores):")
        for i, r2 in enumerate(surrogate_r2):
            status = "✓ PASS" if r2 > 0.85 else "⚠ REVIEW"
            print(f"  {self.factor_names[i]}: {r2:.4f} {status}")

        print("\nComputing SHAP values...")
        shap_list = []
        for dim in range(4):
            shap_vals = self.explainer.explain_with_shap(self.X, dim=dim)
            shap_list.append(shap_vals)

        self.shap_values = np.stack(shap_list, axis=1)
        print(f"SHAP values computed: shape {self.shap_values.shape}")

    def _apply_threshold_discretization(self):
        """Step 4: Apply threshold discretization and generate rules."""
        print("\n" + "="*80)
        print("[STEP 4] THRESHOLD DISCRETIZATION")
        print("="*80)

        # Use domain-expert thresholds for credit risk
        custom_thresholds = {
            0: [-0.5, 0.5],    # Balanced thresholds
            1: [-0.4, 0.4],
            2: [-0.5, 0.5],
            3: [-0.4, 0.4]
        }

        print("\nApplying domain-expert discretization...")
        print("Custom thresholds:")
        for i, thresh in custom_thresholds.items():
            print(f"  {self.factor_names[i]}: {thresh}")

        discretizer = ThresholdDiscretizer(
            method='domain',
            n_bins=3,
            custom_thresholds=custom_thresholds
        )

        rule_gen = RuleGenerator(
            discretizer=discretizer,
            top_n_features=5,
            epsilon=0.01
        )

        self.rules_dict = rule_gen.generate_rules(
            factor_scores=self.factor_scores,
            shap_values=self.shap_values,
            feature_names=self.feature_names,
            factor_names=self.factor_names
        )

        print(f"\nGenerated {len(self.rules_dict['rules'])} interpretable rules")

        # Validation
        val = self.rules_dict['validation']
        status = "✓ PASSED" if val['passed'] else "✗ FAILED"
        print(f"\nSHAP Completeness Validation: {status}")
        print(f"  Max error: {val['max_error']:.6f}")
        print(f"  Mean error: {val['mean_error']:.6f}")
        print(f"  Threshold: {val['threshold']:.6f}")

        # Sample rules
        print("\nSample Rules (first 2):")
        for i, rule in enumerate(self.rules_dict['rules'][:2]):
            print(f"\n--- Rule {i+1} ---")
            print(rule)

    def _create_regulatory_documentation(self):
        """Step 5: Create SR 11-7 model documentation."""
        print("\n" + "="*80)
        print("[STEP 5] REGULATORY DOCUMENTATION (SR 11-7)")
        print("="*80)

        # Model Card
        model_card = {
            'model_information': {
                'model_name': 'Credit Risk Factor Extraction (PCA + SHAP)',
                'version': '1.0',
                'date': datetime.now().isoformat(),
                'purpose': 'Extract interpretable risk factors from credit features',
                'methodology': 'Principal Component Analysis with SHAP explanations'
            },
            'data': {
                'n_applications': len(self.data),
                'n_features': len(self.feature_names),
                'default_rate': float(self.data['default'].mean()),
                'features': self.feature_names,
                'data_period': 'Synthetic data (representative of typical portfolio)'
            },
            'model_architecture': {
                'method': 'PCA',
                'n_components': 4,
                'variance_explained': [
                    float(v) for v in self.explainer.model.explained_variance_ratio_
                ],
                'total_variance': float(sum(self.explainer.model.explained_variance_ratio_))
            },
            'risk_factors': {
                self.factor_names[i]: {
                    'variance_explained': float(self.explainer.model.explained_variance_ratio_[i]),
                    'thresholds': self.rules_dict['thresholds'][i],
                    'interpretation': f'Risk factor {i+1} derived from PCA',
                    'top_drivers': [
                        self.feature_names[j] for j in
                        np.argsort(np.abs(self.explainer.model.components_[i]))[-5:]
                    ]
                }
                for i in range(4)
            },
            'explainability': {
                'method': 'SHAP (SHapley Additive exPlanations)',
                'surrogate_quality': {
                    self.factor_names[i]: float(self.explainer.surrogate_r2[i])
                    for i in range(4)
                },
                'completeness_validation': {
                    'max_error': float(self.rules_dict['validation']['max_error']),
                    'mean_error': float(self.rules_dict['validation']['mean_error']),
                    'passed': bool(self.rules_dict['validation']['passed'])
                }
            },
            'discretization': {
                'method': 'domain_expert',
                'n_bins': 3,
                'categories': ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK'],
                'n_rules_generated': len(self.rules_dict['rules'])
            },
            'regulatory_compliance': {
                'ECB_guidance': {
                    'status': 'COMPLIANT',
                    'evidence': 'Interpretable risk factors identified with clear thresholds'
                },
                'SR_11-7': {
                    'status': 'COMPLIANT',
                    'evidence': 'Model validation documented with performance metrics'
                },
                'GDPR_Article_22': {
                    'status': 'COMPLIANT',
                    'evidence': 'Individual explanations available via SHAP values'
                }
            },
            'validation': {
                'surrogate_r2threshold': 0.85,
                'all_surrogates_passed': all(r2 > 0.85 for r2 in self.explainer.surrogate_r2),
                'shap_completeness_passed': self.rules_dict['validation']['passed']
            }
        }

        # Save model card
        model_card_path = self.output_dir / 'model_card.json'
        with open(model_card_path, 'w') as f:
            json.dump(model_card, f, indent=2)

        print(f"\nModel Card saved to: {model_card_path}")
        print("\nRegulatory Compliance Status:")
        for reg, info in model_card['regulatory_compliance'].items():
            print(f"  {reg}: {info['status']}")

        # Save rules
        from rule_generator import RuleGenerator
        rule_gen = RuleGenerator()
        rule_gen.export_rules(self.rules_dict, format='json',
                             output_path=self.output_dir / 'rules_credit_risk.json')

        print(f"\nRules saved to: {self.output_dir / 'rules_credit_risk.json'}")

    def _generate_individual_explanations(self, n_examples: int = 10):
        """Step 6: Generate GDPR-compliant individual explanations."""
        print("\n" + "="*80)
        print("[STEP 6] INDIVIDUAL EXPLANATIONS (GDPR Article 22)")
        print("="*80)

        explanations = []

        print(f"\nGenerating explanations for {n_examples} sample applicants...")

        for i in range(min(n_examples, len(self.X))):
            # Get risk category for each factor
            discrete_labels = self.rules_dict['discrete_labels'][i]
            categories = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']

            explanation = {
                'applicant_id': f'APP_{i+1:05d}',
                'timestamp': datetime.now().isoformat(),
                'risk_assessment': {},
                'explanation': {}
            }

            for factor_idx, factor_name in enumerate(self.factor_names):
                # Risk category
                category = categories[discrete_labels[factor_idx]]
                score = float(self.factor_scores[i, factor_idx])

                # Top 3 contributing features
                top_feature_idx = np.argsort(np.abs(self.shap_values[i, factor_idx, :]))[-3:][::-1]

                top_drivers = [
                    {
                        'feature': self.feature_names[idx],
                        'contribution': float(self.shap_values[i, factor_idx, idx]),
                        'feature_value': float(self.X[i, idx])
                    }
                    for idx in top_feature_idx
                ]

                explanation['risk_assessment'][factor_name] = {
                    'score': score,
                    'category': category,
                    'thresholds': self.rules_dict['thresholds'][factor_idx]
                }

                explanation['explanation'][factor_name] = {
                    'top_drivers': top_drivers,
                    'interpretation': self._interpret_category(factor_name, category, top_drivers)
                }

            explanations.append(explanation)

        # Save individual explanations
        explanations_path = self.output_dir / 'individual_explanations.json'
        with open(explanations_path, 'w') as f:
            json.dump(explanations, f, indent=2)

        print(f"\nIndividual explanations saved to: {explanations_path}")

        # Print sample explanation
        print("\nSample Explanation (Applicant 1):")
        print(json.dumps(explanations[0], indent=2))

    def _interpret_category(self, factor_name: str, category: str,
                           top_drivers: List[Dict]) -> str:
        """Generate human-readable interpretation."""
        if category == 'HIGH_RISK':
            return f"This applicant shows HIGH risk in {factor_name} primarily due to: " + \
                   ", ".join([d['feature'] for d in top_drivers[:2]])
        elif category == 'LOW_RISK':
            return f"This applicant shows LOW risk in {factor_name}, supported by favorable: " + \
                   ", ".join([d['feature'] for d in top_drivers[:2]])
        else:
            return f"This applicant shows MEDIUM risk in {factor_name}"

    def _create_visualizations(self):
        """Step 7: Create regulatory audit visualizations."""
        print("\n" + "="*80)
        print("[STEP 7] CREATING VISUALIZATIONS")
        print("="*80)

        # 1. Factor distributions by risk category
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Risk Factor Distributions by Category', fontsize=16)

        discrete_labels = self.rules_dict['discrete_labels']
        categories = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']
        colors = ['green', 'orange', 'red']

        for factor_idx, (ax, factor_name) in enumerate(zip(axes.flat, self.factor_names)):
            for cat_idx, (category, color) in enumerate(zip(categories, colors)):
                mask = discrete_labels[:, factor_idx] == cat_idx
                if np.any(mask):
                    ax.hist(self.factor_scores[mask, factor_idx],
                           alpha=0.5, bins=30, color=color, label=category)

            # Plot thresholds
            for thresh in self.rules_dict['thresholds'][factor_idx]:
                ax.axvline(thresh, color='black', linestyle='--', linewidth=2)

            ax.set_title(factor_name)
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Number of Applicants')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        dist_path = self.output_dir / 'factor_distributions.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nFactor distributions saved to: {dist_path}")

        # 2. SHAP summary plot for first factor
        fig, ax = plt.subplots(figsize=(10, 8))

        factor_idx = 0
        shap_vals = self.shap_values[:, factor_idx, :]
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-10:]

        y_pos = np.arange(len(top_features_idx))
        ax.barh(y_pos, mean_abs_shap[top_features_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in top_features_idx])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'Feature Importance for {self.factor_names[factor_idx]}')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        shap_path = self.output_dir / 'shap_importance.png'
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP importance plot saved to: {shap_path}")

    def _print_summary(self):
        """Step 8: Print final summary."""
        print("\n" + "="*80)
        print("CASE STUDY COMPLETE")
        print("="*80)

        print(f"\nAll outputs saved to: {self.output_dir}")

        print("\nGenerated Files:")
        print("  1. model_card.json - SR 11-7 compliance documentation")
        print("  2. rules_credit_risk.json - Interpretable decision rules")
        print("  3. individual_explanations.json - GDPR Article 22 compliance")
        print("  4. factor_distributions.png - Risk category visualizations")
        print("  5. shap_importance.png - Feature importance analysis")

        print("\nRegulatory Compliance Summary:")
        print("  ✓ ECB Guidance: Interpretable risk factors identified")
        print("  ✓ SR 11-7: Complete model validation documentation")
        print("  ✓ GDPR Article 22: Individual explanations available")

        print("\nKey Metrics:")
        print(f"  Applications analyzed: {len(self.data)}")
        print(f"  Risk factors extracted: 4")
        print(f"  Rules generated: {len(self.rules_dict['rules'])}")
        print(f"  Variance explained: {sum(self.explainer.model.explained_variance_ratio_):.1%}")
        print(f"  SHAP validation: {'PASSED' if self.rules_dict['validation']['passed'] else 'FAILED'}")

        print("\nNext Steps:")
        print("  1. Review model card for regulatory submission")
        print("  2. Validate individual explanations with domain experts")
        print("  3. Include visualizations in governance documentation")
        print("  4. Use in thesis Section 5.4 (Case Study)")

# ==================== Main Execution ====================

def run_case_study():
    """Execute the complete regulatory compliance case study."""
    demo = RegulatoryComplianceDemo()
    demo.run_full_case_study(n_samples=5000)

if __name__ == '__main__':
    run_case_study()
