"""
XAI for Dimensionality Reduction - Main Runner
Comprehensive test suite for all XAI methods.

Author: Bernardo Guterres
Date: December 2025

Usage:
    python run_all_experiments.py --method all
    python run_all_experiments.py --method pca
    python run_all_experiments.py --method cluster_shapley
    python run_all_experiments.py --method sparse_ae
    python run_all_experiments.py --method tcvae
    python run_all_experiments.py --method probing
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loaders import (
    load_fama_french_factors,
    load_stock_fundamentals, 
    load_lending_club_data,
    prepare_for_dr,
    create_factor_labels,
    _generate_synthetic_stock_fundamentals,
    _generate_synthetic_credit_data
)

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def run_dr_comparison(X, feature_names, factor_membership=None):
    """Run the main DR comparison experiment."""
    print_header("DIMENSIONALITY REDUCTION COMPARISON")
    
    from xai_dr_comparison import (
        DRExplainer,
        run_comparison,
        visualize_comparison,
        factor_recovery_score
    )
    
    # Create default factor membership if not provided
    if factor_membership is None:
        n_features = len(feature_names)
        quarter = n_features // 4
        factor_membership = {
            'Factor1': feature_names[:quarter],
            'Factor2': feature_names[quarter:2*quarter],
            'Factor3': feature_names[2*quarter:3*quarter],
            'Factor4': feature_names[3*quarter:]
        }
    
    print(f"\nData shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Run comparison
    results_df, explainers = run_comparison(X, feature_names, factor_membership)
    
    # Save results
    results_df.to_csv('outputs/dr_comparison_results.csv', index=False)
    visualize_comparison(results_df, explainers, X, 
                        save_path='outputs/dr_comparison.png')
    
    print("\nResults saved to outputs/dr_comparison_results.csv")
    
    return results_df, explainers

def run_cluster_shapley(X, feature_names):
    """Run ClusterShapley analysis."""
    print_header("CLUSTERSHAPLEY ANALYSIS")
    
    from cluster_shapley import (
        ClusterShapley,
        compare_dr_methods_clustering,
        stability_analysis
    )
    
    print(f"\nData shape: {X.shape}")
    
    # t-SNE + ClusterShapley
    print("\nRunning t-SNE + ClusterShapley...")
    cs = ClusterShapley(dr_method='tsne', cluster_method='kmeans', n_clusters=5)
    cs.fit(X, feature_names)
    
    # Get global importance
    global_imp = cs.get_global_feature_importance(top_n=10)
    print("\nGlobal feature importance:")
    print(global_imp.to_string(index=False))
    
    # Generate rules
    rules = cs.generate_cluster_rules()
    print("\nCluster rules:")
    for cid, rule in rules.items():
        print(f"  {rule}")
    
    # Visualize
    cs.visualize(save_path='outputs/cluster_shapley.png')
    
    # Stability analysis
    print("\nRunning stability analysis...")
    stability = stability_analysis(X, feature_names, n_runs=3, dr_method='tsne')
    print(f"Mean rank correlation: {stability['mean_rank_correlation']:.3f}")
    
    return cs, global_imp

def run_sparse_autoencoder(X, feature_names):
    """Run Sparse Autoencoder analysis."""
    print_header("SPARSE AUTOENCODER ANALYSIS")
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sparse_autoencoder import (
        SparseAutoencoder,
        SparseAutoencoderTrainer,
        FeatureInterpreter,
        compare_with_pca
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Data shape: {X.shape}")
    
    # Prepare data
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Create and train model
    print("\nTraining Sparse Autoencoder...")
    model = SparseAutoencoder(
        input_dim=X.shape[1],
        hidden_dim=64,
        sparsity_k=8
    )
    
    trainer = SparseAutoencoderTrainer(model, learning_rate=1e-3, device=device)
    train_losses, _ = trainer.train(train_loader, n_epochs=100, verbose=True)
    
    # Interpret features
    print("\nInterpreting learned features...")
    interpreter = FeatureInterpreter(model, feature_names)
    dictionary = interpreter.get_feature_dictionary(min_weight=0.15)
    
    print(f"\nDiscovered {len(dictionary)} interpretable features:")
    for unit_idx, interp in list(dictionary.items())[:8]:
        print(f"  Unit {unit_idx}: {interp['interpretation']}")
    
    # Compare with PCA
    comparison, pca_loadings = compare_with_pca(X, model, feature_names)
    print("\nComparison with PCA:")
    for k, v in comparison.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Visualize
    interpreter.visualize_features(n_units=15, 
                                   save_path='outputs/sparse_ae_features.png')
    
    # Save model
    torch.save(model.state_dict(), 'outputs/sparse_ae_model.pt')
    
    return model, interpreter

def run_beta_tcvae(X, feature_names, true_factors=None):
    """Run β-TCVAE analysis."""
    print_header("β-TCVAE DISENTANGLEMENT ANALYSIS")
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from beta_tcvae import (
        BetaTCVAE,
        VAETrainer,
        DisentanglementMetrics,
        interpret_latent_dims
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Data shape: {X.shape}")
    
    # Prepare data
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train β-TCVAE
    print("\nTraining β-TCVAE...")
    model = BetaTCVAE(
        input_dim=X.shape[1],
        hidden_dims=[64, 32],
        latent_dim=8,
        beta=6.0
    )
    
    trainer = VAETrainer(model, learning_rate=1e-3, device=device)
    losses = trainer.train(train_loader, n_epochs=150, verbose=True)
    
    # Compute disentanglement metrics
    if true_factors:
        print("\nComputing disentanglement metrics...")
        metrics = DisentanglementMetrics()
        
        mig = metrics.compute_mig(model, X, true_factors)
        dci = metrics.compute_dci(model, X, true_factors)
        
        print(f"  MIG: {mig:.4f}")
        print(f"  DCI-Disentanglement: {dci['disentanglement']:.4f}")
        print(f"  DCI-Completeness: {dci['completeness']:.4f}")
        print(f"  DCI-Informativeness: {dci['informativeness']:.4f}")
    
    # Interpret latent dimensions
    print("\nInterpreting latent dimensions...")
    corr_df, interpretations = interpret_latent_dims(model, X, feature_names)
    
    for dim, interp in interpretations.items():
        print(f"  {interp}")
    
    # Save
    corr_df.to_csv('outputs/tcvae_latent_correlations.csv')
    torch.save(model.state_dict(), 'outputs/beta_tcvae_model.pt')
    
    return model, corr_df

def run_probing_analysis(X, feature_names, labels_dict, true_factors=None):
    """Run probing classifier analysis."""
    print_header("PROBING CLASSIFIER ANALYSIS")
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from probing_classifiers import (
        EmbeddingProber,
        ComparativeProbingAnalysis
    )
    
    print(f"\nData shape: {X.shape}")
    
    # Create embeddings
    print("\nCreating embeddings...")
    
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)
    print(f"  PCA: {X_pca.shape}")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f"  t-SNE: {X_tsne.shape}")
    
    # Comparative analysis
    print("\nRunning comparative probing...")
    comparison = ComparativeProbingAnalysis(X, feature_names)
    comparison.add_dr_method('PCA', X_pca)
    comparison.add_dr_method('t-SNE', X_tsne)
    comparison.add_dr_method('Original', X)
    
    results_df = comparison.run_comparison(labels_dict, true_factors)
    
    # Display results
    print("\nProbing Results:")
    print(results_df.to_string(index=False))
    
    # Visualize
    comparison.visualize_comparison(save_path='outputs/probing_comparison.png')
    
    # Save results
    results_df.to_csv('outputs/probing_results.csv', index=False)
    
    return results_df

def run_all_experiments():
    """Run all experiments with synthetic data."""
    print_header("RUNNING ALL XAI EXPERIMENTS")
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Generate synthetic data
    print("\nGenerating synthetic financial data...")
    from sklearn.preprocessing import StandardScaler
    
    # Use stock fundamentals
    df = _generate_synthetic_stock_fundamentals(n_stocks=1500)
    
    # Prepare data
    X, feature_names, metadata = prepare_for_dr(
        df,
        exclude_cols=['Ticker', 'Sector'],
        handle_categorical='drop'
    )
    
    print(f"Prepared data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Create labels for probing
    # CIRCULARITY WARNING: These labels are derived from the same input features
    # (e.g., Quality_Tier from ROE, Value_Tier from PE_Ratio) that exist in X.
    # PCA embeddings contain information about every input feature, so a linear
    # probe on PCA can trivially predict ROE-derived labels — this measures
    # reconstruction ability, not whether the embedding encodes meaningful
    # semantic properties. High probing accuracy here is expected by construction
    # and does not generalise to real-world latent concept detection.
    labels_dict = create_factor_labels(df)
    print(f"Created labels: {list(labels_dict.keys())}")
    
    # Define factor membership for evaluation
    # WARNING — SPURIOUS GROUND TRUTH: This factor membership is constructed by
    # string-matching feature names, NOT from a known generating process. The
    # _generate_synthetic_stock_fundamentals() data has NO latent factor structure
    # (features are generated independently). Factor recovery scores computed
    # against this membership are statistically meaningless — any correlation
    # found is coincidental or driven by feature name co-occurrence with similar
    # distributions. For valid factor recovery evaluation, use
    # generate_financial_data_with_factors() from xai_dr_comparison.py instead,
    # which has a genuine latent factor model (though with its own SNR caveats).
    factor_membership = {
        'Valuation': [f for f in feature_names if any(v in f for v in ['PE', 'PB', 'PS', 'EV', 'Dividend'])],
        'Profitability': [f for f in feature_names if any(v in f for v in ['ROE', 'ROA', 'Margin'])],
        'Growth': [f for f in feature_names if 'Growth' in f],
        'Risk': [f for f in feature_names if any(v in f for v in ['Volatility', 'Beta', 'Return'])]
    }
    
    # Track timing
    results = {'experiment': [], 'time_seconds': [], 'status': []}
    
    # 1. DR Comparison
    try:
        start = time.time()
        run_dr_comparison(X, feature_names, factor_membership)
        results['experiment'].append('DR Comparison')
        results['time_seconds'].append(time.time() - start)
        results['status'].append('Success')
    except Exception as e:
        print(f"DR Comparison failed: {e}")
        results['experiment'].append('DR Comparison')
        results['time_seconds'].append(0)
        results['status'].append(f'Failed: {e}')
    
    # 2. ClusterShapley
    try:
        start = time.time()
        run_cluster_shapley(X, feature_names)
        results['experiment'].append('ClusterShapley')
        results['time_seconds'].append(time.time() - start)
        results['status'].append('Success')
    except Exception as e:
        print(f"ClusterShapley failed: {e}")
        results['experiment'].append('ClusterShapley')
        results['time_seconds'].append(0)
        results['status'].append(f'Failed: {e}')
    
    # 3. Sparse Autoencoder
    try:
        start = time.time()
        run_sparse_autoencoder(X, feature_names)
        results['experiment'].append('Sparse AE')
        results['time_seconds'].append(time.time() - start)
        results['status'].append('Success')
    except Exception as e:
        print(f"Sparse AE failed: {e}")
        results['experiment'].append('Sparse AE')
        results['time_seconds'].append(0)
        results['status'].append(f'Failed: {e}')
    
    # 4. β-TCVAE
    try:
        start = time.time()
        run_beta_tcvae(X, feature_names)
        results['experiment'].append('β-TCVAE')
        results['time_seconds'].append(time.time() - start)
        results['status'].append('Success')
    except Exception as e:
        print(f"β-TCVAE failed: {e}")
        results['experiment'].append('β-TCVAE')
        results['time_seconds'].append(0)
        results['status'].append(f'Failed: {e}')
    
    # 5. Probing Classifiers
    try:
        start = time.time()
        # Filter to only valid labels
        valid_labels = {k: v for k, v in labels_dict.items() if v is not None and len(v) == len(X)}
        run_probing_analysis(X, feature_names, valid_labels)
        results['experiment'].append('Probing')
        results['time_seconds'].append(time.time() - start)
        results['status'].append('Success')
    except Exception as e:
        print(f"Probing failed: {e}")
        results['experiment'].append('Probing')
        results['time_seconds'].append(0)
        results['status'].append(f'Failed: {e}')
    
    # Summary
    print_header("EXPERIMENT SUMMARY")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    total_time = results_df['time_seconds'].sum()
    print(f"\nTotal time: {total_time:.1f} seconds")
    
    # List outputs
    print("\nOutput files in outputs/:")
    for f in Path('outputs').glob('*'):
        print(f"  • {f.name}")
    
    return results_df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XAI for Dimensionality Reduction Experiments')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'dr_comparison', 'cluster_shapley', 
                               'sparse_ae', 'tcvae', 'probing'],
                       help='Which experiment to run')
    parser.add_argument('--data', type=str, default='synthetic',
                       choices=['synthetic', 'stocks', 'credit'],
                       help='Which dataset to use')
    
    args = parser.parse_args()
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Load data based on argument
    print_header("LOADING DATA")
    
    if args.data == 'synthetic':
        df = _generate_synthetic_stock_fundamentals(n_stocks=1500)
    elif args.data == 'stocks':
        df = load_stock_fundamentals()
    elif args.data == 'credit':
        df = _generate_synthetic_credit_data(n_samples=3000)
    
    # Prepare data
    X, feature_names, metadata = prepare_for_dr(
        df,
        exclude_cols=['Ticker', 'Sector', 'Industry', 'purpose', 'home_ownership', 'grade'],
        handle_categorical='drop'
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Create labels
    labels_dict = create_factor_labels(df)
    
    # Run specified experiment
    if args.method == 'all':
        run_all_experiments()
    elif args.method == 'dr_comparison':
        run_dr_comparison(X, feature_names)
    elif args.method == 'cluster_shapley':
        run_cluster_shapley(X, feature_names)
    elif args.method == 'sparse_ae':
        run_sparse_autoencoder(X, feature_names)
    elif args.method == 'tcvae':
        run_beta_tcvae(X, feature_names)
    elif args.method == 'probing':
        valid_labels = {k: v for k, v in labels_dict.items() if v is not None}
        run_probing_analysis(X, feature_names, valid_labels)
    
    print("\n✓ Experiment complete!")
