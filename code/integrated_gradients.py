"""
Integrated Gradients for Autoencoder Explanations
Attribution method for explaining autoencoder reconstructions and latent representations.

Author: Bernardo Guterres
Date: December 2025

Integrated Gradients satisfy the Completeness axiom - attributions
sum to the difference between output at input and output at baseline. This makes
them theoretically well-grounded for explaining neural network outputs.

- Sundararajan et al. (2017): "Axiomatic Attribution for Deep Networks"
- Antwarg et al. (2021): "Explaining Anomalies Detected by Autoencoders Using SHAP"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# SECTION 1: Standard Autoencoder for Attribution
# ============================================================================

class Autoencoder(nn.Module):
    """Standard autoencoder for reconstruction-based anomaly detection."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], latent_dim=8):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def reconstruction_error(self, x):
        """Per-sample reconstruction error."""
        x_recon, _ = self.forward(x)
        return torch.mean((x - x_recon) ** 2, dim=1)

# ============================================================================
# SECTION 2: Integrated Gradients Implementation
# ============================================================================

class IntegratedGradients:
    """
    Integrated Gradients for explaining autoencoder outputs.
    
    IG computes the integral of gradients along a path from baseline to input:
    IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x-x'))/∂x_i) dα
    
    Where:
    - x is the input
    - x' is the baseline (typically zeros or mean)
    - F is the model output
    - α is the interpolation parameter
    """
    
    def __init__(self, model, baseline_type='zeros'):
        """
        Args:
            model: Autoencoder model
            baseline_type: 'zeros', 'mean', or a tensor
        """
        self.model = model
        self.baseline_type = baseline_type
        self.model.eval()
    
    def _get_baseline(self, x):
        """Get baseline for attribution."""
        if isinstance(self.baseline_type, torch.Tensor):
            return self.baseline_type
        elif self.baseline_type == 'zeros':
            return torch.zeros_like(x)
        elif self.baseline_type == 'mean':
            return x.mean(dim=0, keepdim=True).expand_as(x)
        else:
            return torch.zeros_like(x)
    
    def attribute_reconstruction(self, x, n_steps=50, internal_batch_size=None):
        """
        Compute IG attributions for reconstruction error.
        
        Explains: Which input features contribute most to reconstruction error?
        
        Args:
            x: Input tensor (batch_size, input_dim)
            n_steps: Number of integration steps
            internal_batch_size: Batch size for integration (memory management)
        
        Returns:
            attributions: (batch_size, input_dim) tensor
        """
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)
        baseline = self._get_baseline(x)
        
        # Generate interpolation steps
        alphas = torch.linspace(0, 1, n_steps + 1, device=x.device)
        
        # Compute gradients at each step
        integrated_gradients = torch.zeros_like(x)
        
        for alpha in alphas[:-1]:
            # Interpolated input
            x_interp = baseline + alpha * (x - baseline)
            x_interp = x_interp.requires_grad_(True)
            
            # Forward pass
            x_recon, _ = self.model(x_interp)
            
            # Compute reconstruction error
            recon_error = torch.mean((x_interp - x_recon) ** 2, dim=1).sum()
            
            # Backward pass
            recon_error.backward()
            
            # Accumulate gradients
            integrated_gradients += x_interp.grad / n_steps
            
            # Clear gradients
            x_interp.grad.zero_()
        
        # Scale by (x - baseline)
        attributions = (x - baseline) * integrated_gradients
        
        return attributions.detach()
    
    def attribute_latent(self, x, latent_dim_idx, n_steps=50):
        """
        Compute IG attributions for a specific latent dimension.
        
        Explains: Which input features contribute to latent dimension z_i?
        
        Args:
            x: Input tensor
            latent_dim_idx: Which latent dimension to explain
            n_steps: Number of integration steps
        
        Returns:
            attributions: (batch_size, input_dim) tensor
        """
        self.model.eval()
        baseline = self._get_baseline(x)
        
        alphas = torch.linspace(0, 1, n_steps + 1, device=x.device)
        integrated_gradients = torch.zeros_like(x)
        
        for alpha in alphas[:-1]:
            x_interp = baseline + alpha * (x - baseline)
            x_interp = x_interp.clone().requires_grad_(True)
            
            # Get latent representation
            z = self.model.encode(x_interp)
            
            # Take specific dimension
            target = z[:, latent_dim_idx].sum()
            
            # Backward
            target.backward()
            integrated_gradients += x_interp.grad / n_steps
            x_interp.grad.zero_()
        
        attributions = (x - baseline) * integrated_gradients
        
        return attributions.detach()
    
    def attribute_anomaly(self, x, threshold=None, n_steps=50):
        """
        Explain why a sample is considered anomalous.
        
        Uses reconstruction error as anomaly score.
        Higher attribution = more contribution to anomaly.
        """
        attributions = self.attribute_reconstruction(x, n_steps)
        
        # For anomalies, we want features that INCREASE reconstruction error
        # These are features that are hard to reconstruct
        
        # Also compute reconstruction error
        with torch.no_grad():
            recon_error = self.model.reconstruction_error(x)
        
        # If threshold provided, flag anomalies
        if threshold is not None:
            is_anomaly = recon_error > threshold
        else:
            is_anomaly = None
        
        return {
            'attributions': attributions,
            'reconstruction_error': recon_error,
            'is_anomaly': is_anomaly
        }

# ============================================================================
# SECTION 3: Attribution Analysis Tools
# ============================================================================

class AutoencoderExplainer:
    """
    Comprehensive explanation framework for autoencoders.
    """
    
    def __init__(self, model, feature_names=None, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_names = feature_names or [f'F{i}' for i in range(model.input_dim)]
        self.ig = IntegratedGradients(model, baseline_type='zeros')
    
    def explain_reconstruction(self, x, n_steps=50):
        """
        Explain which features contribute to reconstruction error.
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        attrs = self.ig.attribute_reconstruction(x, n_steps)
        
        return pd.DataFrame(
            attrs.cpu().numpy(),
            columns=self.feature_names
        )
    
    def explain_latent_space(self, x, n_steps=50):
        """
        Explain which features contribute to each latent dimension.
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        latent_dim = self.model.latent_dim
        all_attrs = []
        
        for dim in range(latent_dim):
            attrs = self.ig.attribute_latent(x, dim, n_steps)
            all_attrs.append(attrs.mean(dim=0).cpu().numpy())
        
        return pd.DataFrame(
            np.array(all_attrs).T,
            index=self.feature_names,
            columns=[f'Z{i}' for i in range(latent_dim)]
        )
    
    def explain_anomaly(self, x, normal_samples, percentile=95, n_steps=50):
        """
        Explain why a sample is anomalous compared to normal samples.
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(normal_samples, np.ndarray):
            normal_samples = torch.FloatTensor(normal_samples).to(self.device)
        
        # Compute threshold from normal samples
        with torch.no_grad():
            normal_errors = self.model.reconstruction_error(normal_samples)
            threshold = np.percentile(normal_errors.cpu().numpy(), percentile)
        
        # Get attribution
        result = self.ig.attribute_anomaly(x, threshold=threshold, n_steps=n_steps)
        
        return {
            'attributions': pd.DataFrame(
                result['attributions'].cpu().numpy(),
                columns=self.feature_names
            ),
            'reconstruction_error': result['reconstruction_error'].cpu().numpy(),
            'threshold': threshold,
            'is_anomaly': result['is_anomaly'].cpu().numpy() if result['is_anomaly'] is not None else None
        }
    
    def get_global_feature_importance(self, X, n_samples=500, n_steps=30):
        """
        Compute global feature importance across many samples.
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        # Sample subset
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X = X[idx]
        
        attrs = self.ig.attribute_reconstruction(X, n_steps)
        
        # Mean absolute attribution
        mean_abs = np.abs(attrs.cpu().numpy()).mean(axis=0)
        
        return pd.Series(mean_abs, index=self.feature_names).sort_values(ascending=False)
    
    def visualize_explanation(self, x, save_path='ae_explanation.png'):
        """
        Create comprehensive visualization for a single sample.
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x).to(self.device)
        else:
            x_tensor = x
        
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Original vs Reconstructed
        ax1 = axes[0, 0]
        with torch.no_grad():
            x_recon, z = self.model(x_tensor)
            x_np = x_tensor.cpu().numpy().flatten()
            x_recon_np = x_recon.cpu().numpy().flatten()
        
        x_pos = np.arange(len(self.feature_names))
        width = 0.35
        ax1.bar(x_pos - width/2, x_np, width, label='Original', alpha=0.8)
        ax1.bar(x_pos + width/2, x_recon_np, width, label='Reconstructed', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_title('Original vs Reconstructed Features')
        ax1.grid(alpha=0.3, axis='y')
        
        # Plot 2: Reconstruction error by feature
        ax2 = axes[0, 1]
        feature_errors = (x_np - x_recon_np) ** 2
        colors = plt.cm.Reds(feature_errors / feature_errors.max())
        ax2.bar(x_pos, feature_errors, color=colors)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax2.set_title('Per-Feature Reconstruction Error')
        ax2.set_ylabel('Squared Error')
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: IG attributions for reconstruction
        ax3 = axes[1, 0]
        attrs = self.explain_reconstruction(x_tensor).values.flatten()
        colors = ['red' if a > 0 else 'blue' for a in attrs]
        ax3.bar(x_pos, attrs, color=colors, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax3.set_title('Integrated Gradients Attribution')
        ax3.set_ylabel('Attribution Score')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.grid(alpha=0.3, axis='y')
        
        # Plot 4: Latent space attributions
        ax4 = axes[1, 1]
        latent_attrs = self.explain_latent_space(x_tensor)
        sns.heatmap(latent_attrs.T, cmap='RdBu_r', center=0, 
                   annot=True, fmt='.2f', ax=ax4)
        ax4.set_title('Feature Attribution to Latent Dimensions')
        ax4.set_xlabel('Input Feature')
        ax4.set_ylabel('Latent Dimension')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()
    
    def validate_completeness(self, x, n_steps=100):
        """
        Validate that IG satisfies the completeness axiom.
        
        Sum of attributions should equal: F(x) - F(baseline)
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        baseline = torch.zeros_like(x)
        
        # Compute attributions
        attrs = self.ig.attribute_reconstruction(x, n_steps)
        attr_sum = attrs.sum(dim=1)
        
        # Compute actual difference
        with torch.no_grad():
            error_x = self.model.reconstruction_error(x)
            error_baseline = self.model.reconstruction_error(baseline)
            actual_diff = error_x - error_baseline
        
        completeness_error = torch.abs(attr_sum - actual_diff)
        
        return {
            'attribution_sum': attr_sum.cpu().numpy(),
            'actual_difference': actual_diff.cpu().numpy(),
            'completeness_error': completeness_error.cpu().numpy(),
            'relative_error': (completeness_error / torch.abs(actual_diff)).cpu().numpy()
        }

# ============================================================================
# SECTION 4: Training and Example Usage
# ============================================================================

def train_autoencoder(X, hidden_dims=[64, 32, 16], latent_dim=8, n_epochs=100):
    """Train autoencoder on data."""
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = Autoencoder(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_x, in loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            x_recon, z = model(batch_x)
            loss = F.mse_loss(x_recon, batch_x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(loader):.6f}")
    
    return model

def generate_synthetic_data(n_samples=3000, n_features=15, anomaly_ratio=0.05):
    """Generate synthetic data with anomalies."""
    np.random.seed(42)
    
    # Normal data: multivariate normal
    mean = np.zeros(n_features)
    cov = np.eye(n_features) * 0.5 + np.ones((n_features, n_features)) * 0.5
    
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal
    
    X_normal = np.random.multivariate_normal(mean, cov, n_normal)
    
    # Anomalies: outliers in random features
    X_anomaly = np.random.multivariate_normal(mean, cov, n_anomaly)
    for i in range(n_anomaly):
        # Modify 2-4 random features
        n_modified = np.random.randint(2, 5)
        modified_features = np.random.choice(n_features, n_modified, replace=False)
        X_anomaly[i, modified_features] += np.random.choice([-1, 1], n_modified) * np.random.uniform(3, 5, n_modified)
    
    X = np.vstack([X_normal, X_anomaly])
    labels = np.array([0] * n_normal + [1] * n_anomaly)
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    labels = labels[shuffle_idx]
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    return X, labels, feature_names

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INTEGRATED GRADIENTS FOR AUTOENCODER EXPLANATIONS")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating synthetic data with anomalies...")
    X, labels, feature_names = generate_synthetic_data(n_samples=3000, n_features=15)
    
    print(f"Data shape: {X.shape}")
    print(f"Anomaly ratio: {labels.mean():.2%}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split normal for training
    X_normal = X_scaled[labels == 0]
    X_anomaly = X_scaled[labels == 1]
    
    print(f"Normal samples: {len(X_normal)}")
    print(f"Anomaly samples: {len(X_anomaly)}")
    
    # Train autoencoder
    print("\n2. Training autoencoder...")
    model = train_autoencoder(X_normal, hidden_dims=[32, 16], latent_dim=6, n_epochs=100)
    
    # Create explainer
    print("\n3. Creating explainer...")
    explainer = AutoencoderExplainer(model, feature_names, device=device)
    
    # Global feature importance
    print("\n4. Computing global feature importance...")
    global_importance = explainer.get_global_feature_importance(X_scaled)
    print("\nTop 10 features by importance:")
    print(global_importance.head(10))
    
    # Explain a normal sample
    print("\n5. Explaining a normal sample...")
    normal_idx = 0
    normal_attrs = explainer.explain_reconstruction(X_normal[normal_idx:normal_idx+1])
    print(f"Normal sample attribution:")
    print(normal_attrs.T.sort_values(by=0, ascending=False).head())
    
    # Explain an anomaly
    print("\n6. Explaining an anomalous sample...")
    anomaly_result = explainer.explain_anomaly(
        X_anomaly[0:1], 
        X_normal,
        percentile=95
    )
    
    print(f"Reconstruction error: {anomaly_result['reconstruction_error'][0]:.4f}")
    print(f"Threshold (95th percentile): {anomaly_result['threshold']:.4f}")
    print(f"Is anomaly: {anomaly_result['is_anomaly'][0]}")
    print("\nTop features contributing to anomaly:")
    top_attrs = anomaly_result['attributions'].T.abs().sort_values(by=0, ascending=False)
    print(top_attrs.head())
    
    # Validate completeness
    print("\n7. Validating completeness axiom...")
    completeness = explainer.validate_completeness(X_scaled[:10], n_steps=100)
    print(f"Mean relative error: {completeness['relative_error'].mean():.4f}")
    print(f"(Should be close to 0 if completeness holds)")
    
    # Latent space interpretation
    print("\n8. Interpreting latent space...")
    latent_attrs = explainer.explain_latent_space(X_scaled[:100])
    print("\nMean attribution to latent dimensions:")
    print(latent_attrs.abs().mean().sort_values(ascending=False))
    
    # Visualize
    print("\n9. Creating visualizations...")
    
    # Visualize anomaly explanation
    explainer.visualize_explanation(X_anomaly[0], 
                                    save_path='/home/claude/ig_anomaly_explanation.png')
    
    # Create summary visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Global importance
    ax1 = axes[0]
    global_importance.head(10).plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Global Feature Importance')
    ax1.set_xlabel('Mean |IG Attribution|')
    ax1.invert_yaxis()
    
    # Normal vs Anomaly reconstruction error
    ax2 = axes[1]
    with torch.no_grad():
        normal_errors = model.reconstruction_error(torch.FloatTensor(X_normal).to(device)).cpu().numpy()
        anomaly_errors = model.reconstruction_error(torch.FloatTensor(X_anomaly).to(device)).cpu().numpy()
    
    ax2.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
    ax2.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)
    ax2.axvline(anomaly_result['threshold'], color='red', linestyle='--', label='Threshold')
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Density')
    ax2.set_title('Reconstruction Error Distribution')
    ax2.legend()
    
    # Latent attribution heatmap
    ax3 = axes[2]
    sns.heatmap(latent_attrs.T, cmap='RdBu_r', center=0, ax=ax3)
    ax3.set_title('Feature → Latent Attribution')
    ax3.set_xlabel('Input Feature')
    ax3.set_ylabel('Latent Dim')
    
    plt.tight_layout()
    plt.savefig('/home/claude/ig_analysis_summary.png', dpi=150, bbox_inches='tight')
    print("Summary saved to: ig_analysis_summary.png")
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), '/home/claude/autoencoder_model.pt')
    
    print("\n" + "=" * 70)
    print("INTEGRATED GRADIENTS ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\nKey findings:")
    print(f"  • Top feature: {global_importance.index[0]} (importance: {global_importance.iloc[0]:.4f})")
    print(f"  • Anomaly detection accuracy: {(anomaly_errors > anomaly_result['threshold']).mean():.2%}")
    print(f"  • Completeness error: {completeness['relative_error'].mean():.4f}")
    
    print("\nOutput files:")
    print("  • ig_anomaly_explanation.png - Individual anomaly explanation")
    print("  • ig_analysis_summary.png - Summary visualization")
    print("  • autoencoder_model.pt - Trained model")
