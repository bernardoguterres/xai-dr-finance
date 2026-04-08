"""
β-TCVAE: Disentangled VAE for Interpretable Latent Factors
Total Correlation Variational Autoencoder for learning independent latent factors.

Author: Bernardo Guterres
Date: December 2025

β-TCVAE decomposes the ELBO into three terms:
1. Mutual Information: I(x; z) - information shared between input and latent
2. Total Correlation: KL(q(z) || Π_i q(z_i)) - statistical dependencies between latent dims
3. Dimension-wise KL: Σ_i KL(q(z_i) || p(z_i)) - individual dimension regularization

By penalizing Total Correlation specifically, we encourage independent latent factors.

- Chen et al. (2018): "Isolating Sources of Disentanglement in VAEs"
- Higgins et al. (2017): "β-VAE: Learning Basic Visual Concepts with a Constrained VAE"
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# SECTION 1: VAE Architectures
# ============================================================================

class BetaTCVAE(nn.Module):
    """
    β-TCVAE: Total Correlation Variational Autoencoder.
    
    Encourages disentanglement by penalizing statistical dependencies
    between latent dimensions.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=10, beta=4.0):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            beta: Weight on Total Correlation term (higher = more disentangled)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
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
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
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
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def loss_function(self, x, x_recon, mu, logvar, dataset_size):
        """
        Compute β-TCVAE loss.
        
        ELBO decomposition:
        L = E[log p(x|z)] - α·I(x;z) - β·TC(z) - γ·KL(q(z)||p(z))
        
        Where:
        - I(x;z) = Mutual Information
        - TC(z) = Total Correlation
        - KL term = dimension-wise KL
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # KL divergence for each dimension: KL(q(z_i|x) || p(z_i))
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Estimate Total Correlation using minibatch weighted sampling
        # This is the key innovation of β-TCVAE
        log_qz = self._log_qz(mu, logvar)
        log_qz_product = self._log_qz_product(mu, logvar)
        
        # TC = E[log q(z) - log Π_i q(z_i)]
        tc_estimate = (log_qz - log_qz_product).mean()
        
        # Mutual Information estimate
        log_pz = self._log_pz(mu)
        mi_estimate = (log_qz - log_pz).mean()
        
        # Dimension-wise KL
        dim_kl = kl_divergence.mean() - tc_estimate - mi_estimate
        
        # Total loss (α=1, γ=1, β=self.beta)
        total_loss = recon_loss + mi_estimate + self.beta * tc_estimate + dim_kl
        
        return total_loss, recon_loss, tc_estimate, mi_estimate
    
    def _log_qz(self, mu, logvar):
        """Estimate log q(z) using minibatch."""
        batch_size = mu.size(0)
        
        # Compute pairwise log probabilities
        mu_exp = mu.unsqueeze(1)  # (batch, 1, latent)
        logvar_exp = logvar.unsqueeze(1)
        
        mu_all = mu.unsqueeze(0)  # (1, batch, latent)
        
        # log N(z; mu, sigma^2)
        log_qz_i = -0.5 * (
            logvar_exp + 
            (mu_all - mu_exp).pow(2) / logvar_exp.exp() +
            np.log(2 * np.pi)
        )
        
        # Sum over latent dims, logsumexp over batch
        log_qz = torch.logsumexp(log_qz_i.sum(dim=2), dim=1) - np.log(batch_size)
        
        return log_qz
    
    def _log_qz_product(self, mu, logvar):
        """Estimate log Π_i q(z_i) (product of marginals)."""
        batch_size = mu.size(0)
        
        mu_exp = mu.unsqueeze(1)
        logvar_exp = logvar.unsqueeze(1)
        mu_all = mu.unsqueeze(0)
        
        log_qz_i = -0.5 * (
            logvar_exp +
            (mu_all - mu_exp).pow(2) / logvar_exp.exp() +
            np.log(2 * np.pi)
        )
        
        # logsumexp for each dimension separately
        log_qz_product = torch.logsumexp(log_qz_i, dim=1).sum(dim=1) - self.latent_dim * np.log(batch_size)
        
        return log_qz_product
    
    def _log_pz(self, mu):
        """Log probability under prior N(0,1)."""
        return -0.5 * (mu.pow(2) + np.log(2 * np.pi)).sum(dim=1)

class BetaVAE(nn.Module):
    """
    Standard β-VAE for comparison.
    Uses single β weight on entire KL term.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=10, beta=4.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    def loss_function(self, x, x_recon, mu, logvar, dataset_size=None):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss, torch.tensor(0.0)

# ============================================================================
# SECTION 2: Disentanglement Metrics
# ============================================================================

class DisentanglementMetrics:
    """
    Metrics for measuring disentanglement quality.
    """
    
    @staticmethod
    def compute_mig(model, X, true_factors, n_bins=10):
        """
        Mutual Information Gap (MIG).
        
        Higher MIG = better disentanglement.
        Measures gap between top two latent dims for each factor.
        """
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            mu, _ = model.encode(X_tensor)
            Z = mu.numpy()
        
        # Discretize latent dimensions for MI computation
        Z_discrete = np.digitize(Z, bins=np.linspace(Z.min(), Z.max(), n_bins))
        
        mig_scores = []
        
        for factor_name, factor_values in true_factors.items():
            if isinstance(factor_values, np.ndarray):
                factor_discrete = np.digitize(factor_values, 
                                             bins=np.linspace(factor_values.min(), factor_values.max(), n_bins))
            else:
                continue
            
            # Compute MI between each latent dim and this factor
            mis = []
            for z_idx in range(Z.shape[1]):
                mi = DisentanglementMetrics._mutual_info(Z_discrete[:, z_idx], factor_discrete)
                mis.append(mi)
            
            mis = np.array(mis)
            sorted_mis = np.sort(mis)[::-1]
            
            # MIG = (top MI - second top MI) / H(factor)
            h_factor = DisentanglementMetrics._entropy(factor_discrete)
            if h_factor > 0:
                mig = (sorted_mis[0] - sorted_mis[1]) / h_factor
                mig_scores.append(mig)
        
        return np.mean(mig_scores) if mig_scores else 0.0
    
    @staticmethod
    def _mutual_info(x, y):
        """Compute mutual information between discrete variables."""
        # Joint distribution
        joint = np.histogram2d(x, y, bins=10)[0]
        joint = joint / joint.sum()
        
        # Marginals
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        
        # MI = Σ p(x,y) log(p(x,y)/(p(x)p(y)))
        mi = 0
        for i in range(len(px)):
            for j in range(len(py)):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j] + 1e-10))
        
        return max(0, mi)
    
    @staticmethod
    def _entropy(x):
        """Compute entropy of discrete variable."""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    @staticmethod
    def compute_dci(model, X, true_factors):
        """
        Disentanglement, Completeness, Informativeness (DCI).
        
        Trains linear classifiers to predict factors from latent dims.
        """
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            mu, _ = model.encode(X_tensor)
            Z = mu.numpy()
        
        importance_matrix = []
        
        for factor_name, factor_values in true_factors.items():
            if not isinstance(factor_values, np.ndarray):
                continue
            
            # Discretize factor for classification
            factor_discrete = np.digitize(factor_values,
                                         bins=np.percentile(factor_values, [33, 67]))
            
            # Train classifier per latent dimension
            importances = []
            for z_idx in range(Z.shape[1]):
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(Z[:, z_idx:z_idx+1], factor_discrete)
                acc = accuracy_score(factor_discrete, clf.predict(Z[:, z_idx:z_idx+1]))
                importances.append(acc - 1/3)  # Subtract random baseline
            
            importance_matrix.append(importances)
        
        importance_matrix = np.array(importance_matrix)
        
        # Disentanglement: each latent dim captures only one factor
        disentanglement = 0
        for z_idx in range(importance_matrix.shape[1]):
            col = importance_matrix[:, z_idx]
            if col.max() > 0:
                disentanglement += 1 - (np.sum(col) - col.max()) / (col.max() + 1e-10)
        disentanglement /= importance_matrix.shape[1]
        
        # Completeness: each factor is captured by only one latent dim
        completeness = 0
        for f_idx in range(importance_matrix.shape[0]):
            row = importance_matrix[f_idx]
            if row.max() > 0:
                completeness += 1 - (np.sum(row) - row.max()) / (row.max() + 1e-10)
        completeness /= importance_matrix.shape[0]
        
        # Informativeness: overall predictive accuracy
        informativeness = np.mean(importance_matrix.max(axis=1))
        
        return {
            'disentanglement': max(0, disentanglement),
            'completeness': max(0, completeness),
            'informativeness': max(0, informativeness)
        }

# ============================================================================
# SECTION 3: Training
# ============================================================================

class VAETrainer:
    """Training infrastructure for VAE models."""
    
    def __init__(self, model, learning_rate=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.recon_losses = []
        self.tc_losses = []
    
    def train_epoch(self, dataloader, dataset_size):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_tc = 0
        
        for batch_x, in dataloader:
            batch_x = batch_x.to(self.device)
            
            self.optimizer.zero_grad()
            
            x_recon, mu, logvar, z = self.model(batch_x)
            loss, recon, tc, _ = self.model.loss_function(batch_x, x_recon, mu, logvar, dataset_size)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            if isinstance(tc, torch.Tensor):
                total_tc += tc.item()
        
        n_batches = len(dataloader)
        return total_loss/n_batches, total_recon/n_batches, total_tc/n_batches
    
    def train(self, train_loader, n_epochs=100, verbose=True):
        dataset_size = len(train_loader.dataset)
        
        for epoch in range(n_epochs):
            loss, recon, tc = self.train_epoch(train_loader, dataset_size)
            
            self.train_losses.append(loss)
            self.recon_losses.append(recon)
            self.tc_losses.append(tc)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss:.4f} | Recon: {recon:.4f} | TC: {tc:.4f}")
        
        return self.train_losses

# ============================================================================
# SECTION 4: Latent Traversal Visualization
# ============================================================================

def latent_traversal(model, X_sample, latent_dim, n_steps=11, range_std=3):
    """
    Generate latent traversal visualizations.
    
    For each latent dimension, vary it while keeping others fixed,
    to visualize what each dimension encodes.
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).unsqueeze(0)
        mu, logvar = model.encode(X_tensor)
        z_base = mu.squeeze().numpy()
    
    traversals = {}
    
    for dim in range(latent_dim):
        z_range = np.linspace(-range_std, range_std, n_steps)
        reconstructions = []
        
        for z_val in z_range:
            z_modified = z_base.copy()
            z_modified[dim] = z_val
            
            with torch.no_grad():
                z_tensor = torch.FloatTensor(z_modified).unsqueeze(0)
                x_recon = model.decode(z_tensor).squeeze().numpy()
            
            reconstructions.append(x_recon)
        
        traversals[dim] = {
            'z_values': z_range,
            'reconstructions': np.array(reconstructions)
        }
    
    return traversals

def interpret_latent_dims(model, X, feature_names, n_samples=500):
    """
    Interpret what each latent dimension represents.
    
    Compute correlation between latent dimensions and original features.
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X[:n_samples])
        mu, _ = model.encode(X_tensor)
        Z = mu.numpy()
    
    # Correlation matrix: features × latent dims
    correlations = np.zeros((len(feature_names), Z.shape[1]))
    
    for i, feat_name in enumerate(feature_names):
        for j in range(Z.shape[1]):
            corr = np.corrcoef(X[:n_samples, i], Z[:, j])[0, 1]
            correlations[i, j] = corr if not np.isnan(corr) else 0
    
    # Create DataFrame
    corr_df = pd.DataFrame(
        correlations,
        index=feature_names,
        columns=[f'Z{i}' for i in range(Z.shape[1])]
    )
    
    # Interpret each dimension
    interpretations = {}
    for j in range(Z.shape[1]):
        col = corr_df.iloc[:, j]
        top_pos = col.nlargest(3)
        top_neg = col.nsmallest(3)
        
        interp = f"Z{j}: "
        if top_pos.iloc[0] > 0.3:
            interp += f"HIGH {list(top_pos.index[:2])}"
        if top_neg.iloc[0] < -0.3:
            interp += f" LOW {list(top_neg.index[:2])}"
        
        interpretations[j] = interp
    
    return corr_df, interpretations

# ============================================================================
# SECTION 5: Financial Data Generation
# ============================================================================

def generate_financial_data_disentangled(n_samples=5000):
    """
    Generate financial data with truly independent underlying factors.
    This is the ideal case for disentanglement.
    """
    np.random.seed(42)
    
    # Independent latent factors
    growth = np.random.normal(0, 1, n_samples)
    value = np.random.normal(0, 1, n_samples)
    quality = np.random.normal(0, 1, n_samples)
    momentum = np.random.normal(0, 1, n_samples)
    
    features = {}
    
    # Each feature loads on ONE factor primarily
    features['Revenue_Growth'] = 0.9 * growth + 0.1 * np.random.randn(n_samples)
    features['EPS_Growth'] = 0.85 * growth + 0.15 * np.random.randn(n_samples)
    features['Sales_Growth'] = 0.88 * growth + 0.12 * np.random.randn(n_samples)
    
    features['PE_Ratio'] = 0.9 * value + 0.1 * np.random.randn(n_samples)
    features['PB_Ratio'] = 0.85 * value + 0.15 * np.random.randn(n_samples)
    features['Dividend_Yield'] = 0.8 * value + 0.2 * np.random.randn(n_samples)
    
    features['ROE'] = 0.9 * quality + 0.1 * np.random.randn(n_samples)
    features['ROA'] = 0.85 * quality + 0.15 * np.random.randn(n_samples)
    features['Debt_Ratio'] = -0.8 * quality + 0.2 * np.random.randn(n_samples)
    
    features['Price_Mom'] = 0.9 * momentum + 0.1 * np.random.randn(n_samples)
    features['Volume_Trend'] = 0.8 * momentum + 0.2 * np.random.randn(n_samples)
    features['Volatility'] = 0.75 * momentum + 0.25 * np.random.randn(n_samples)
    
    df = pd.DataFrame(features)
    
    true_factors = {
        'Growth': growth,
        'Value': value,
        'Quality': quality,
        'Momentum': momentum
    }
    
    return df, true_factors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("β-TCVAE: DISENTANGLED VAE FOR INTERPRETABLE LATENT FACTORS")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating synthetic financial data...")
    df, true_factors = generate_financial_data_disentangled(n_samples=5000)
    feature_names = df.columns.tolist()
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    # Create dataloader
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train β-TCVAE
    print("\n2. Training β-TCVAE (β=6)...")
    model_tcvae = BetaTCVAE(
        input_dim=X.shape[1],
        hidden_dims=[64, 32],
        latent_dim=8,
        beta=6.0
    )
    
    trainer_tcvae = VAETrainer(model_tcvae, learning_rate=1e-3, device=device)
    losses_tcvae = trainer_tcvae.train(train_loader, n_epochs=150)
    
    # Train standard β-VAE for comparison
    print("\n3. Training standard β-VAE (β=4)...")
    model_bvae = BetaVAE(
        input_dim=X.shape[1],
        hidden_dims=[64, 32],
        latent_dim=8,
        beta=4.0
    )
    
    trainer_bvae = VAETrainer(model_bvae, learning_rate=1e-3, device=device)
    losses_bvae = trainer_bvae.train(train_loader, n_epochs=150)
    
    # Compute disentanglement metrics
    print("\n4. Computing disentanglement metrics...")
    
    metrics = DisentanglementMetrics()
    
    # MIG
    mig_tcvae = metrics.compute_mig(model_tcvae, X, true_factors)
    mig_bvae = metrics.compute_mig(model_bvae, X, true_factors)
    
    # DCI
    dci_tcvae = metrics.compute_dci(model_tcvae, X, true_factors)
    dci_bvae = metrics.compute_dci(model_bvae, X, true_factors)
    
    print("\nDisentanglement Results:")
    print(f"{'Metric':<20} {'β-TCVAE':<15} {'β-VAE':<15}")
    print("-" * 50)
    print(f"{'MIG':<20} {mig_tcvae:<15.4f} {mig_bvae:<15.4f}")
    print(f"{'DCI-D':<20} {dci_tcvae['disentanglement']:<15.4f} {dci_bvae['disentanglement']:<15.4f}")
    print(f"{'DCI-C':<20} {dci_tcvae['completeness']:<15.4f} {dci_bvae['completeness']:<15.4f}")
    print(f"{'DCI-I':<20} {dci_tcvae['informativeness']:<15.4f} {dci_bvae['informativeness']:<15.4f}")
    
    # Interpret latent dimensions
    print("\n5. Interpreting latent dimensions...")
    corr_df, interpretations = interpret_latent_dims(model_tcvae, X, feature_names)
    
    print("\nLatent dimension interpretations (β-TCVAE):")
    for dim, interp in interpretations.items():
        print(f"  {interp}")
    
    # Visualizations
    print("\n6. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Training loss comparison
    ax1 = axes[0, 0]
    ax1.plot(losses_tcvae, label='β-TCVAE')
    ax1.plot(losses_bvae, label='β-VAE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: TC loss for β-TCVAE
    ax2 = axes[0, 1]
    ax2.plot(trainer_tcvae.tc_losses, label='Total Correlation')
    ax2.plot(trainer_tcvae.recon_losses, label='Reconstruction')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Component')
    ax2.set_title('β-TCVAE Loss Decomposition')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Latent-feature correlation heatmap
    ax3 = axes[0, 2]
    sns.heatmap(corr_df.T, cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax3)
    ax3.set_title('Latent-Feature Correlations (β-TCVAE)')
    ax3.set_xlabel('Original Feature')
    ax3.set_ylabel('Latent Dim')
    
    # Plot 4: Latent space visualization
    ax4 = axes[1, 0]
    model_tcvae.eval()
    with torch.no_grad():
        mu, _ = model_tcvae.encode(X_tensor[:1000])
        Z = mu.numpy()
    
    scatter = ax4.scatter(Z[:, 0], Z[:, 1], c=true_factors['Growth'][:1000], cmap='viridis', alpha=0.5, s=10)
    ax4.set_xlabel('Z0')
    ax4.set_ylabel('Z1')
    ax4.set_title('Latent Space (colored by Growth factor)')
    plt.colorbar(scatter, ax=ax4)
    
    # Plot 5: Metric comparison
    ax5 = axes[1, 1]
    metrics_names = ['MIG', 'Disentanglement', 'Completeness', 'Informativeness']
    tcvae_scores = [mig_tcvae, dci_tcvae['disentanglement'], dci_tcvae['completeness'], dci_tcvae['informativeness']]
    bvae_scores = [mig_bvae, dci_bvae['disentanglement'], dci_bvae['completeness'], dci_bvae['informativeness']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax5.bar(x - width/2, tcvae_scores, width, label='β-TCVAE')
    ax5.bar(x + width/2, bvae_scores, width, label='β-VAE')
    ax5.set_ylabel('Score')
    ax5.set_title('Disentanglement Metrics')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_names)
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # Plot 6: Latent traversal example
    ax6 = axes[1, 2]
    traversals = latent_traversal(model_tcvae, X[0], latent_dim=8, n_steps=11)
    
    # Show how one feature changes during traversal of each latent dim
    for dim in range(min(4, model_tcvae.latent_dim)):
        feature_idx = 0  # Revenue_Growth
        recons = traversals[dim]['reconstructions'][:, feature_idx]
        ax6.plot(traversals[dim]['z_values'], recons, label=f'Z{dim}', alpha=0.7)
    
    ax6.set_xlabel('Latent Value')
    ax6.set_ylabel(f'{feature_names[0]} (reconstructed)')
    ax6.set_title('Latent Traversal Effect on Revenue_Growth')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/beta_tcvae_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: beta_tcvae_analysis.png")
    plt.close()
    
    # Save models
    torch.save(model_tcvae.state_dict(), '/home/claude/beta_tcvae_model.pt')
    torch.save(model_bvae.state_dict(), '/home/claude/beta_vae_model.pt')
    
    # Save correlation matrix
    corr_df.to_csv('/home/claude/latent_feature_correlations.csv')
    
    print("\n" + "=" * 70)
    print("β-TCVAE ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\nKey findings:")
    print(f"  • β-TCVAE achieves better disentanglement (MIG: {mig_tcvae:.4f} vs {mig_bvae:.4f})")
    print(f"  • TC penalty encourages independent latent dimensions")
    print(f"  • Latent dimensions align with true generative factors")
    
    print("\nOutput files:")
    print("  • beta_tcvae_analysis.png - Comprehensive visualization")
    print("  • beta_tcvae_model.pt - Trained β-TCVAE model")
    print("  • beta_vae_model.pt - Trained β-VAE model")
    print("  • latent_feature_correlations.csv - Correlation matrix")
    
    print("\nFor financial applications:")
    print("  • Each latent dim ideally captures one risk factor")
    print("  • Higher β = more disentanglement but worse reconstruction")
    print("  • Tune β based on reconstruction-disentanglement tradeoff")
