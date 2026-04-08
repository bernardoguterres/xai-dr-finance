"""
Sparse Autoencoders for Interpretable Feature Discovery
Based on Anthropic's research on sparse autoencoders for monosemantic features.

Author: Bernardo Guterres
Date: December 2025

Neural networks exhibit "superposition" - encoding more concepts 
than they have neurons. Sparse autoencoders decompose this into monosemantic,
individually interpretable features through sparse coding.

- Cunningham et al. (2023): "Sparse Autoencoders Find Highly Interpretable Features"
- Anthropic's research on dictionary learning in language models
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
from sklearn.decomposition import PCA
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# SECTION 1: Sparse Autoencoder Architectures
# ============================================================================

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with TopK activation.
    
    TopK sparsity is superior to L1 regularization because:
    1. No "activation shrinkage" - activations maintain magnitude
    2. Exact sparsity control - exactly K features active
    3. Better scaling properties across different K values
    """
    
    def __init__(self, input_dim, hidden_dim, sparsity_k=32):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_k = sparsity_k
        
        # Encoder: input → hidden (overcomplete)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: hidden → input
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize with unit norm columns
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights with unit norm."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder columns
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x):
        """Encode input to sparse hidden representation."""
        # Pre-activation
        h = self.encoder(x)
        
        # ReLU then TopK sparsity
        h = F.relu(h)
        
        # TopK: keep only top K activations
        if self.sparsity_k < self.hidden_dim:
            topk_values, topk_indices = torch.topk(h, self.sparsity_k, dim=-1)
            sparse_h = torch.zeros_like(h)
            sparse_h.scatter_(-1, topk_indices, topk_values)
            return sparse_h
        else:
            return h
    
    def decode(self, h):
        """Decode sparse representation back to input space."""
        return self.decoder(h)
    
    def forward(self, x):
        """Full forward pass."""
        h = self.encode(x)
        x_reconstructed = self.decode(h)
        return x_reconstructed, h
    
    def get_feature_activations(self, x):
        """Get which features activate for each input."""
        with torch.no_grad():
            h = self.encode(x)
            return h

class L1SparseAutoencoder(nn.Module):
    """
    Alternative: L1-regularized Sparse Autoencoder.
    Included for comparison with TopK approach.
    """
    
    def __init__(self, input_dim, hidden_dim, l1_coeff=0.001):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def forward(self, x):
        h = F.relu(self.encoder(x))
        x_reconstructed = self.decoder(h)
        return x_reconstructed, h
    
    def get_l1_loss(self, h):
        """L1 penalty on activations."""
        return self.l1_coeff * torch.mean(torch.abs(h))

class GatedSparseAutoencoder(nn.Module):
    """
    Gated Sparse Autoencoder - addresses activation shrinkage better than L1.
    Uses separate gating network to determine which features fire.
    """
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Magnitude encoder
        self.encoder_mag = nn.Linear(input_dim, hidden_dim)
        
        # Gate encoder (determines which features fire)
        self.encoder_gate = nn.Linear(input_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Compute magnitude
        magnitude = F.relu(self.encoder_mag(x))
        
        # Compute gate (sigmoid for soft gating)
        gate = torch.sigmoid(self.encoder_gate(x))
        
        # Sparse representation = magnitude * gate
        h = magnitude * gate
        
        # Reconstruct
        x_reconstructed = self.decoder(h)
        
        return x_reconstructed, h, gate

# ============================================================================
# SECTION 2: Training Infrastructure
# ============================================================================

class SparseAutoencoderTrainer:
    """
    Training and analysis pipeline for sparse autoencoders.
    """
    
    def __init__(self, model, learning_rate=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.sparsity_history = []
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_sparsity = 0
        
        for batch_x, in dataloader:
            batch_x = batch_x.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, L1SparseAutoencoder):
                x_recon, h = self.model(batch_x)
                recon_loss = F.mse_loss(x_recon, batch_x)
                l1_loss = self.model.get_l1_loss(h)
                loss = recon_loss + l1_loss
            elif isinstance(self.model, GatedSparseAutoencoder):
                x_recon, h, gate = self.model(batch_x)
                recon_loss = F.mse_loss(x_recon, batch_x)
                # Encourage sparsity in gate
                gate_sparsity = torch.mean(gate)
                loss = recon_loss + 0.01 * gate_sparsity
            else:
                x_recon, h = self.model(batch_x)
                loss = F.mse_loss(x_recon, batch_x)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Track sparsity (fraction of zero activations)
            with torch.no_grad():
                sparsity = (h == 0).float().mean().item()
                total_sparsity += sparsity
        
        avg_loss = total_loss / len(dataloader)
        avg_sparsity = total_sparsity / len(dataloader)
        
        return avg_loss, avg_sparsity
    
    def evaluate(self, dataloader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                
                if isinstance(self.model, GatedSparseAutoencoder):
                    x_recon, h, _ = self.model(batch_x)
                else:
                    x_recon, h = self.model(batch_x)
                
                loss = F.mse_loss(x_recon, batch_x)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader=None, n_epochs=100, verbose=True):
        """Full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            train_loss, sparsity = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.sparsity_history.append(sparsity)
            
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 20 == 0:
                msg = f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Sparsity: {sparsity:.3f}"
                if val_loader:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)
        
        return self.train_losses, self.val_losses

# ============================================================================
# SECTION 3: Feature Interpretation
# ============================================================================

class FeatureInterpreter:
    """
    Interpret learned sparse autoencoder features.
    """
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.model.eval()
    
    def get_decoder_weights(self):
        """Get decoder weight matrix (features × hidden units)."""
        with torch.no_grad():
            weights = self.model.decoder.weight.cpu().numpy()
        return pd.DataFrame(weights, index=self.feature_names)
    
    def interpret_hidden_unit(self, unit_idx, top_n=5):
        """
        Interpret what a specific hidden unit represents.
        
        Returns:
            Dict with top contributing input features
        """
        decoder_weights = self.get_decoder_weights()
        unit_weights = decoder_weights.iloc[:, unit_idx]
        
        # Top positive and negative contributions
        top_positive = unit_weights.nlargest(top_n)
        top_negative = unit_weights.nsmallest(top_n)
        
        return {
            'unit_idx': unit_idx,
            'top_positive': top_positive.to_dict(),
            'top_negative': top_negative.to_dict(),
            'interpretation': self._generate_interpretation(top_positive, top_negative)
        }
    
    def _generate_interpretation(self, top_pos, top_neg):
        """Generate human-readable interpretation."""
        pos_features = list(top_pos.index[:2])
        neg_features = list(top_neg.index[:2])
        
        interp = f"HIGH when: {', '.join(pos_features)}"
        if top_neg.iloc[0] < -0.1:
            interp += f" | LOW when: {', '.join(neg_features)}"
        
        return interp
    
    def find_activated_units(self, x, threshold=0.1):
        """Find which hidden units activate for input x."""
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        
        with torch.no_grad():
            h = self.model.encode(x_tensor)
            h = h.squeeze().numpy()
        
        activated = np.where(h > threshold)[0]
        activations = {idx: h[idx] for idx in activated}
        
        return activations
    
    def explain_sample(self, x, top_n=5):
        """
        Explain a sample in terms of activated sparse features.
        """
        activations = self.find_activated_units(x)
        
        explanation = {
            'n_activated': len(activations),
            'active_features': []
        }
        
        # Sort by activation strength
        sorted_activations = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        
        for unit_idx, activation_value in sorted_activations[:top_n]:
            unit_interp = self.interpret_hidden_unit(unit_idx)
            explanation['active_features'].append({
                'unit_idx': unit_idx,
                'activation': activation_value,
                'interpretation': unit_interp['interpretation']
            })
        
        return explanation
    
    def get_feature_dictionary(self, min_weight=0.2):
        """
        Get dictionary mapping hidden units to their interpretations.
        Only includes units with strong input feature associations.
        """
        decoder_weights = self.get_decoder_weights()
        dictionary = {}
        
        for unit_idx in range(decoder_weights.shape[1]):
            weights = decoder_weights.iloc[:, unit_idx]
            max_weight = weights.abs().max()
            
            if max_weight >= min_weight:
                interp = self.interpret_hidden_unit(unit_idx)
                dictionary[unit_idx] = interp
        
        return dictionary
    
    def visualize_features(self, n_units=20, save_path='sparse_ae_features.png'):
        """Visualize top hidden unit weight patterns."""
        decoder_weights = self.get_decoder_weights()
        
        # Select most interpretable units (highest max weight)
        max_weights = decoder_weights.abs().max()
        top_units = max_weights.nlargest(n_units).index
        
        weights_subset = decoder_weights.iloc[:, top_units]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            weights_subset.values.T,
            xticklabels=self.feature_names,
            yticklabels=[f'Unit {i}' for i in top_units],
            cmap='RdBu_r',
            center=0,
            annot=False,
            ax=ax
        )
        ax.set_title(f'Top {n_units} Sparse Autoencoder Features')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature visualization saved to: {save_path}")
        plt.close()

# ============================================================================
# SECTION 4: Comparison with PCA
# ============================================================================

def compare_with_pca(X, sparse_ae_model, feature_names, n_components=10):
    """
    Compare sparse autoencoder features with PCA components.
    """
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Get PCA loadings
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Get sparse AE decoder weights
    interpreter = FeatureInterpreter(sparse_ae_model, feature_names)
    ae_weights = interpreter.get_decoder_weights()
    
    # Compare sparsity
    pca_sparsity = (np.abs(pca.components_) < 0.1).mean()
    
    with torch.no_grad():
        ae_decoder = sparse_ae_model.decoder.weight.cpu().numpy()
    ae_sparsity = (np.abs(ae_decoder) < 0.1).mean()
    
    comparison = {
        'PCA_explained_variance': pca.explained_variance_ratio_.sum(),
        'PCA_sparsity': pca_sparsity,
        'SAE_decoder_sparsity': ae_sparsity,
        'PCA_n_components': n_components,
        'SAE_hidden_dim': sparse_ae_model.hidden_dim
    }
    
    return comparison, pca_loadings

# ============================================================================
# SECTION 5: Financial Data Generation
# ============================================================================

def generate_financial_data(n_samples=5000, n_features=20):
    """
    Generate synthetic financial data with interpretable structure.
    """
    np.random.seed(42)
    
    # Latent factors
    growth = np.random.normal(0, 1, n_samples)
    value = np.random.normal(0, 1, n_samples)
    quality = np.random.normal(0, 1, n_samples)
    momentum = np.random.normal(0, 1, n_samples)
    
    features = {}
    
    # Growth features
    features['Revenue_Growth'] = 0.8 * growth + 0.2 * np.random.randn(n_samples)
    features['EPS_Growth'] = 0.7 * growth + 0.15 * quality + 0.25 * np.random.randn(n_samples)
    features['Sales_Growth'] = 0.75 * growth + 0.25 * np.random.randn(n_samples)
    features['R&D_Intensity'] = 0.5 * growth + 0.3 * quality + 0.35 * np.random.randn(n_samples)
    features['Capex_Growth'] = 0.6 * growth + 0.2 * momentum + 0.35 * np.random.randn(n_samples)
    
    # Value features
    features['PE_Ratio'] = 0.8 * value + 0.2 * np.random.randn(n_samples)
    features['PB_Ratio'] = 0.7 * value + 0.2 * quality + 0.25 * np.random.randn(n_samples)
    features['Dividend_Yield'] = 0.6 * value - 0.3 * growth + 0.35 * np.random.randn(n_samples)
    features['FCF_Yield'] = 0.5 * value + 0.3 * quality + 0.35 * np.random.randn(n_samples)
    features['EV_EBITDA'] = 0.65 * value + 0.35 * np.random.randn(n_samples)
    
    # Quality features
    features['ROE'] = 0.8 * quality + 0.2 * np.random.randn(n_samples)
    features['ROA'] = 0.75 * quality + 0.25 * np.random.randn(n_samples)
    features['Debt_to_Equity'] = -0.7 * quality + 0.3 * np.random.randn(n_samples)
    features['Interest_Coverage'] = 0.6 * quality + 0.4 * np.random.randn(n_samples)
    features['Gross_Margin'] = 0.55 * quality + 0.2 * growth + 0.4 * np.random.randn(n_samples)
    
    # Momentum features
    features['Price_Mom_1M'] = 0.8 * momentum + 0.2 * np.random.randn(n_samples)
    features['Price_Mom_6M'] = 0.7 * momentum + 0.1 * growth + 0.3 * np.random.randn(n_samples)
    features['Volume_Trend'] = 0.5 * momentum + 0.5 * np.random.randn(n_samples)
    features['Volatility'] = 0.4 * momentum - 0.3 * quality + 0.5 * np.random.randn(n_samples)
    features['Beta'] = 0.45 * momentum + 0.55 * np.random.randn(n_samples)
    
    df = pd.DataFrame(features)
    
    true_factors = {
        'Growth': ['Revenue_Growth', 'EPS_Growth', 'Sales_Growth', 'R&D_Intensity', 'Capex_Growth'],
        'Value': ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'FCF_Yield', 'EV_EBITDA'],
        'Quality': ['ROE', 'ROA', 'Debt_to_Equity', 'Interest_Coverage', 'Gross_Margin'],
        'Momentum': ['Price_Mom_1M', 'Price_Mom_6M', 'Volume_Trend', 'Volatility', 'Beta']
    }
    
    return df, true_factors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPARSE AUTOENCODERS FOR INTERPRETABLE FEATURE DISCOVERY")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating synthetic financial data...")
    df, true_factors = generate_financial_data(n_samples=5000, n_features=20)
    feature_names = df.columns.tolist()
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    # Create datasets
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Create and train TopK Sparse Autoencoder
    print("\n2. Training TopK Sparse Autoencoder...")
    input_dim = X.shape[1]
    hidden_dim = 128  # Overcomplete (more hidden units than input features)
    sparsity_k = 8    # Only 8 features active at a time
    
    model_topk = SparseAutoencoder(input_dim, hidden_dim, sparsity_k=sparsity_k)
    trainer_topk = SparseAutoencoderTrainer(model_topk, learning_rate=1e-3, device=device)
    
    train_losses, val_losses = trainer_topk.train(train_loader, val_loader, n_epochs=100)
    
    # Train L1 Sparse Autoencoder for comparison
    print("\n3. Training L1 Sparse Autoencoder for comparison...")
    model_l1 = L1SparseAutoencoder(input_dim, hidden_dim, l1_coeff=0.001)
    trainer_l1 = SparseAutoencoderTrainer(model_l1, learning_rate=1e-3, device=device)
    
    train_losses_l1, val_losses_l1 = trainer_l1.train(train_loader, val_loader, n_epochs=100)
    
    # Interpret features
    print("\n4. Interpreting learned features...")
    interpreter = FeatureInterpreter(model_topk, feature_names)
    
    # Get feature dictionary
    dictionary = interpreter.get_feature_dictionary(min_weight=0.15)
    
    print(f"\nDiscovered {len(dictionary)} interpretable features:")
    for unit_idx, interp in list(dictionary.items())[:10]:
        print(f"  Unit {unit_idx}: {interp['interpretation']}")
    
    # Explain sample
    print("\n5. Explaining a sample...")
    sample_idx = 42
    sample = X[sample_idx]
    explanation = interpreter.explain_sample(sample, top_n=5)
    
    print(f"Sample {sample_idx} explanation:")
    print(f"  Activated features: {explanation['n_activated']}")
    for feat in explanation['active_features']:
        print(f"  • {feat['interpretation']} (activation: {feat['activation']:.3f})")
    
    # Compare with PCA
    print("\n6. Comparing with PCA...")
    comparison, pca_loadings = compare_with_pca(X, model_topk, feature_names, n_components=10)
    
    print("\nComparison results:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Visualize
    print("\n7. Creating visualizations...")
    interpreter.visualize_features(n_units=15, save_path='/home/claude/sparse_ae_features.png')
    
    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(train_losses, label='TopK Train')
    ax1.plot(val_losses, label='TopK Val')
    ax1.plot(train_losses_l1, label='L1 Train', linestyle='--')
    ax1.plot(val_losses_l1, label='L1 Val', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sparsity comparison
    ax2 = axes[1]
    ax2.plot(trainer_topk.sparsity_history, label='TopK')
    ax2.plot(trainer_l1.sparsity_history, label='L1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sparsity (fraction zeros)')
    ax2.set_title('Sparsity Over Training')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Activation histogram
    ax3 = axes[2]
    with torch.no_grad():
        sample_batch = X_tensor[:500]
        h_topk = model_topk.encode(sample_batch).numpy().flatten()
        h_l1 = F.relu(model_l1.encoder(sample_batch)).numpy().flatten()
    
    ax3.hist(h_topk[h_topk > 0], bins=50, alpha=0.7, label='TopK', density=True)
    ax3.hist(h_l1[h_l1 > 0], bins=50, alpha=0.7, label='L1', density=True)
    ax3.set_xlabel('Activation Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Non-zero Activation Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/sparse_ae_training.png', dpi=150, bbox_inches='tight')
    print("Training visualization saved to: sparse_ae_training.png")
    plt.close()
    
    # Save model
    torch.save(model_topk.state_dict(), '/home/claude/sparse_ae_model.pt')
    print("Model saved to: sparse_ae_model.pt")
    
    print("\n" + "=" * 70)
    print("SPARSE AUTOENCODER ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\nKey findings:")
    print(f"  • TopK SAE final val loss: {val_losses[-1]:.6f}")
    print(f"  • L1 SAE final val loss: {val_losses_l1[-1]:.6f}")
    print(f"  • Interpretable features discovered: {len(dictionary)}")
    print(f"  • Hidden dimension: {hidden_dim}, Sparsity K: {sparsity_k}")
    
    print("\nAdvantages of TopK over L1:")
    print("  • No activation shrinkage")
    print("  • Exact sparsity control")
    print("  • Better gradient flow")
    
    print("\nOutput files:")
    print("  • sparse_ae_features.png - Feature weight patterns")
    print("  • sparse_ae_training.png - Training curves")
    print("  • sparse_ae_model.pt - Saved model")
