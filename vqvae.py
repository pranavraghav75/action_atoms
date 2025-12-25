"""
3D VQ-VAE for Event Volume Tokenization
Learns discrete "Action Atoms" from neuromorphic event streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer - the "Codebook" of Action Atoms
    Maps continuous encoder outputs to discrete codes
    """
    
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        """
        Args:
            num_embeddings: Size of codebook (number of Action Atoms)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # The Codebook - learnable discrete vectors
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize encoder output to nearest codebook vector
        
        Args:
            z: Encoder output (B, C, T, H, W)
            
        Returns:
            quantized: Quantized vectors (B, C, T, H, W)
            loss: VQ loss (commitment + codebook)
            encoding_indices: Which atoms were used (B, T, H, W)
        """
        # Reshape z -> (B*T*H*W, C)
        B, C, T, H, W = z.shape
        z_flattened = z.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
        z_flattened = z_flattened.view(-1, C)  # (B*T*H*W, C)
        
        # Calculate distances to all codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )
        
        # Find nearest codebook vector (the "atom")
        encoding_indices = torch.argmin(distances, dim=1)  # (B*T*H*W,)
        
        # Get quantized vectors
        quantized = self.embeddings(encoding_indices)  # (B*T*H*W, C)
        
        # Reshape back to (B, C, T, H, W)
        quantized = quantized.view(B, T, H, W, C)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
        # VQ Losses
        # Codebook loss: move embeddings towards encoder outputs
        codebook_loss = F.mse_loss(quantized.detach(), z)
        
        # Commitment loss: encourage encoder to commit to embeddings
        commitment_loss = F.mse_loss(quantized, z.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = z + (quantized - z).detach()
        
        # Reshape encoding_indices for visualization
        encoding_indices = encoding_indices.view(B, T, H, W)
        
        return quantized, vq_loss, encoding_indices


class Residual3DBlock(nn.Module):
    """3D Residual block for temporal-spatial processing"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        return F.relu(out + residual)


class Encoder3D(nn.Module):
    """
    3D Encoder: Event Volume -> Latent Space
    Uses 3D convolutions to capture spatial-temporal motion patterns
    """
    
    def __init__(self, in_channels=2, hidden_dims=[64, 128, 256], latent_dim=64):
        """
        Args:
            in_channels: Input channels (2 for ON/OFF events)
            hidden_dims: Hidden layer dimensions
            latent_dim: Output dimension (matches codebook embedding_dim)
        """
        super().__init__()
        
        layers = []
        
        # Initial conv
        layers.append(nn.Conv3d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm3d(hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Residual blocks with downsampling
        for i in range(len(hidden_dims) - 1):
            layers.append(Residual3DBlock(hidden_dims[i], hidden_dims[i]))
            layers.append(nn.Conv3d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final residual blocks
        layers.append(Residual3DBlock(hidden_dims[-1], hidden_dims[-1]))
        layers.append(Residual3DBlock(hidden_dims[-1], hidden_dims[-1]))
        
        # Project to latent dimension
        layers.append(nn.Conv3d(hidden_dims[-1], latent_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder3D(nn.Module):
    """
    3D Decoder: Latent Space -> Event Volume
    Reconstructs event stream from quantized codes
    """
    
    def __init__(self, latent_dim=64, hidden_dims=[256, 128, 64], out_channels=2):
        """
        Args:
            latent_dim: Input dimension (matches codebook embedding_dim)
            hidden_dims: Hidden layer dimensions (reversed from encoder)
            out_channels: Output channels (2 for ON/OFF events)
        """
        super().__init__()
        
        layers = []
        
        # Project from latent dimension
        layers.append(nn.Conv3d(latent_dim, hidden_dims[0], kernel_size=1))
        
        # Residual blocks
        layers.append(Residual3DBlock(hidden_dims[0], hidden_dims[0]))
        layers.append(Residual3DBlock(hidden_dims[0], hidden_dims[0]))
        
        # Upsampling blocks
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ConvTranspose3d(hidden_dims[i], hidden_dims[i+1], 
                                            kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(Residual3DBlock(hidden_dims[i+1], hidden_dims[i+1]))
        
        # Final upsampling and output
        layers.append(nn.ConvTranspose3d(hidden_dims[-1], out_channels, 
                                        kernel_size=4, stride=2, padding=1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class VQVAE3D(nn.Module):
    """
    Complete 3D VQ-VAE Model
    Learns discrete "Action Atoms" from event volumes
    """
    
    def __init__(
        self,
        in_channels=2,
        hidden_dims=[64, 128, 256],
        latent_dim=64,
        num_embeddings=512,
        commitment_cost=0.25
    ):
        """
        Args:
            in_channels: Input channels (2 for ON/OFF events)
            hidden_dims: Hidden dimensions for encoder/decoder
            latent_dim: Latent space dimension
            num_embeddings: Codebook size (number of Action Atoms)
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        
        self.encoder = Encoder3D(in_channels, hidden_dims, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder3D(latent_dim, list(reversed(hidden_dims)), in_channels)
        
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through VQ-VAE
        
        Args:
            x: Event volume (B, 2, T, H, W)
            
        Returns:
            reconstructed: Reconstructed events (B, 2, T, H, W)
            vq_loss: Vector quantization loss
            encoding_indices: Discrete atom indices (B, T, H, W)
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize (the "bottleneck" - discrete atoms)
        quantized, vq_loss, encoding_indices = self.vq(z)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        return reconstructed, vq_loss, encoding_indices
    
    def encode(self, x: Tensor) -> Tensor:
        """Encode to discrete indices (for Transformer input)"""
        z = self.encoder(x)
        _, _, encoding_indices = self.vq(z)
        return encoding_indices
    
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Decode from discrete indices (for generation)"""
        # Get quantized vectors from indices
        quantized = self.vq.embeddings(indices)
        
        # Reshape to (B, C, T, H, W)
        B, T, H, W = indices.shape
        quantized = quantized.view(B, T, H, W, self.latent_dim)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
        # Decode
        reconstructed = self.decoder(quantized)
        return reconstructed


if __name__ == "__main__":
    # Test the model
    print("Testing 3D VQ-VAE Architecture...")
    
    # Create model
    model = VQVAE3D(
        in_channels=2,
        hidden_dims=[64, 128, 256],
        latent_dim=64,
        num_embeddings=512,
        commitment_cost=0.25
    )
    
    # Test input (B=2, C=2, T=16, H=240, W=320)
    x = torch.randn(2, 2, 16, 240, 320)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    reconstructed, vq_loss, encoding_indices = model(x)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Encoding indices shape: {encoding_indices.shape}")
    print(f"Unique atoms used: {len(torch.unique(encoding_indices))}/{model.num_embeddings}")
    
    # Test encoding/decoding
    indices = model.encode(x)
    decoded = model.decode_from_indices(indices)
    
    print(f"\nEncode-decode test:")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Decoded shape: {decoded.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
