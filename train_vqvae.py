"""
Training script for 3D VQ-VAE
Learns discrete "Action Atoms" from event volumes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

from vqvae import VQVAE3D
from dataset import get_dataloaders


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch (event_clip, rgb_clip, metadata) or (event_clip, metadata)
        if len(batch) == 3:
            event_clip, _, _ = batch
        else:
            event_clip, _ = batch
        
        event_clip = event_clip.to(device)
        
        # Forward pass
        reconstructed, vq_loss, encoding_indices = model(event_clip)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, event_clip)
        
        # Total loss
        loss = recon_loss + vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'vq': vq_loss.item()
        })
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches
    }


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Unpack batch
            if len(batch) == 3:
                event_clip, _, _ = batch
            else:
                event_clip, _ = batch
            
            event_clip = event_clip.to(device)
            
            # Forward pass
            reconstructed, vq_loss, encoding_indices = model(event_clip)
            
            # Losses
            recon_loss = nn.functional.mse_loss(reconstructed, event_clip)
            loss = recon_loss + vq_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
    
    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'vq_loss': total_vq_loss / n_batches
    }


def visualize_reconstruction(model, val_loader, device, output_dir, epoch):
    """Save visualization of original vs reconstructed events"""
    import matplotlib.pyplot as plt
    
    model.eval()
    
    # Get one batch
    batch = next(iter(val_loader))
    if len(batch) == 3:
        event_clip, _, metadata = batch
    else:
        event_clip, metadata = batch
    
    event_clip = event_clip.to(device)
    
    with torch.no_grad():
        reconstructed, _, encoding_indices = model(event_clip)
    
    # Move to CPU
    event_clip = event_clip.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    encoding_indices = encoding_indices.cpu().numpy()
    
    # Visualize first sample, middle frame
    sample_idx = 0
    frame_idx = event_clip.shape[2] // 2  # Middle frame
    
    # Original events (ON and OFF channels)
    orig_on = event_clip[sample_idx, 0, frame_idx]
    orig_off = event_clip[sample_idx, 1, frame_idx]
    
    # Reconstructed events
    recon_on = reconstructed[sample_idx, 0, frame_idx]
    recon_off = reconstructed[sample_idx, 1, frame_idx]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(orig_on, cmap='Reds', vmin=0, vmax=1)
    axes[0, 0].set_title('Original ON Events')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(recon_on, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Reconstructed ON Events')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(orig_off, cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('Original OFF Events')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(recon_off, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title('Reconstructed OFF Events')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - Frame {frame_idx}')
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'reconstruction_epoch_{epoch:03d}.png', dpi=150)
    plt.close()
    
    print(f"Saved visualization to {output_dir / f'reconstruction_epoch_{epoch:03d}.png'}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup tensorboard if available
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / 'logs')
        print("TensorBoard logging enabled")
    else:
        print("TensorBoard logging disabled (tensorboard not installed)")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        video_dir=args.video_dir,
        event_dir=args.event_dir,
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        stride=args.stride,
        num_workers=args.num_workers,
        train_split=args.train_split,
        load_rgb=False  # Don't need RGB for VQ-VAE training
    )
    
    # Create model
    print("Creating model...")
    model = VQVAE3D(
        in_channels=2,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Log metrics
        if writer:
            for key in ['loss', 'recon_loss', 'vq_loss']:
                writer.add_scalar(f'train/{key}', train_metrics[key], epoch)
                writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}, VQ: {train_metrics['vq_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon_loss']:.4f}, VQ: {val_metrics['vq_loss']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch:03d}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"✓ New best model! Val loss: {best_val_loss:.4f}")
        
        # Visualize reconstruction
        if epoch % args.vis_every == 0:
            visualize_reconstruction(model, val_loader, device, output_dir / 'visualizations', epoch)
    
    if writer:
        writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQ-VAE on event volumes')
    
    # Data
    parser.add_argument('--video-dir', type=str, default='20bn-something-something-v2/filtered_videos')
    parser.add_argument('--event-dir', type=str, default='events_output')
    parser.add_argument('--output-dir', type=str, default='vqvae_output')
    
    # Model
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--num-embeddings', type=int, default=512, help='Codebook size (number of atoms)')
    parser.add_argument('--commitment-cost', type=float, default=0.25)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)  # Reduce to 50 epochs (still plenty)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip-length', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--train-split', type=float, default=0.8)

    # Logging
    parser.add_argument('--save-every', type=int, default=5)  # Save more frequently in case of crashes
    parser.add_argument('--vis-every', type=int, default=5)
    
    args = parser.parse_args()
    
    main(args)
