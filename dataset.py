"""
PyTorch Dataset for Event Volumes and RGB Videos
Loads event data (.npz) and corresponding RGB videos for training VQ-VAE
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json


class EventVideoDataset(Dataset):
    """
    Dataset that loads event volumes and corresponding RGB videos
    
    Returns:
        event_clip: Event volume (T, H, W, 2) - ON/OFF channels
        rgb_clip: RGB video frames (T, H, W, 3) for visualization
        metadata: Dict with video info (name, fps, etc.)
    """
    
    def __init__(
        self, 
        video_dir,
        event_dir,
        clip_length=16,
        stride=8,
        transform=None,
        load_rgb=True,
        target_size=(240, 320)  # Add this parameter
    ):
        """
        Args:
            video_dir: Directory containing original RGB videos
            event_dir: Directory containing event .npz files
            clip_length: Number of frames per clip (T dimension)
            stride: Stride for sliding window over video
            transform: Optional transforms to apply
            load_rgb: Whether to load RGB videos (set False for faster training)
            target_size: (H, W) to resize all clips to for batching
        """
        self.video_dir = Path(video_dir)
        self.event_dir = Path(event_dir)
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.load_rgb = load_rgb
        self.target_size = target_size  # Add this
                
        # Find all event files
        self.event_files = sorted(list(self.event_dir.rglob("*_events.npz")))
        
        # Build index of clips (video_id, start_frame)
        self.clips = []
        for event_file in self.event_files:
            # Load event data to get number of frames
            data = np.load(event_file)
            events = data['events']
            T = events.shape[0]
            
            # Create clips with sliding window
            for start in range(0, T - clip_length + 1, stride):
                self.clips.append({
                    'event_file': event_file,
                    'start_frame': start,
                    'video_name': event_file.stem.replace('_events', '')
                })
        
        print(f"Loaded {len(self.event_files)} videos, {len(self.clips)} clips")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        
        # Load event clip
        event_file = clip_info['event_file']
        start = clip_info['start_frame']
        end = start + self.clip_length
        
        data = np.load(event_file)
        events = data['events'][start:end]  # Shape: (T, H, W, 2)
        
        # Resize events to target size
        if self.target_size:
            T, H, W, C = events.shape
            resized_events = np.zeros((T, self.target_size[0], self.target_size[1], C), dtype=events.dtype)
            for t in range(T):
                for c in range(C):
                    resized_events[t, :, :, c] = cv2.resize(
                        events[t, :, :, c], 
                        (self.target_size[1], self.target_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
            events = resized_events
        
        # Convert to torch tensor and permute to (C, T, H, W) for 3D Conv
        event_clip = torch.from_numpy(events).float()
        event_clip = event_clip.permute(3, 0, 1, 2)  # (2, T, H, W)
        
        # Optionally load RGB video
        rgb_clip = None
        if self.load_rgb:
            video_name = clip_info['video_name']
            # Try different video extensions
            video_file = None
            for ext in ['.webm', '.mp4', '.avi']:
                candidate = self.video_dir / f"{video_name}{ext}"
                if candidate.exists():
                    video_file = candidate
                    break
            
            if video_file and video_file.exists():
                rgb_clip = self._load_rgb_clip(video_file, start, end)
            else:
                # Return dummy RGB if file not found
                H, W = self.target_size if self.target_size else events.shape[1:3]
                rgb_clip = torch.zeros(3, self.clip_length, H, W)
                
        # Apply transforms if any
        if self.transform:
            event_clip = self.transform(event_clip)
            if rgb_clip is not None:
                rgb_clip = self.transform(rgb_clip)
        
        metadata = {
            'video_name': clip_info['video_name'],
            'start_frame': start,
            'clip_idx': idx
        }
        
        if rgb_clip is not None:
            return event_clip, rgb_clip, metadata
        else:
            return event_clip, metadata
    
    def _load_rgb_clip(self, video_file, start_frame, end_frame):
        """Load RGB frames from video file"""
        cap = cv2.VideoCapture(str(video_file))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if start_frame <= frame_idx < end_frame:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                if self.target_size:
                    frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            frame_idx += 1
            
            if frame_idx >= end_frame:
                break
        
        cap.release()
        
        # Pad if needed
        target_h, target_w = self.target_size if self.target_size else (240, 360)
        while len(frames) < self.clip_length:
            frames.append(np.zeros((target_h, target_w, 3), dtype=np.float32))
        
        # Convert to tensor (C, T, H, W)
        rgb_clip = np.stack(frames[:self.clip_length])  # (T, H, W, 3)
        rgb_clip = torch.from_numpy(rgb_clip).float()
        rgb_clip = rgb_clip.permute(3, 0, 1, 2)  # (3, T, H, W)
        
        return rgb_clip

def get_dataloaders(
    video_dir,
    event_dir,
    batch_size=8,
    clip_length=16,
    stride=8,
    num_workers=4,
    train_split=0.8,
    load_rgb=True,
    target_size=(240, 320)  # Add this
):
    """
    Create train and validation dataloaders
    
    Args:
        video_dir: Directory with RGB videos
        event_dir: Directory with event .npz files
        batch_size: Batch size
        clip_length: Frames per clip
        stride: Sliding window stride
        num_workers: DataLoader workers
        train_split: Fraction of data for training
        load_rgb: Whether to load RGB videos
        target_size: (H, W) to resize all clips
        
    Returns:
        train_loader, val_loader
    """
    dataset = EventVideoDataset(
        video_dir=video_dir,
        event_dir=event_dir,
        clip_length=clip_length,
        stride=stride,
        load_rgb=load_rgb,
        target_size=target_size
    )
        
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"Train: {len(train_dataset)} clips, Val: {len(val_dataset)} clips")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    video_dir = "20bn-something-something-v2/filtered_videos"
    event_dir = "events_output"
    
    # Create dataset
    dataset = EventVideoDataset(
        video_dir=video_dir,
        event_dir=event_dir,
        clip_length=16,
        stride=8,
        load_rgb=True
    )
    
    print(f"\nDataset size: {len(dataset)} clips")
    
    # Test loading a sample
    event_clip, rgb_clip, metadata = dataset[0]
    
    print(f"\nSample clip:")
    print(f"  Event shape: {event_clip.shape}")  # Should be (2, 16, H, W)
    print(f"  RGB shape: {rgb_clip.shape}")      # Should be (3, 16, H, W)
    print(f"  Metadata: {metadata}")
    
    # Test dataloader
    train_loader, val_loader = get_dataloaders(
        video_dir=video_dir,
        event_dir=event_dir,
        batch_size=4,
        clip_length=16
    )
    
    print(f"\nDataLoader test:")
    for batch in train_loader:
        if len(batch) == 3:
            events, rgb, meta = batch
            print(f"  Event batch: {events.shape}")  # (B, 2, 16, H, W)
            print(f"  RGB batch: {rgb.shape}")        # (B, 3, 16, H, W)
        else:
            events, meta = batch
            print(f"  Event batch: {events.shape}")
        break