"""
Generate Synthetic Events from RGB Frames
Converts RGB video frames into Event Volumes using frame differencing
Mimics v2e behavior without the dependency
"""

import cv2
import numpy as np
from pathlib import Path
import json


class EventGenerator:
    """Generate event volumes from video frames using frame differencing (v2e-style)"""
    
    def __init__(self, threshold=0.2, contrast_threshold=0.2, refractory_period=0):
        """
        Args:
            threshold: Brightness change threshold (log scale)
            contrast_threshold: Contrast threshold for event generation
            refractory_period: Minimum time between events at same pixel (frames)
        """
        self.threshold = threshold
        self.contrast_threshold = contrast_threshold
        self.refractory_period = refractory_period
        self.last_event_time = None
    
    def load_video(self, video_path, target_shape=None):
        """
        Load video frames from a video file
        
        Args:
            video_path: Path to video file
            target_shape: Optional (H, W) to resize frames
            
        Returns:
            frames: numpy array of shape (T, H, W, 3) with values in [0, 1]
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get original dimensions
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Store original dimensions BEFORE any processing
            if not frames:
                self._orig_width = orig_width
                self._orig_height = orig_height
            
            # Convert BGR to grayscale (v2e uses intensity)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize if needed
            if target_shape:
                frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Avoid log(0) issues
            frame = np.clip(frame, 1e-6, 1.0)
            
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def load_frames_from_dir(self, frames_dir, target_shape=None, pattern="*.png"):
        """
        Load frames from a directory
        
        Args:
            frames_dir: Directory containing frame images
            target_shape: Optional (H, W) to resize frames
            pattern: Glob pattern for frame files
            
        Returns:
            frames: numpy array of shape (T, H, W)
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(frames_dir.glob(pattern))
        frames = []
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"Warning: Could not read {frame_file}")
                continue
            
            # Resize if needed
            if target_shape:
                frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
            
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            frame = np.clip(frame, 1e-6, 1.0)
            
            frames.append(frame)
        
        return np.array(frames)
    
    def generate_events(self, frames):
        """
        Generate event volumes from consecutive frame differences (v2e-style log change)
        
        Args:
            frames: numpy array of shape (T, H, W) with values in [0, 1]
            
        Returns:
            events: numpy array of shape (T-1, H, W, 2) 
                    Channel 0: ON events (positive change)
                    Channel 1: OFF events (negative change)
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to generate events")
        
        T, H, W = frames.shape
        events = np.zeros((T - 1, H, W, 2), dtype=np.uint8)
        
        # Initialize refractory period tracking
        if self.refractory_period > 0:
            self.last_event_time = np.full((H, W), -self.refractory_period, dtype=np.int32)
        
        for t in range(1, T):
            # Calculate log intensity change (like v2e)
            log_change = np.log(frames[t] + 1e-6) - np.log(frames[t-1] + 1e-6)
            
            # ON events (brightness increase)
            on_mask = log_change > self.threshold
            
            # OFF events (brightness decrease)
            off_mask = log_change < -self.threshold
            
            # Apply refractory period if enabled
            if self.refractory_period > 0:
                valid_pixels = (t - self.last_event_time) >= self.refractory_period
                on_mask = on_mask & valid_pixels
                off_mask = off_mask & valid_pixels
                
                # Update last event times
                event_pixels = on_mask | off_mask
                self.last_event_time[event_pixels] = t
            
            events[t - 1, :, :, 0] = on_mask.astype(np.uint8)
            events[t - 1, :, :, 1] = off_mask.astype(np.uint8)
        
        return events
    
    def save_events_as_video(self, events, output_path, fps=30, orig_width=None, orig_height=None):
        """
        Save event volumes as a video file (visualizes ON/OFF events)
        
        Args:
            events: numpy array of shape (T, H, W, 2)
            output_path: Path to save video
            fps: Frames per second for output video
            orig_width: Original video width (for exact dimension match)
            orig_height: Original video height (for exact dimension match)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        T, H, W = events.shape[:3]
        
        # Use original dimensions if provided
        out_width = orig_width if orig_width else W
        out_height = orig_height if orig_height else H
        
        # Use H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height), isColor=True)
        
        if not writer.isOpened():
            print(f"Warning: Could not open writer, trying mp4v codec")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height), isColor=True)
        
        for t in range(T):
            # Create BGR visualization: ON=red, OFF=blue
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:, :, 2] = events[t, :, :, 0] * 255  # ON events in red channel
            frame[:, :, 0] = events[t, :, :, 1] * 255  # OFF events in blue channel
            
            # Resize to exact original dimensions if needed
            if (H, W) != (out_height, out_width):
                frame = cv2.resize(frame, (out_width, out_height))
            
            writer.write(frame)
        
        writer.release()
        print(f"Saved event video to {output_path} ({out_width}x{out_height})")
    
    def save_events_as_frames(self, events, output_dir, prefix="event"):
        """
        Save each event frame as a PNG image
        
        Args:
            events: numpy array of shape (T, H, W, 2)
            output_dir: Directory to save frames
            prefix: Prefix for filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        T = events.shape[0]
        
        for t in range(T):
            # Save ON/OFF channels separately
            on_frame = (events[t, :, :, 0] * 255).astype(np.uint8)
            off_frame = (events[t, :, :, 1] * 255).astype(np.uint8)
            
            cv2.imwrite(str(output_dir / f"{prefix}_on_{t:06d}.png"), on_frame)
            cv2.imwrite(str(output_dir / f"{prefix}_off_{t:06d}.png"), off_frame)
        
        print(f"Saved {T} event frames to {output_dir}")
    
    def save_events_npz(self, events, output_path):
        """
        Save event volumes as compressed NPZ file
        
        Args:
            events: numpy array of shape (T, H, W, 2)
            output_path: Path to save .npz file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), events=events)
        print(f"Saved event volume to {output_path}")


def process_video_to_events(video_path, output_dir, threshold=0.2, fps=None):
    generator = EventGenerator(threshold=threshold)

    # Get original dimensions first
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps is None:
        fps = src_fps

    frames = generator.load_video(video_path)
    events = generator.generate_events(frames)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    # Pass original dimensions to preserve exact size
    generator.save_events_as_video(
        events, 
        output_dir / f"{video_name}_events.mp4", 
        fps=fps,
        orig_width=orig_width,
        orig_height=orig_height
    )
    generator.save_events_npz(events, output_dir / f"{video_name}_events.npz")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic events from videos (v2e-style)")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--video-dir", type=str, 
                       help="Process all videos in a directory")
    parser.add_argument("--output", type=str, default="events_output",
                       help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.2,
                       help="Log brightness change threshold")
    parser.add_argument("--fps", type=int, default=None,
                       help="Output video FPS")
    
    args = parser.parse_args()
    
    if args.video:
        process_video_to_events(args.video, args.output, args.threshold, args.fps)
    
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        for video_file in sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.webm")):
            try:
                process_video_to_events(
                    video_file, 
                    Path(args.output) / video_file.stem,
                    args.threshold, 
                    args.fps
                )
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
    
    else:
        parser.print_help()