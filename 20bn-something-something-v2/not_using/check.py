import cv2
import numpy as np
from pathlib import Path

def create_comparison_video(original_path, event_path, output_path):
    """Create side-by-side comparison of original and event video"""
    
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_event = cv2.VideoCapture(str(event_path))
    
    # Get dimensions
    orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    event_w = int(cap_event.get(cv2.CAP_PROP_FRAME_WIDTH))
    event_h = int(cap_event.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS) or 12.0
    
    print(f"Original: {orig_w}x{orig_h}, Event: {event_w}x{event_h}")
    
    # Create side-by-side output
    out_w = orig_w + event_w
    out_h = max(orig_h, event_h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h), isColor=True)
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_event, frame_event = cap_event.read()
        
        if not ret_orig or not ret_event:
            break
        
        # Resize event frame to match original height if needed
        if event_h != orig_h:
            frame_event = cv2.resize(frame_event, (event_w, orig_h))
        
        # Create side-by-side frame
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:orig_h, :orig_w] = frame_orig
        combined[:orig_h, orig_w:orig_w+event_w] = frame_event
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Events", (orig_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        writer.write(combined)
    
    cap_orig.release()
    cap_event.release()
    writer.release()
    print(f"Saved comparison to {output_path}")

# Compare first video
original_dir = Path("20bn-something-something-v2/filtered_videos")
event_dir = Path("events_output")

orig_video = next(original_dir.glob("*.webm"))
event_video = event_dir / f"{orig_video.stem}_events.mp4"

if event_video.exists():
    create_comparison_video(orig_video, event_video, "comparison.mp4")
else:
    print(f"Event video not found: {event_video}")