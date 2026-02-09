import argparse
import cv2
import time
import os
import sys

def resolve_huya_url(url):
    """
    Uses yt-dlp to resolve a Huya VOD/Live URL to a direct streamable link.
    """
    import shutil
    import subprocess
    
    import shutil
    import subprocess
    import sys
    import os

    yt_dlp_cmd = "yt-dlp"
    if not shutil.which(yt_dlp_cmd):
        # Check if in same dir as python executable (venv)
        potential_path = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
        if os.path.exists(potential_path):
            yt_dlp_cmd = potential_path
        else:
            print("Error: yt-dlp not found in PATH or venv. Cannot resolve Huya URL.")
            return None
        
    try:
        print(f"Resolving URL via {yt_dlp_cmd}: {url}")
        cmd = [yt_dlp_cmd, '-g', url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            urls = result.stdout.strip().split('\n')
            if urls:
                final_url = urls[0] # Usually the first is the video stream
                print(f"Resolved to: {final_url[:50]}...")
                return final_url
    except Exception as e:
        print(f"Failed to resolve URL: {e}")
    return None

def capture_samples(source, output_dir, count, interval, start_time=0):
    """
    Captures 'count' frames from 'source' with 'interval' seconds between them.
    Starts capture at 'start_time' seconds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Handle numeric source (camera index)
    if str(source).isdigit():
        source = int(source)
    elif "huya.com" in str(source):
        resolved = resolve_huya_url(source)
        if resolved:
            source = resolved
        else:
            print("Could not resolve Huya URL. Attempting direct open (likely to fail)...")
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    if start_time > 0:
        print(f"Seeking to {start_time} seconds (this might take a while for remote streams)...")
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        # Verify seek status?
        # pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        # print(f"Seeked to: {pos/1000:.2f}s")

    print(f"Capturing {count} frames from {source} to {output_dir}...")
    
    captured = 0
    last_processed_time = time.time()
    
    try:
        while captured < count:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or read error.")
                break

            # Save immediately since we seeked to start_time
            # For multiple frames, we might want to space them out by 'interval'
            # But relying on stream time is tricky. Let's just grab sequential frames if interval is small
            # Or assume stream is playing at 1x speed and use time.sleep logic?
            # For VOD analysis, we usually want specific snapshots.
            
            filename = f"sample_seek{start_time}_{captured:03d}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            captured += 1
            
            # Simple skip logic for next frame if count > 1
            if count > 1 and interval > 0:
                 # Skip frames to simulate interval (assuming 30fps)
                 frames_to_skip = int(interval * 30)
                 for _ in range(frames_to_skip):
                     cap.grab()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print(f"Finished. Captured {captured} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture frames from video/stream")
    parser.add_argument("--source", required=True, help="Video source (URL or file)")
    parser.add_argument("--out", default="samples", help="Output directory")
    parser.add_argument("--count", type=int, default=10, help="Number of frames to capture")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between frames (seconds)")
    parser.add_argument("--start_time", type=int, default=0, help="Start time in seconds (for seeking)")
    
    args = parser.parse_args()
    
    try:
        capture_samples(args.source, args.out, args.count, args.interval, args.start_time)
    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")
