import cv2
import time
import subprocess
import shutil

def get_stream_url(url):
    yt_dlp = shutil.which("yt-dlp")
    cmd = [yt_dlp, '-g', url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().split('\n')[0]
    return None

def test_opencv(url):
    print(f"Resolving {url}...")
    stream_url = get_stream_url(url)
    if not stream_url:
        print("Failed to resolve URL")
        return

    print(f"Opening {stream_url[:50]}...")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Failed to open stream with CV2")
        return

    print("Stream opened! Recording to mock_stream.mp4...")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    out = cv2.VideoWriter('mock_stream.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    
    start = time.time()
    frames = 0
    
    while time.time() - start < 30: # Record 30 seconds
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break
            
        out.write(frame)
        frames += 1
        if frames % 30 == 0:
            print(f"Recorded {frames} frames...")
    
    cap.release()
    out.release()
    print(f"Success! Recorded {frames} frames in {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_opencv("https://www.huya.com/lpl")
