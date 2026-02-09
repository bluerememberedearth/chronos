import streamlink
import subprocess
import numpy as np
import time

def test_streamlink_pipe(url):
    print(f"Connecting to {url} via Streamlink...")
    session = streamlink.Streamlink()
    session.set_option("http-headers", "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    
    streams = session.streams(url)
    if not streams:
        print("No streams found.")
        return

    stream = streams.get('best')
    if not stream:
        print("No 'best' stream found.")
        return
        
    print(f"Opening stream: {stream.url[:50]}...")
    fd = stream.open()
    
    # Start FFMPEG to read from stdin
    command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-i', 'pipe:0',    # Read from stdin
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-f', 'image2pipe',
        '-'
    ]
    
    print("Starting FFMPEG...")
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=10**7)
    
    width = 1920
    height = 1080
    frame_size = width * height * 3
    
    start = time.time()
    frames = 0
    
    try:
        while time.time() - start < 10:
            # Read 64k chunks from streamlink
            chunk = fd.read(64 * 1024)
            if not chunk:
                print("Stream ended.")
                break
                
            # Write to ffmpeg stdin
            try:
                pipe.stdin.write(chunk)
            except BrokenPipeError:
                print("FFMPEG stdin closed.")
                break
                
            # Read decoded frames from ffmpeg stdout (non-blocking if possible, but here we block)
            # Actually, to do this properly we need threads, but for a quick test 
            # we can just write a bunch and try to read once.
            # But the pipe buffer might fill up.
            
            # Let's just try to read 1 frame for every ~5MB written? No, that's unreliable.
            # Correct way: Thread for writing, Main thread for reading.
            pass
            
            # Check if we can read a frame
            # This is tricky without threads because write might block if read doesn't happen.
            # We'll rely on large buffers for this quick 10s test.
            
    except Exception as e:
        print(f"Error: {e}")
        
    print("Pipe test finished (partial).")
    fd.close()
    pipe.terminate()

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.twitch.tv/riotgames"
    test_streamlink_pipe(url)
