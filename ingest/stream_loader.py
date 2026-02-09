import sys
import subprocess
import logging
import numpy as np
import time
import shutil
import threading
from threading import Lock
import os

# Safe import for streamlink
try:
    import streamlink
    STREAMLINK_AVAILABLE = True
except ImportError:
    STREAMLINK_AVAILABLE = False

logger = logging.getLogger(__name__)

class GenericStreamLoader:
    def __init__(self, url, quality="best"):
        self.url = url
        self.quality = quality
        self.width = 1920
        self.height = 1080
        self.frame_size = self.width * self.height * 3 # BGR24
        
        # Threading state
        self._latest_frame = None
        self.frames_read = 0
        self._lock = Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.sl_process = None
        self.ffmpeg_process = None
        self.cap = None
        
        # Check for ffmpeg
        if not shutil.which("ffmpeg"):
            logger.error("FFmpeg not found! Please install it.")
            raise RuntimeError("FFmpeg not found")

    def start(self):
        if self._thread and self._thread.is_alive():
             logger.warning("Stream loader already running.")
             return

        self.is_local_file = os.path.exists(self.url)
        
        if self.is_local_file:
            logger.info(f"Using local file: {self.url}")
            import cv2
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                logger.error("Failed to open local file with OpenCV.")
                return
        else:
            # Live stream logic via Streamlink -> FFMpeg
            logger.info(f"Starting Streamlink pipeline for {self.url}...")
            
            # 1. Start Streamlink process
            sl_cmd = [
                sys.executable, "-m", "streamlink", 
                "--stdout", 
                "--twitch-disable-ads", 
                self.url, 
                self.quality
            ]
            
            self.sl_process = subprocess.Popen(
                sl_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # 2. Start FFMpeg process reading from stdin
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-loglevel', 'error',
                '-i', 'pipe:0',
                '-s', f'{self.width}x{self.height}',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-an',
                '-f', 'image2pipe',
                '-'
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self.sl_process.stdout,
                stdout=subprocess.PIPE,
                bufsize=10**7
            )
            
            # Allow sl_process to receive SIGPIPE
            self.sl_process.stdout.close() 

        # Start background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_frames, daemon=True)
        self._thread.start()
        logger.info("Stream Loader Started.")

    def _update_frames(self):
        """Background thread to read frames."""
        import cv2
        
        while not self._stop_event.is_set():
            if self.is_local_file:
                if self.cap is None: break
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Local file loop: restarting.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                with self._lock:
                    self._latest_frame = frame
                    self.frames_read += 1
                time.sleep(1/30) 
            else:
                if self.ffmpeg_process is None: break
                
                try:
                    raw_image = self.ffmpeg_process.stdout.read(self.frame_size)
                    
                    if len(raw_image) != self.frame_size:
                        logger.warning("Incomplete frame read. Stream might be re-buffering or ended.")
                        # Could implement reconnect logic here
                        break
                        
                    image = np.frombuffer(raw_image, dtype='uint8').reshape((self.height, self.width, 3))
                    
                    with self._lock:
                        self._latest_frame = image
                        self.frames_read += 1
                except Exception as e:
                    logger.error(f"Error reading frame: {e}")
                    break

        logger.info("Stream thread exiting.")
        self._cleanup()

    def _cleanup(self):
        if hasattr(self, 'cap') and self.cap:
             self.cap.release()
        
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=1)
            except:
                self.ffmpeg_process.kill()
                
        if hasattr(self, 'sl_process') and self.sl_process:
            self.sl_process.terminate()
            try:
                self.sl_process.wait(timeout=1)
            except:
                self.sl_process.kill()

    def get_frame(self):
        """Returns the MOST RECENT fully decoded frame."""
        if self._thread is None or not self._thread.is_alive():
             try:
                 self.start()
                 time.sleep(2)
             except Exception as e:
                 logger.error(f"Failed to auto-start stream: {e}")
                 return None

        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy(), self.frames_read 

    def close(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._cleanup()
