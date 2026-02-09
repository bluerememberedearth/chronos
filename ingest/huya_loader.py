import subprocess
import logging
import numpy as np
import time
import shutil
import threading
from threading import Lock

# Safe import for streamlink
try:
    import streamlink
    STREAMLINK_AVAILABLE = True
except ImportError:
    STREAMLINK_AVAILABLE = False

logger = logging.getLogger(__name__)

class HuyaStreamLoader:
    def __init__(self, url="https://www.huya.com/lpl", quality="best"):
        self.url = url
        self.quality = quality
        self.pipe = None
        self.width = 1920  # LPL Source is typically 1080p
        self.height = 1080
        self.frame_size = self.width * self.height * 3 # BGR24
        
        # Threading state
        self._latest_frame = None
        self.frames_read = 0
        self._lock = Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        # Check for ffmpeg
        if not shutil.which("ffmpeg"):
            logger.error("FFmpeg not found! Please install it (e.g., sudo apt-get install ffmpeg).")
            raise RuntimeError("FFmpeg not found")

    def _get_stream_url(self):
        # Prefer yt-dlp as it worked in diagnostics where Streamlink failed
        logger.info(f"Resolving stream URL for {self.url} via yt-dlp...")
        return self._resolve_with_ytdlp()
            
    def _unused_streamlink_resolution(self):
        """Legacy Streamlink resolution (requires headers)"""
        if not STREAMLINK_AVAILABLE:
            raise ImportError("Streamlink is required for Huya ingestion.")
            
        logger.info(f"Resolving stream URL for {self.url} via Streamlink...")
        session = streamlink.Streamlink()
        session.set_option("http-headers", "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        
        try:
            streams = session.streams(self.url)
            if not streams:
                logger.warning(f"No streams found via Streamlink for {self.url}. Attempting yt-dlp fallback...")
                return self._resolve_with_ytdlp()
            
            if self.quality not in streams:
                logger.warning(f"Quality '{self.quality}' not found, falling back to best available.")
                url = streams['best'].url if 'best' in streams else next(iter(streams.values())).url
                return url
                
            return streams[self.quality].url
        except Exception as e:
            logger.warning(f"Streamlink extraction failed: {e}. Attempting yt-dlp fallback...")
            return self._resolve_with_ytdlp()

    def _resolve_with_ytdlp(self):
        """Fallback: Use yt-dlp to extract URL (works for VODs)."""
        if not shutil.which("yt-dlp"):
            logger.error("yt-dlp not found! Please install it (pip install yt-dlp).")
            return None
            
        try:
            # Run yt-dlp -g [url]
            cmd = ['yt-dlp', '-g', self.url]
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                # yt-dlp might return video_url\naudio_url. We make sure to take the m3u8.
                urls = result.stdout.strip().split('\n')
                # Usually the first one is video or combined
                if urls:
                    logger.info("Url resolved via yt-dlp.")
                    return urls[0]
            else:
                logger.error(f"yt-dlp failed: {result.stderr}")
        except Exception as e:
            logger.error(f"yt-dlp execution error: {e}")
        
        return None

    def start(self):
        if self._thread and self._thread.is_alive():
             logger.warning("Stream loader already running.")
             return

        import os
        self.is_local_file = os.path.exists(self.url)
        
        if self.is_local_file:
            logger.info(f"Using local file: {self.url}")
            # Use OpenCV for local files
            import cv2
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                logger.error("Failed to open local file with OpenCV.")
                return
        else:
            # Use Streamlink -> FFMpeg pipe for live streams
            logger.info(f"Starting Streamlink pipeline for {self.url}...")
            
            # 1. Start Streamlink process
            # Note: We use the CLI via subprocess to avoid Python API threading issues
            sl_cmd = ["python", "-m", "streamlink", "--stdout", self.url, "best"]
            self.sl_process = subprocess.Popen(
                sl_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE # Capture stderr to avoid console spam, or use DEVNULL
            )
            
            # 2. Start FFMpeg process reading from stdin
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-loglevel', 'error',
                '-i', 'pipe:0',    # Read from stdin (which will be streamlink's stdout)
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
            
            # Essential: Allow sl_process to receive SIGPIPE if ffmpeg exits
            self.sl_process.stdout.close() 

        # Start background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_frames, daemon=True)
        self._thread.start()
        logger.info("Huya Stream Loader Started.")

    def _update_frames(self):
        """Background thread to read frames."""
        import cv2
        
        while not self._stop_event.is_set():
            if self.is_local_file:
                # Local file logic
                if self.cap is None: break
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Local file loop: restarting.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                with self._lock:
                    self._latest_frame = frame
                    self.frames_read += 1
                
                # Control playback speed for local file
                time.sleep(1/30) 
                
            else:
                # Live stream logic (reading from ffmpeg stdout)
                if self.ffmpeg_process is None: break
                
                raw_image = self.ffmpeg_process.stdout.read(self.frame_size)
                
                if len(raw_image) != self.frame_size:
                    logger.warning("Incomplete frame read. Stream might be re-buffering or ended.")
                    # Reconnect logic could go here
                    break
                    
                image = np.frombuffer(raw_image, dtype='uint8').reshape((self.height, self.width, 3))
                
                with self._lock:
                    self._latest_frame = image
                    self.frames_read += 1

        logger.info("Stream thread exiting.")
        if hasattr(self, 'cap') and self.cap:
             self.cap.release()
        
        # Cleanup subprocesses
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
        """
        Returns the MOST RECENT fully decoded frame.
        """
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
        
        # Cleanup is handled in thread exit, but double check
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            self.ffmpeg_process.kill()
        if hasattr(self, 'sl_process') and self.sl_process:
            self.sl_process.kill()
            
        self._latest_frame = None
