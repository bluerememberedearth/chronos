import time
import logging
from typing import Generator, Tuple, Optional, Any


# Graceful optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        uint8 = "uint8" # Define uint8 for compatibility
        def zeros(self, shape, dtype=None):
            # This method will only be called if NUMPY_AVAILABLE is True,
            # so this branch should ideally not be reached.
            # If it were, it would need to return something array-like.
            # For now, we'll return a placeholder that might cause issues
            # if array operations are attempted on it.
            logging.warning("Numpy not available - returning mock object for np.zeros")
            return "MOCK_ARRAY_PLACEHOLDER"
    np = MockNumpy()

try:
    from chronos_v3.ingest.huya_loader import HuyaStreamLoader
    HUYA_AVAILABLE = True
except ImportError:
    HUYA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoIngest:
    def __init__(self, source: str = "0", target_fps: int = 1):
        """
        Args:
            source: Path to video file, camera index, or Huya URL.
            target_fps: Frames per second to yield (for non-live sources or downsampling).
        """
        self.source = source
        self.target_fps = target_fps
        self._cap = None
        self._huya_loader = None
        self._stop_event = False
        
        self.use_huya = "huya.com" in source
        self.is_live_source = self.source.isdigit() or "rtmp" in str(self.source) or "rtsp" in str(self.source) or self.use_huya

        if self.use_huya and not HUYA_AVAILABLE:
            logger.error("Huya URL provided but imports failed (ffmpeg/streamlink).")
        
        if not CV2_AVAILABLE and not self.use_huya:
            logger.warning("OpenCV not found. Running in SIMULATION MODE.")
        if not NUMPY_AVAILABLE:
            logger.warning("Numpy not found. Using Mock data structures.")

    def start(self):
        """Opens the video source."""
        if self.use_huya:
             logger.info(f"Initializing Huya Stream: {self.source}")
             self._huya_loader = HuyaStreamLoader(url=self.source)
             self._huya_loader.start()
             return

        if not CV2_AVAILABLE:
            logger.info("Simulation mode: Source 'opened' (virtual).")
            return

        logger.info(f"Opening video source: {self.source}")
        # If source is a digit, treat as camera index
        if self.source.isdigit():
            self._cap = cv2.VideoCapture(int(self.source))
        else:
            self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            raise RuntimeError(f"Could not open source {self.source}")
            
        logger.info(f"Source opened. FPS: {self._cap.get(cv2.CAP_PROP_FPS)}")

    def stop(self):
        """Releases the video source."""
        self._stop_event = True
        if self._cap:
            self._cap.release()
        logger.info("Video source released.")

    def stream(self) -> Generator[Tuple[float, Any], None, None]:
        """
        Yields (timestamp, frame) tuples.
        """
        if CV2_AVAILABLE and not self._cap:
            self.start()

        # Determine wait time
        frame_interval = 1.0 / self.target_fps
        
        if CV2_AVAILABLE and self._cap:
            source_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0 or np.isnan(source_fps):
                 source_fps = 30
            skip_frames = max(1, int(source_fps / self.target_fps))
        else:
            skip_frames = 1

        last_yield_time = time.time()
        frame_count = 0
        
        # For file playback simulation: regulate speed to match source FPS (or target FPS)
        # If source is file, we want to simulate "live" arrival of frames.
        is_live_source = self.source.isdigit() or "rtmp" in str(self.source) or "rtsp" in str(self.source)
        
        while not self._stop_event:
            loop_start = time.time()
            
            # 1. Capture Frame
            # 1. Capture Frame
            video_time = 0.0
            
            if self.use_huya:
                # Expect tuple return now
                result = self._huya_loader.get_frame()
                if result is None:
                     frame = None
                else:
                     frame, frame_idx = result
                     video_time = frame_idx / 60.0 # LPL VODs are 60fps
                
                if frame is None:
                    # Reconnection logic
                    logger.warning("Huya stream lost. Attempting reconnect in 2s...")
                    time.sleep(2) 
                    try:
                        self._huya_loader.start()
                        continue
                    except Exception as e:
                        logger.error(f"Reconnect failed: {e}")
                        break # Or continue to retry indefinitely
            elif CV2_AVAILABLE and self._cap:
                ret, frame = self._cap.read()
                if not ret:
                    logger.info("End of stream reached.")
                    break
                # Approx video time for CV2 source
                video_time = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                # Simulation Mode
                # ... existing mock logic ...
                if NUMPY_AVAILABLE:
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    frame[500:580, 900:1020] = [0, 255, 0] 
                else:
                    frame = "MOCK_FRAME_DATA"
                video_time = time.time() # Mock time

            # 2. Throttling for "Real-Time" Simulation (File Sources)
            # Only apply this logic if NOT live huya and NOT live webcam (digit) and NOT rtmp
            # Re-evaluating is_live_source definition
            if not self.is_live_source and CV2_AVAILABLE and not self.use_huya:
                # ... existing file throttling ...
                pass 
                
            # Resize to Target Resolution (1080p) to standardize Monitors
            if frame is not None:
                h, w = frame.shape[:2]
                if w != 1920 or h != 1080:
                    frame = cv2.resize(frame, (1920, 1080))

            # 2. Yield Logic
            frame_count += 1
            if self.use_huya or not CV2_AVAILABLE or (frame_count % skip_frames == 0):
                yield video_time, frame
                
                # Enforce Timing HERE for the yielded frame
                if not is_live_source:
                   # expected duration since last yield
                   expected_gap = 1.0 / self.target_fps
                   actual_gap = time.time() - last_yield_time
                   if actual_gap < expected_gap:
                       time.sleep(expected_gap - actual_gap)
                
                last_yield_time = time.time()

