import threading
import time
import logging
import cv2
import numpy as np
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class MinimapMonitor:
    def __init__(self, resolution=(1920, 1080)):
        """
        Args:
            resolution: Resolution of the full game frame.
        """
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Define ROI for Minimap (Approximate for 1080p)
        # Bottom Right Corner. 
        # Tuning needed for exact UI scale, but roughly:
        # x: 1650 -> 1920 (Width ~270)
        # y: 810 -> 1080 (Height ~270)
        self.roi_x = int(self.width * 0.86) 
        self.roi_y = int(self.height * 0.78) # Adjusted for LPL: Map is very low in corner
        self.roi_w = self.width - self.roi_x
        self.roi_h = self.height - self.roi_y
        
        # Threading
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        # Detection State
        self.last_activity_time = 0
        self.alerts = []

    def start(self):
        if self._thread and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Minimap Monitor Started.")

    def update(self, frame: np.ndarray):
        """
        Push a new full frame to the monitor.
        In a zero-copy optimization, we'd pass a shared memory buffer.
        Here we just store the reference (Python GIL protects access usually, but we use lock).
        """
        with self._lock:
            self._latest_frame = frame

    def _monitor_loop(self):
        """
        Continuous loop to process the latest available frame.
        """
        prev_roi_gray = None
        
        while not self._stop_event.is_set():
            # 1. Get Frame
            with self._lock:
                if self._latest_frame is None:
                    time.sleep(0.1)
                    continue
                # Create a view/copy of ROI
                # Important: Minimap is at [y:y+h, x:x+w]
                roi_color = self._latest_frame[self.roi_y:, self.roi_x:].copy()
                
            if roi_color.size == 0:
                logger.warning(f"Invalid ROI: {self.roi_x},{self.roi_y} for frame {self.width}x{self.height}")
                continue
            
            # 2. Process ROI (Motion Detection)
            # Simple approach: Frame Difference
            gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_roi_gray is None:
                prev_roi_gray = gray
                continue
                
            frame_delta = cv2.absdiff(prev_roi_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate to fill holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            activity_score = 0
            for c in contours:
                if cv2.contourArea(c) < 50: # Ignore noise
                    continue
                activity_score += cv2.contourArea(c)
            
            # Logic: If high activity -> Potential Fight / Gank
            if activity_score > 500:
                self._trigger_alert("high_activity", activity_score)
            
            prev_roi_gray = gray
            
            # Sleep to yield CPU - Minimap doesn't update at 1000hz
            time.sleep(0.05) # 20 FPS monitoring

    def _trigger_alert(self, type: str, score: float):
        # logger.info(f"Minimap Alert: {type} (Score: {score})")
        self.alerts.append({"type": type, "score": score, "time": time.time()})

    def get_alerts(self) -> List[Dict]:
        """Returns and clears current alerts."""
        alerts = list(self.alerts)
        self.alerts.clear()
        return alerts

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
