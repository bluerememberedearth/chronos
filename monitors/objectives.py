import threading
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional
import pytesseract
import re

logger = logging.getLogger(__name__)

class ObjectiveMonitor:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Threading
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        self.state = {
            "next_dragon_time": None, # Seconds remaining
            "next_baron_time": None,
            "next_herald_time": None
        }
        
        # ROI Configuration (LPL Top Left)
        # Screenshot shows timers roughly at x=30..300, y=30..80
        # There are usually two timers stacked or side-by-side.
        self.roi_x = 20
        self.roi_y = 20
        self.roi_w = 350
        self.roi_h = 100

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Objective Monitor Started.")

    def update(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                if self._latest_frame is None:
                    time.sleep(0.1)
                    continue
                roi = self._latest_frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w].copy()

            if roi.size == 0:
                continue

            # OCR Preprocessing
            # Timers are usually white text on dark background
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Invert? Usually no need for tesseract if contrast is good
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # OCR
            try:
                # psm 6 = Assume a single uniform block of text
                text = pytesseract.image_to_string(thresh, config='--psm 6')
                self._parse_timers(text)
            except Exception:
                pass
            
            # Update Rate: Low (1Hz) - Timers tick slowly
            time.sleep(1.0)

    def _parse_timers(self, text: str):
        """
        Parses strings like '4:03' or '1:20'.
        Refines state.
        """
        # Regex for M:SS
        matches = re.findall(r'(\d{1,2}:\d{2})', text)
        if matches:
            # logger.info(f"Objective Timers Detected: {matches}")
            # Heuristic assignment:
            # Usually strict ordering: Baron/Herald (Top), Dragon (Bottom)?
            # Or Left/Right.
            # Lacking context, we just store "upcoming_events" list
            self.state["detected_timers"] = matches
            
            # TODO: Add icon recognition (template match) to assign label to timer
            # For now, raw timers are better than nothing.
            
    def get_state(self):
        return self.state

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
