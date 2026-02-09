import threading
import time
import logging
import cv2
import numpy as np
from typing import Dict, List
import pytesseract

logger = logging.getLogger(__name__)

class ScoreboardMonitor:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.width, self.height = resolution
        
        # ROI Definitions (Approximate for 1080p client)
        # Top center clock/score area
        self.roi_x = int(self.width * 0.40)
        self.roi_y = 0
        self.roi_w = int(self.width * 0.20)
        self.roi_h = int(self.height * 0.10)
        
        # Threading
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        self.state = {
            "time": "00:00",
            "kills_blue": 0,
            "kills_red": 0,
            "gold_diff": 0 # This usually requires "Tab" press or derivation
        }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Scoreboard Monitor Started.")

    def update(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                if self._latest_frame is None:
                    time.sleep(0.1)
                    continue
                # ROI Crop
                roi = self._latest_frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w].copy()

            if roi.size == 0:
                continue

            # Preprocessing for OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Threshold to isolate white text (scores/time)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Run OCR (Tesseract)
            # config: searching for digits/time patterns
            try:
                data = pytesseract.image_to_string(thresh, config='--psm 6')
                self._parse_ocr(data)
            except Exception as e:
                # logger.warning(f"OCR Failed: {e}")
                pass
            
            # OCR is slow, so we don't need to sleep much, but let's be safe
            time.sleep(0.5) # Update every 500ms is plenty for finding game time

    def _parse_ocr(self, text: str):
        # Very distinct logic needed here depends on font/layout
        # For now, just logging clean results
        clean = text.strip()
        if len(clean) > 3:
            # logger.info(f"Scoreboard OCR: {clean}")
            # Update internal state (fusion source)
            self.state["raw_text"] = clean

    def get_state(self):
        return self.state

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
