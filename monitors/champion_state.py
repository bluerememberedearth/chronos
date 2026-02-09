import threading
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional
import pytesseract

logger = logging.getLogger(__name__)

class ChampionStateMonitor:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Threading
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        # State Storage
        # 0-4: Blue Team, 5-9: Red Team
        self.champion_states = [{} for _ in range(10)]
        
        # ROI Configuration (Approximate for 1080p Spectator HUD)
        # Left Sidebar (Blue) ~ x=0..100, y=200..800 ?
        # Right Sidebar (Red) ~ x=1820..1920, y=200..800 ?
        # This needs precise calibration based on the specific HUD skin (LPL vs Worlds).
        # For MVP, we define generic "slots".
        
        self.slots = self._define_slots()

    def _define_slots(self):
        """
        Defines the bounding boxes for the 10 champion sidebars (LPL Bottom Layout).
        Based on user screenshots: 
        - 5 rows stacked for Blue (Left-Center)
        - 5 rows stacked for Red (Right-Center)
        Returns list of (x, y, w, h) tuples.
        """
        slots = []
        
        # Approximate Geometry for 1080p LPL Stream
        # Blue Team Block
        # Starts after Player Window (Left) -> ~ x=300?
        # Height: Bottom ~200px? Let's assume y=850 start.
        # Spacing: ~40px per row.
        
        blue_x_start = 280
        blue_y_start = 860
        row_height = 42 
        row_width = 500 # Wide enough to capture Items + KDA
        
        for i in range(5):
             slots.append((blue_x_start, blue_y_start + (i * row_height), row_width, row_height)) 
             
        # Red Team Block
        # Starts after some gap? Or symmetrical?
        # Screenshot shows them quite close in the center?
        # Wait, Screenshot 0 (Tab view) shows them adjacent. Screenshot 1 (Live) shows them separated.
        # Let's target the Live View (Screenshot 1).
        # Live View: Red block starts further right ~ x=1140?
        
        red_x_start = 1140
        red_y_start = 860
        
        for i in range(5):
             slots.append((red_x_start, red_y_start + (i * row_height), row_width, row_height))
             
        return slots

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Champion State Monitor Started.")

    def update(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            # 1. Get Frame
            with self._lock:
                if self._latest_frame is None:
                    time.sleep(0.1)
                    continue
                # Make a distinct copy if we plan to do heavy cropping to avoid race conditions on the array buffer? 
                # Actually, slicing is safe, but let's be careful.
                frame_ref = self._latest_frame
            
            # 2. Iterate through all 10 champions
            # To save CPU, we might process only 1 or 2 champions per 'tick' or process all if fast enough.
            # Let's try processing all, but skip OCR if no change detected.
            
            for i, (x, y, w, h) in enumerate(self.slots):
                # Safety Clip
                x = max(0, x); y = max(0, y)
                
                roi = frame_ref[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # Analysis Logic (Simplistic)
                # Check for "Dead" (Grayscale / Darkened)
                # Check for "Ultimate Ready" (Green dot?)
                
                # Mean Intensity check for "Death"
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray)
                
                is_dead = mean_val < 50 # Heuristic
                
                self.champion_states[i] = {
                    "is_dead": bool(is_dead),
                    "mean_intensity": mean_val,
                    "last_updated": time.time()
                }
                
                # Optional: Deep OCR for Level / CS
                # if i == 0: # Debug prints for first champ
                #    pass

            # Update Rate: 2 Hz (Every 500ms) - Individual champ states don't change that fast
            time.sleep(0.5)

    def get_states(self) -> List[Dict]:
        """Returns the list of 10 champion states."""
        return self.champion_states

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
