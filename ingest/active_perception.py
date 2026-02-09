"""
Active Perception Engine - Brain-Inspired Architecture

Inspired by human vision:
- Peripheral (Lizard Brain): CV-based, always on, detects CHANGE
- Foveal (Cortex): VLM-based, expensive, only triggered when needed

This conserves our 20 RPD per model limit by only calling VLM
when the CV layer detects something worth analyzing.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertType(Enum):
    MOTION_SPIKE = "motion_spike"
    SCOREBOARD_CHANGE = "scoreboard_change"
    HEALTH_DROP = "health_drop"
    OBJECTIVE_ACTIVE = "objective_active"
    STATE_TRANSITION = "state_transition"


@dataclass
class PerceptionAlert:
    """Alert from peripheral layer triggering foveal analysis."""
    alert_type: AlertType
    urgency: float  # 0-1, higher = more urgent
    region: Optional[tuple] = None  # (x, y, w, h) if localized
    context: Optional[str] = None


class PeripheralVision:
    """
    The 'Lizard Brain' - fast CV-based change detection.
    Runs on every frame, costs $0, ~5ms.
    """
    
    def __init__(self):
        self.prev_frame = None
        self.prev_scoreboard = None
        self.motion_history = deque(maxlen=30)  # Last 1 second @ 30fps
        self.state_history = deque(maxlen=10)
        
        # ROI definitions for LPL broadcast layout
        self.ROI = {
            'scoreboard': (0, 0, 1920, 100),       # Top bar
            'minimap': (1550, 780, 370, 300),      # Bottom-right
            'left_health': (200, 700, 300, 50),    # Left team health bars
            'right_health': (1420, 700, 300, 50),  # Right team health bars
            'dragon_pit': (750, 600, 200, 150),    # Center-bottom area
        }
        
        # Motion detection params
        self.motion_threshold = 0.02  # 2% pixel change
        self.scoreboard_change_threshold = 0.05  # 5% change triggers update (was 1%)
        
    def process(self, frame: np.ndarray) -> List[PerceptionAlert]:
        """
        Analyze frame for changes. Returns alerts if VLM should be triggered.
        """
        alerts = []
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640, 360))
        
        if self.prev_frame is None:
            self.prev_frame = gray
            # First frame always triggers analysis
            alerts.append(PerceptionAlert(
                alert_type=AlertType.STATE_TRANSITION,
                urgency=1.0,
                context="Initial frame"
            ))
            return alerts
        
        # 1. Global motion detection
        motion = self._detect_motion(gray)
        self.motion_history.append(motion)
        
        # Spike detection: current motion >> recent average
        avg_motion = np.mean(list(self.motion_history)) if len(self.motion_history) > 5 else motion
        if motion > avg_motion * 2.0 and motion > 0.03:
            alerts.append(PerceptionAlert(
                alert_type=AlertType.MOTION_SPIKE,
                urgency=min(motion * 10, 1.0),
                context=f"Motion spike: {motion:.1%} vs avg {avg_motion:.1%}"
            ))
        
        # 2. Scoreboard change detection
        scoreboard_roi = self._extract_roi(frame, 'scoreboard')
        if self.prev_scoreboard is not None:
            sb_change = self._measure_change(scoreboard_roi, self.prev_scoreboard)
            if sb_change > self.scoreboard_change_threshold:
                alerts.append(PerceptionAlert(
                    alert_type=AlertType.SCOREBOARD_CHANGE,
                    urgency=0.8,
                    region=self.ROI['scoreboard'],
                    context=f"Scoreboard changed: {sb_change:.1%}"
                ))
        self.prev_scoreboard = scoreboard_roi
        
        # 3. Health bar rapid change (fight detection)
        # TODO: Implement health bar specific detection
        
        self.prev_frame = gray
        return alerts
    
    def _detect_motion(self, gray: np.ndarray) -> float:
        """Calculate frame-to-frame motion as percentage of changed pixels."""
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion = np.sum(thresh > 0) / thresh.size
        return motion
    
    def _extract_roi(self, frame: np.ndarray, roi_name: str) -> np.ndarray:
        """Extract a region of interest from frame."""
        x, y, w, h = self.ROI[roi_name]
        # Scale ROI to actual frame size
        fh, fw = frame.shape[:2]
        scale_x, scale_y = fw / 1920, fh / 1080
        x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
        return frame[y:y+h, x:x+w]
    
    def _measure_change(self, current: np.ndarray, previous: np.ndarray) -> float:
        """Measure change between two ROI crops."""
        if current.shape != previous.shape:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]))
        
        gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if len(current.shape) == 3 else current
        gray_prev = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if len(previous.shape) == 3 else previous
        
        diff = cv2.absdiff(gray_curr, gray_prev)
        return np.mean(diff) / 255.0


class ActivePerceptionEngine:
    """
    Brain-inspired perception engine.
    
    - Peripheral layer runs on every frame (free)
    - Foveal layer (VLM) only triggered when peripheral detects change
    """
    
    def __init__(self, vlm_engine=None):
        self.peripheral = PeripheralVision()
        self.vlm = vlm_engine  # Any VLM engine (single or multi-model)
        
        self.last_vlm_call = 0
        self.min_vlm_interval = 5  # At least 5s between VLM calls
        self.last_game_state = None
        self.alerts_since_last_vlm = []
        
        logger.info("Active Perception Engine initialized")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a frame using peripheral + foveal pipeline.
        
        Returns:
            - If no VLM call: {"vlm_called": False, "alerts": [...]}
            - If VLM called: Full game state + {"vlm_called": True}
        """
        # 1. Peripheral processing (always runs, ~5ms)
        alerts = self.peripheral.process(frame)
        self.alerts_since_last_vlm.extend(alerts)
        
        # 2. Decision: Should we call VLM?
        should_call_vlm = self._should_trigger_vlm(alerts)
        
        if not should_call_vlm:
            return {
                "vlm_called": False,
                "alerts": [a.context for a in alerts],
                "cached_state": self.last_game_state
            }
        
        # 3. Foveal processing (VLM call)
        if self.vlm is None:
            logger.warning("No VLM engine configured")
            return {"vlm_called": False, "error": "No VLM"}
        
        # Build context from accumulated alerts
        context = self._build_alert_context()
        
        result = self.vlm.analyze_frame(frame)
        self.last_vlm_call = time.time()
        self.alerts_since_last_vlm = []
        
        if result.get("is_live_game"):
            self.last_game_state = result.get("game_state")
        
        result["vlm_called"] = True
        result["trigger_context"] = context
        return result
    
    def _should_trigger_vlm(self, current_alerts: List[PerceptionAlert]) -> bool:
        """Decide if VLM should be called based on alerts."""
        now = time.time()
        elapsed = now - self.last_vlm_call
        
        # Respect minimum interval
        if elapsed < self.min_vlm_interval:
            return False
        
        # High urgency alert = immediate trigger
        for alert in current_alerts:
            if alert.urgency >= 0.8:
                logger.info(f"VLM triggered by high-urgency: {alert.context}")
                return True
        
        # Accumulated alerts trigger
        if len(self.alerts_since_last_vlm) >= 3:
            logger.info(f"VLM triggered by accumulated alerts: {len(self.alerts_since_last_vlm)}")
            return True
        
        # Periodic refresh if no triggers for a while
        if elapsed > 30 and len(self.alerts_since_last_vlm) > 0:
            logger.info("VLM triggered by periodic refresh")
            return True
        
        return False
    
    def _build_alert_context(self) -> str:
        """Build context string from accumulated alerts for VLM prompt."""
        if not self.alerts_since_last_vlm:
            return "Routine check"
        
        contexts = [a.context for a in self.alerts_since_last_vlm if a.context]
        return "; ".join(contexts[-5:])  # Last 5 alerts


# Quick test
if __name__ == "__main__":
    import sys
    
    # Test peripheral vision on a video
    cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else 0)
    peripheral = PeripheralVision()
    
    print("Testing Peripheral Vision (press Q to quit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        alerts = peripheral.process(frame)
        if alerts:
            for a in alerts:
                print(f"[{a.alert_type.value}] {a.context}")
        
        cv2.imshow("Peripheral Test", frame)
        if cv2.waitKey(30) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
