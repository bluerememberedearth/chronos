
import threading
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class BroadcastStateMonitor:
    def __init__(self, perception_engine=None):
        self.resolution = (1920, 1080)
        self.engine = perception_engine
        self.state = "UNKNOWN"
        self.frame_interval = 60 # Check every 60 frames (1 second of 60fps video)
        self.frame_counter = 0
        self.consecutive_live_counter = 0 # Stability check
        self.REQUIRED_STREAK = 3 # Require 3 consecutive LIVE_GAME detections (3 seconds)
        
        # Concurrency Control
        self._is_checking = False
        self._check_lock = threading.Lock()

    def start(self):
        pass # No thread loop needed, driven by main loop

    def update(self, frame: np.ndarray):
        """
        Called by main.py for every frame.
        """
        self.frame_counter += 1
        
        # Check Condition: Every 60 frames (1 video second)
        if self.frame_counter % self.frame_interval == 0:
            # Check if we are already running a check (Fast check before thread spawn)
            if self._is_checking:
                return # Skip to prevent backlog
            
            # We run this in a separate thread to avoid blocking the main ingest loop
            threading.Thread(target=self._run_cognitive_check_wrapper, args=(frame.copy(),), daemon=True).start()

    def _run_cognitive_check_wrapper(self, frame):
        """Wrapper to handle locking safely"""
        # Double check with lock
        with self._check_lock:
            if self._is_checking:
                return
            self._is_checking = True
            
        try:
            self._run_cognitive_check(frame)
        finally:
            with self._check_lock:
                self._is_checking = False

    def _run_cognitive_check(self, frame):
        """
        Asks the VLM: 'What is the state of this LoL broadcast?'
        """
        if not self.engine or not self.engine.client:
            return

        try:
            # Resizing to lower res to save tokens/bandwidth
            small_frame = cv2.resize(frame, (640, 360))
            
            import base64
            from io import BytesIO
            from PIL import Image
            
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{img_b64}"
            
            # Refined Prompt for Start Detection - JSON MODE
            prompt = (
                "Analyze this League of Legends broadcast frame. "
                "Classify into exactly one of these categories: "
                "LIVE_GAME (Active gameplay, HUD visible, minimap, health bars), "
                "DRAFT_PHASE (Champion select grid, bans, picks, player names), "
                "LOADING_SCREEN (Splash arts of champions, loading bar), "
                "CASTER_DESK (Analysts/Commentators at a desk, studio background), "
                "OTHER (Anything else not fitting the above categories). "
                "Return valid JSON only: {\"state\": \"CATEGORY_NAME\"}."
            )
            
            response = self.engine.client.chat.completions.create(
                model=self.engine.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=30
                # response_format={"type": "json_object"} # Removed: Not supported by current SiliconFlow model
            )
            
            import json
            import re
            content = response.choices[0].message.content.strip()
            
            # Strip markdown code blocks if present
            if "```" in content:
                content = re.sub(r"```json|```", "", content).strip()
            
            try:
                data = json.loads(content)
                raw_state = data.get("state", "OTHER").upper()
            except json.JSONDecodeError:
                # Fallback if model refuses JSON
                logger.warning(f"VLM returned non-JSON: {content}")
                raw_state = content.upper()
            
            # Simple cleaning
            final_state = "OTHER" # Default
            for valid in ["LIVE_GAME", "DRAFT_PHASE", "LOADING_SCREEN", "CASTER_DESK", "OTHER"]:
                 if valid in raw_state:
                     final_state = valid
                     break
            
            # STABILITY LOGIC
            if final_state == "LIVE_GAME":
                self.consecutive_live_counter += 1
                if self.consecutive_live_counter < self.REQUIRED_STREAK:
                     logger.info(f"VLM Potential Match: LIVE_GAME ({self.consecutive_live_counter}/{self.REQUIRED_STREAK})")
            else:
                self.consecutive_live_counter = 0
            
            # Only trigger state change if we met the streak requirement (for LIVE_GAME)
            # OR if we are transitioning OUT of LIVE_GAME immediately
            if self.consecutive_live_counter >= self.REQUIRED_STREAK:
                if self.state != "LIVE_GAME":
                    logger.info(f"Broadcast State (VLM): {self.state} -> LIVE_GAME")
                    self.state = "LIVE_GAME"
            elif final_state != "LIVE_GAME" and self.state == "LIVE_GAME":
                # Immediate drop out if we lose live game confidence
                logger.info(f"Broadcast State (VLM): {self.state} -> {final_state}")
                self.state = final_state
            elif final_state != "LIVE_GAME" and self.state != "LIVE_GAME":
                 # Update non-critical states freely
                 if final_state != self.state:
                      logger.info(f"Broadcast State (VLM): {self.state} -> {final_state}")
                      self.state = final_state

        except Exception as e:
            logger.error(f"Cognitive Check Failed: {e}")

    def get_state(self):
        return self.state

    def stop(self):
        pass
