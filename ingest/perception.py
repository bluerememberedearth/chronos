import logging
import time
import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        pass
    np = MockNumpy()
    np.ndarray = Any # Treat as Any type for hints

logger = logging.getLogger(__name__)

# --- Safe Imports ---
load_dotenv() # Load environment variables

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    # logger.warning("Ultralytics not found. Fast Path (YOLO) will run in MOCK mode.")
    YOLO_AVAILABLE = False

try:
    from openai import OpenAI
    SILICONFLOW_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI SDK not found. Smart Path (SiliconFlow) will run in MOCK mode.")
    SILICONFLOW_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import threading

class HybridPerceptionEngine:
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        """
        Args:
            yolo_model_path: Path to YOLO weights for fast detection.
        """
        self.fast_mode = True 
        self.smart_mode_interval = 30 # Run smart path every 30 frames
        self.frame_counter = 0
        
        # Load SiliconFlow Key from Env
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
             logger.warning("SILICONFLOW_API_KEY not found in .env")

        # --- Initialize YOLO ---
        if YOLO_AVAILABLE:
            try:
                # logger.info(f"Loading YOLO model: {yolo_model_path}")
                self.yolo = YOLO(yolo_model_path) 
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                self.yolo = None
        else:
            self.yolo = None

        # --- Initialize SiliconFlow (OpenAI Client) ---
        if SILICONFLOW_AVAILABLE and self.api_key:
            logger.info("Initializing SiliconFlow Client")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.siliconflow.com/v1"
            )
            # self.model = "Qwen/Qwen2.5-VL-72B-Instruct" 
            # Using slightly smaller model for speed/cost if preferred, or the big one
            self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        else:
            self.client = None
            if SILICONFLOW_AVAILABLE and not self.api_key:
                logger.warning("OpenAI SDK available but SILICONFLOW_API_KEY not found.")
        
        # --- Async VLM Control ---
        self._latest_smart_result = None
        self._is_smart_running = False
        self._smart_lock = threading.Lock()

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Main entry point for frame processing.
        Returns a dict containing 'fast' and optional 'smart' data.
        """
        self.frame_counter += 1
        
        result = {
            "timestamp": timestamp,
            "fast": {},
            "smart": self._latest_smart_result # Always return latest cached result
        }
        
        # 1. Fast Path (Every Frame)
        result["fast"] = self._run_fast_path(frame)
        
        # 2. Smart Path (Throttled & Async)
        # Only trigger if interval met AND not currently running
        if self.frame_counter % self.smart_mode_interval == 0:
            if not self._is_smart_running:
                # Spawn thread
                threading.Thread(target=self._run_smart_path_async, args=(frame.copy(),), daemon=True).start()
            
        return result

    def _run_fast_path(self, frame: np.ndarray) -> Dict:
        """
        Executes YOLO / Lightweight CV.
        """
        if self.yolo:
            # Tracking with persistence
            results = self.yolo.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes
            return {
                "status": "active",
                "object_count": len(boxes),
            }
        
        return {
            "status": "mock_tracking", 
            "object_count": 5,
            "note": "YOLO not loaded"
        }

    def _run_smart_path_async(self, frame: np.ndarray):
        """Wrapper to manage state lock"""
        with self._smart_lock:
            if self._is_smart_running: return
            self._is_smart_running = True
            
        try:
             result = self._run_smart_path(frame)
             self._latest_smart_result = result
        except Exception as e:
             logger.error(f"Async Smart Path Error: {e}")
        finally:
             with self._smart_lock:
                 self._is_smart_running = False

    def _run_smart_path(self, frame: np.ndarray) -> Dict:
        """
        Executes VLM using SiliconFlow API.
        Now called from background thread.
        """
        if self.client and PIL_AVAILABLE:
            try:
                # 0. Resize for Performance (960x540 qHD)
                # Reduces token count and upload latency
                if NUMPY_AVAILABLE:
                     import cv2 # Ensure cv2 is available or use PIL resize
                     # Assuming cv2 is imported at top or available in env
                     try:
                        import cv2
                        frame = cv2.resize(frame, (960, 540))
                     except ImportError:
                        pass # Fallback to PIL resize if cv2 fails here (unlikely in this env)

                # 1. Encode Image to Base64
                if frame.shape[2] == 3:
                     # BGR to RGB
                     rgb_frame = frame[..., ::-1]
                     image = Image.fromarray(rgb_frame)
                else:
                     image = Image.fromarray(frame)
                
                # Resize if cv2 missed
                if image.size[0] > 960:
                    image = image.resize((960, 540))

                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=85) # Slight compression
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_data_url = f"data:image/jpeg;base64,{img_str}"

                # 2. API Call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_data_url}},
                                {"type": "text", "text": "Analyze this LoL frame. Return JSON: {gold_diff: int, objectives: list}."}
                            ]
                        }
                    ],
                    # response_format={"type": "json_object"}, # Valid for some models, Qwen supports it usually
                    max_tokens=512
                )
                
                content = response.choices[0].message.content
                
                # Cleanup markdown if present
                if "```" in content:
                    import re
                    content = re.sub(r"```json|```", "", content).strip()
                    
                return {
                    "status": "success",
                    "raw_response": content,
                    "timestamp": time.time()
                }

            except Exception as e:
                logger.error(f"SiliconFlow Inference Failed: {e}")
                return {"status": "error", "error": str(e)}
        
        return {
            "status": "mock_analysis", 
            "gold_lead": 1500,
            "note": "Client not initialized"
        }

    def analyze_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts detailed game state (Gold, Kills, Objectives) using VLM.
        Targeted for LPL/Professional broadcast overlays.
        """
        if not self.client or not PIL_AVAILABLE:
             return {"error": "VLM not initialized"}

        try:
             # Resize to qHD for speed/cost balance
             # if NUMPY_AVAILABLE:
             #      import cv2
             #      frame = cv2.resize(frame, (960, 540))
             
             # Encode
             image = Image.fromarray(frame[..., ::-1]) # BGR -> RGB
             buffered = BytesIO()
             image.save(buffered, format="JPEG", quality=85)
             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
             img_url = f"data:image/jpeg;base64,{img_str}"

             prompt = """Analyze this League of Legends LPL broadcast frame.
Extract the following Game State data into JSON:
- game_time (string, e.g. '15:30')
- left_team (Scoreboard Left Side): 
    - kills (int)
    - gold_k (float)
    - towers (int)
    - dragons (int)
    - grubs (int)
    - ult_count (int): How many green "Ultimate Ready" dots are visible next to player portraits?
- right_team (Scoreboard Right Side): 
    - kills (int)
    - gold_k (float)
    - towers (int)
    - dragons (int)
    - grubs (int)
    - ult_count (int): How many green "Ultimate Ready" dots are visible?

Strategic Analysis (Look at Minimap & Lane States):
- next_objective: {
    "timer": "Time until next Dragon or Baron (e.g. '1:30' or 'Spawning Soon' or 'None visible')",
    "type": "Dragon or Baron"
}
- minimap: {
    "left_team_positions": [list of visible roles or champs],
    "right_team_positions": [list of visible roles or champs],
    "objective_control": "Who controls river/dragon area?",
    "vision_notes": "Key wards or vision gaps"
}
Note: Gold is usually shown in 'k' (e.g. 30.1k). Grubs/Heralds are the purple icon count.
Return valid JSON only."""

             response = self.client.chat.completions.create(
                 model=self.model,
                 messages=[
                     {
                         "role": "user",
                         "content": [
                             {"type": "image_url", "image_url": {"url": img_url}},
                             {"type": "text", "text": prompt}
                         ]
                     }
                 ],
                 max_tokens=300
             )

             content = response.choices[0].message.content
             
             # Cleanup markdown
             if "```" in content:
                 import re
                 content = re.sub(r"```json|```", "", content).strip()
                 
             import json
             return json.loads(content)

        except Exception as e:
             logger.error(f"Game State Extraction Failed: {e}")
             return {"error": str(e)}
