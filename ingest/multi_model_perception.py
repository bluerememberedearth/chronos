"""
Multi-Model Gemini Perception Engine - Sensor Fusion

Rotates between multiple Gemini models to maximize effective throughput.
Each model has its own rate limit, so rotating = multiplied capacity.

Available Models (from dashboard):
- gemini-2.5-flash-lite: 4K RPM (primary)
- gemini-2-flash: 2K RPM
- gemini-2-flash-lite: 4K RPM  
- gemini-2.5-flash: 1K RPM
- gemini-3-flash: 1K RPM
- gemini-2.5-pro: 150 RPM (use sparingly for complex analysis)
"""

import os
import time
import json
import re
import logging
from typing import Optional, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed")


class ModelPool:
    """Manages a pool of Gemini models with rate limit awareness."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Model definitions: (name, rpd_limit, priority)
        # CORRECT names from genai.list_models()
        # Each model has ~20 RPD limit on free tier
        self.model_configs = [
            ("gemini-2.5-flash-lite", 20, 1),      # Primary - fast & cheap
            ("gemini-2.0-flash-lite", 20, 2),      # Backup 1
            ("gemini-2.0-flash", 20, 3),           # Backup 2  
            ("gemini-2.5-flash", 20, 4),           # Backup 3
            ("gemini-3-flash-preview", 20, 5),    # Backup 4
            ("gemini-2.5-pro", 20, 10),            # Premium - save for complex
        ]
        
        # Track usage per model
        self.usage = {name: deque(maxlen=100) for name, _, _ in self.model_configs}
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize all model instances."""
        for name, rpm, _ in self.model_configs:
            try:
                self.models[name] = genai.GenerativeModel(name)
                logger.info(f"Initialized model: {name} (limit: {rpm} RPM)")
            except Exception as e:
                logger.warning(f"Failed to init {name}: {e}")
    
    def get_available_model(self) -> Optional[tuple]:
        """Get the best available model based on recent usage."""
        now = time.time()
        
        for name, rpm_limit, priority in sorted(self.model_configs, key=lambda x: x[2]):
            if name not in self.models:
                continue
            
            # Count requests in last 60 seconds
            recent = [t for t in self.usage[name] if now - t < 60]
            
            # Leave 20% headroom
            if len(recent) < rpm_limit * 0.8:
                return (self.models[name], name)
        
        return None
    
    def record_usage(self, model_name: str):
        """Record that a model was used."""
        self.usage[model_name].append(time.time())
    
    def get_status(self) -> Dict[str, int]:
        """Get current usage status for all models."""
        now = time.time()
        return {
            name: len([t for t in self.usage[name] if now - t < 60])
            for name, _, _ in self.model_configs
        }


class MultiModelPerceptionEngine:
    """
    Perception engine that rotates between multiple Gemini models.
    Implements "sensor fusion" by leveraging multiple rate-limited APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.pool = None
        self.last_call_time = 0
        self.min_interval = 2  # Minimum 2s between calls (safety)
        self.call_count = 0
        
        if GEMINI_AVAILABLE and self.api_key:
            self.pool = ModelPool(self.api_key)
            logger.info("Multi-Model Perception Engine initialized")
        else:
            logger.error("Gemini not available")
    
    def analyze_frame(self, frame) -> Dict[str, Any]:
        """
        Analyze a frame using the best available model.
        Automatically rotates to avoid rate limits.
        """
        if not self.pool:
            return {"error": "Not initialized", "is_live_game": False}
        
        # Rate limiting
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        # Get available model
        model_info = self.pool.get_available_model()
        if not model_info:
            logger.warning("All models rate limited!")
            return {"error": "All models rate limited", "is_live_game": False}
        
        model, model_name = model_info
        
        try:
            import cv2
            from PIL import Image
            
            # Resize for efficiency
            small = cv2.resize(frame, (1280, 720))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            prompt = """Analyze this League of Legends broadcast frame.

TASK 1: Classify broadcast state
- LIVE_GAME: Active gameplay (minimap, HUD, health bars visible)
- DRAFT_PHASE: Champion select screen
- LOADING_SCREEN: Champion splash arts
- CASTER_DESK: Analysts at desk
- OTHER: Anything else

TASK 2: If LIVE_GAME, extract:
- game_time: "MM:SS"
- left_team: {kills, gold_k, towers, dragons}
- right_team: {kills, gold_k, towers, dragons}
- minimap_summary: Brief position/objective description

Return ONLY valid JSON:
{
  "broadcast_state": "STATE",
  "game_state": null OR {game_time, left_team, right_team, minimap_summary}
}"""

            self.last_call_time = time.time()
            self.call_count += 1
            
            response = model.generate_content(
                [img, prompt],
                generation_config={'temperature': 0.1, 'max_output_tokens': 500}
            )
            
            self.pool.record_usage(model_name)
            
            # Parse response
            if not response or not response.text:
                return {"error": "Empty response", "is_live_game": False}
            
            content = response.text.strip()
            if "```" in content:
                content = re.sub(r"```json|```", "", content).strip()
            
            data = json.loads(content)
            broadcast_state = data.get("broadcast_state", "OTHER")
            game_state = data.get("game_state")
            
            result = {
                "broadcast_state": broadcast_state,
                "is_live_game": broadcast_state == "LIVE_GAME",
                "game_state": game_state,
                "model_used": model_name,
                "call_number": self.call_count
            }
            
            gt = game_state.get('game_time', 'N/A') if game_state else 'N/A'
            logger.info(f"[{model_name}] {broadcast_state} @ {gt} (call #{self.call_count})")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON error: {e}")
            return {"error": str(e), "is_live_game": False}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                logger.warning(f"[{model_name}] Rate limited, will try another model")
                # Mark this model as heavily used
                for _ in range(100):
                    self.pool.record_usage(model_name)
            logger.error(f"API error: {e}")
            return {"error": str(e), "is_live_game": False}
    
    def get_status(self) -> Dict:
        """Get current status of all models."""
        if self.pool:
            return {
                "total_calls": self.call_count,
                "model_usage": self.pool.get_status()
            }
        return {"error": "Not initialized"}


# Quick test
if __name__ == "__main__":
    import cv2
    from dotenv import load_dotenv
    load_dotenv()
    
    engine = MultiModelPerceptionEngine()
    print("Status:", engine.get_status())
    
    # Test with a frame if available
    print("\nTo test, run:")
    print("  from ingest.multi_model_perception import MultiModelPerceptionEngine")
    print("  engine = MultiModelPerceptionEngine()")
    print("  result = engine.analyze_frame(frame)")
