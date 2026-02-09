"""
Tiered Intelligence Engine - Model Selection by Task Complexity

Fast Models (2.5-flash-lite, 2.0-flash):
  - State extraction (gold, kills, time)
  - Broadcast state detection
  - Simple OCR tasks
  
Pro Models (2.5-pro, 3-pro):  
  - Strategic fight predictions ("Who wins next fight?")
  - Item power spike detection ("ADC just completed Infinity Edge")
  - Objective timing analysis ("Blue is positioned for dragon")
  - Long-term game trajectory predictions
  
This conserves Pro model quota (20 RPD) for high-value cognitive tasks.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TaskComplexity(Enum):
    EXTRACTION = "extraction"     # Fast models - simple state reading
    ANALYSIS = "analysis"         # Pro models - strategic reasoning


@dataclass
class IntelligenceRequest:
    task: TaskComplexity
    prompt: str
    image: Optional[Any] = None
    history_context: Optional[str] = None


class TieredIntelligenceEngine:
    """
    Routes requests to appropriate model tier based on task complexity.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Model tiers
        self.fast_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite", 
            "gemini-2.0-flash",
        ]
        self.pro_models = [
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
        ]
        
        # Track daily usage
        self.usage = {model: 0 for model in self.fast_models + self.pro_models}
        self.daily_limit = 20  # Free tier limit per model
        
        # Initialize models
        self.models = {}
        if GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            for name in self.fast_models + self.pro_models:
                try:
                    self.models[name] = genai.GenerativeModel(name)
                    logger.info(f"Loaded: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
    
    def _get_available_model(self, tier: List[str]) -> Optional[str]:
        """Get first available model from tier that hasn't hit quota."""
        for model_name in tier:
            if model_name in self.models and self.usage[model_name] < self.daily_limit:
                return model_name
        return None
    
    def extract_state(self, image) -> Dict[str, Any]:
        """
        FAST PATH: Simple state extraction.
        Uses flash-lite models.
        """
        model_name = self._get_available_model(self.fast_models)
        if not model_name:
            return {"error": "All fast models exhausted"}
        
        prompt = """Analyze this League of Legends frame.
        
Return JSON with:
{
  "broadcast_state": "LIVE_GAME" or "DRAFT_PHASE" or "OTHER",
  "game_state": {
    "game_time": "MM:SS",
    "left_team": {"kills": N, "gold_k": N.N, "towers": N, "dragons": N},
    "right_team": {"kills": N, "gold_k": N.N, "towers": N, "dragons": N}
  }
}"""

        return self._call_model(model_name, image, prompt)
    
    def analyze_strategy(self, image, game_state: Dict, history: str) -> Dict[str, Any]:
        """
        PRO PATH: Deep strategic analysis.
        Uses pro models for complex temporal reasoning.
        
        Answers questions like:
        - Who wins the next fight?
        - Who is best positioned for the next objective?
        - Are there any item power spikes?
        """
        model_name = self._get_available_model(self.pro_models)
        if not model_name:
            # Fallback to fast model with simpler prompt
            logger.warning("Pro models exhausted, falling back to fast")
            model_name = self._get_available_model(self.fast_models)
            if not model_name:
                return {"error": "All models exhausted"}
        
        # Build rich context for strategic analysis
        left = game_state.get('left_team', {})
        right = game_state.get('right_team', {})
        
        prompt = f"""You are an expert League of Legends analyst.

CURRENT GAME STATE:
- Time: {game_state.get('game_time', 'Unknown')}
- Left Team: {left.get('kills', 0)} kills, {left.get('gold_k', 0)}k gold, {left.get('towers', 0)} towers
- Right Team: {right.get('kills', 0)} kills, {right.get('gold_k', 0)}k gold, {right.get('towers', 0)} towers

RECENT GAME HISTORY:
{history}

ANALYZE THE IMAGE AND ANSWER:

1. **Fight Prediction**: If a 5v5 teamfight happens in the next 60 seconds, who wins?
   Consider: Gold lead, item power spikes, ability cooldowns visible, positioning

2. **Objective Control**: Who is better positioned for the next major objective (Dragon/Baron)?
   Consider: Minimap positions, number of players nearby, vision control

3. **Power Spikes**: Are there any significant item completions or level spikes visible?
   Look for: Mythic item completions, level 6/11/16 thresholds

4. **Game Trajectory**: Based on current state and momentum, who is more likely to win?

Return JSON:
{{
  "next_fight_winner": "LEFT" or "RIGHT" or "EVEN",
  "fight_confidence": 0.0-1.0,
  "fight_reasoning": "Why?",
  
  "objective_advantage": "LEFT" or "RIGHT" or "NEUTRAL",
  "objective_reasoning": "Why?",
  
  "power_spikes": ["List any visible item/level spikes"],
  
  "game_winner": "LEFT" or "RIGHT",
  "game_confidence": 0.0-1.0,
  "trajectory_reasoning": "Why? Be specific about momentum."
}}"""

        result = self._call_model(model_name, image, prompt)
        result["analysis_model"] = model_name  # Track which model did analysis
        return result
    
    def _call_model(self, model_name: str, image, prompt: str) -> Dict[str, Any]:
        """Make API call and track usage."""
        import json
        import re
        from PIL import Image
        import cv2
        
        model = self.models.get(model_name)
        if not model:
            return {"error": f"Model {model_name} not available"}
        
        try:
            # Prepare image
            if isinstance(image, type(None)):
                return {"error": "No image provided"}
            
            if hasattr(image, 'shape'):  # numpy array
                small = cv2.resize(image, (1280, 720))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
            else:
                img = image
            
            # Call API
            response = model.generate_content(
                [img, prompt],
                generation_config={'temperature': 0.2, 'max_output_tokens': 1000}
            )
            
            self.usage[model_name] += 1
            
            if not response or not response.text:
                return {"error": "Empty response"}
            
            content = response.text.strip()
            if "```" in content:
                content = re.sub(r"```json|```", "", content).strip()
            
            data = json.loads(content)
            data["model_used"] = model_name
            return data
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse: {e}", "raw": content[:200] if 'content' in dir() else ""}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                self.usage[model_name] = self.daily_limit  # Mark exhausted
                logger.warning(f"[{model_name}] Hit rate limit")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, int]:
        """Get usage status for all models."""
        return {
            "usage": dict(self.usage),
            "fast_remaining": sum(self.daily_limit - self.usage.get(m, 0) for m in self.fast_models),
            "pro_remaining": sum(self.daily_limit - self.usage.get(m, 0) for m in self.pro_models),
        }


# Test
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    engine = TieredIntelligenceEngine()
    print("Status:", engine.get_status())
