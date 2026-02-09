"""
Gemini-based Perception Engine - Rate Limit Optimized

This module uses Google's Gemini API for vision tasks, designed
for severe rate limiting (1-2 calls per minute max).

Key optimization: ONE call returns BOTH:
1. Broadcast state (LIVE_GAME, DRAFT, etc.)
2. Full game state extraction (gold, kills, etc.)
"""

import os
import time
import json
import re
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import Gemini SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")


class GeminiPerceptionEngine:
    """
    Rate-limit optimized perception engine using Gemini with multi-model rotation.
    
    Combines broadcast state detection AND game extraction into
    a single VLM call to minimize API usage. Rotates through available
    models when quota is reached.
    """
    
    AVAILABLE_MODELS = [
        'gemini-2.5-flash',
        'gemini-3-flash-preview',
        'gemini-2.0-flash',
        'gemini-2.5-flash-lite',
        'gemini-1.5-flash',
        'gemini-1.5-pro' # Backup for deep analysis
    ]
    
    MAX_RPD = 20  # User reported strict 21/day limit. Safety margin.
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.current_model_idx = 0
        self.models = {}  # Cache GenerativeModel instances
        self.usage_stats = {m: 0 for m in self.AVAILABLE_MODELS}  # Track calls per model
        self.last_call_time = 0
        self.min_call_interval = 10  # 10s is plenty conservative for 4K RPM
        
        if GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info(f"Gemini Perception Engine initialized with {len(self.AVAILABLE_MODELS)} models. Max RPD: {self.MAX_RPD}")
        else:
            logger.error("Gemini not available - check API key and installation")

    def _get_model(self, rotate=False):
        """Get current model, rotating if quota exceeded or explicitly requested."""
        if not GEMINI_AVAILABLE or not self.api_key:
            return None
            
        if rotate:
            self.current_model_idx = (self.current_model_idx + 1) % len(self.AVAILABLE_MODELS)
            logger.info(f"Rotating to model: {self.AVAILABLE_MODELS[self.current_model_idx]}")
            
        # Proactive Rotation: Check if current model is exhausted
        start_idx = self.current_model_idx
        current_name = self.AVAILABLE_MODELS[self.current_model_idx]
        
        while self.usage_stats[current_name] >= self.MAX_RPD:
            logger.warning(f"Model {current_name} hit daily limit ({self.MAX_RPD}). Auto-rotating...")
            self.current_model_idx = (self.current_model_idx + 1) % len(self.AVAILABLE_MODELS)
            current_name = self.AVAILABLE_MODELS[self.current_model_idx]
            
            if self.current_model_idx == start_idx:
                logger.error("CRITICAL: All Gemini models have exhausted their daily quota (20 RPD).")
                return None
            
        model_name = self.AVAILABLE_MODELS[self.current_model_idx]
        if model_name not in self.models:
            self.models[model_name] = genai.GenerativeModel(model_name)
            
        return self.models[model_name]

    def get_usage_report(self) -> Dict[str, int]:
        """Return usage counts per model."""
        return self.usage_stats.copy()
    
    def _clean_json_string(self, content: str) -> str:
        """Robust JSON cleanup."""
        # 1. Strip markdown
        if "```" in content:
            content = re.sub(r"```json|```", "", content).strip()
        
        # 2. Find outer braces
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end+1]
            
        # 3. Remove trailing commas (simple case: , before })
        # This regex matches a comma followed by whitespace and a closing brace/bracket
        content = re.sub(r",\s*([\]}])", r"\1", content)
        
        return content

    def analyze_frame(self, frame) -> Dict[str, Any]:
        """
        Analyze a single frame and return BOTH broadcast state AND game state.
        
        Rotates models on quota exceeded (429).
        """
        for _ in range(len(self.AVAILABLE_MODELS)):
            model = self._get_model()
            if not model:
                return {"error": "Gemini not configured", "is_live_game": False}
            
            # Rate limiting check
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_call_interval:
                wait_time = self.min_call_interval - elapsed
                logger.info(f"Rate limit: waiting {wait_time:.1f}s before next call")
                time.sleep(wait_time)
            
            try:
                import cv2
                from PIL import Image
                
                # Resize for efficiency (720p is plenty for UI reading)
                small = cv2.resize(frame, (1280, 720))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                
                # Combined prompt - ONE call does everything
                prompt = """Analyze this League of Legends broadcast frame.
    
    TASK 1: Classify the broadcast state
    - LIVE_GAME: Active gameplay visible (minimap, health bars, HUD, scoreboard)
    - DRAFT_PHASE: Champion select screen (picks/bans grid, player names)
    - LOADING_SCREEN: Champion splash arts, loading indicators
    - CASTER_DESK: Analysts at desk, studio background
    - OTHER: Advertisements, replays, or unclear content
    
    TASK 2: If LIVE_GAME, extract the game state data:
    - game_time: Current game clock (e.g. "15:30")
    - left_team (scoreboard LEFT side):
      - kills, gold_k (as float like 25.3), towers, dragons
    - right_team (scoreboard RIGHT side):
      - kills, gold_k (as float like 25.3), towers, dragons  
    - minimap_summary: Brief description of team positions and objective control
    
    Return ONLY valid JSON in this exact format:
    {
      "broadcast_state": "LIVE_GAME" or "DRAFT_PHASE" or "LOADING_SCREEN" or "CASTER_DESK" or "OTHER",
      "game_state": null OR {
        "game_time": "MM:SS",
        "left_team": {"kills": 0, "gold_k": 0.0, "towers": 0, "dragons": 0},
        "right_team": {"kills": 0, "gold_k": 0.0, "towers": 0, "dragons": 0},
        "minimap_summary": "description"
      }
    }"""
    
                self.last_call_time = time.time()
                
                response = model.generate_content(
                    [img, prompt],
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 500
                    }
                )
                
                if not response or not response.text:
                    logger.error(f"Empty response from Gemini. Response: {response}")
                    return {"error": "Empty response", "is_live_game": False}
                
                # Increment usage stats
                model_name = self.AVAILABLE_MODELS[self.current_model_idx]
                self.usage_stats[model_name] += 1
                
                # Parse response
                content = self._clean_json_string(response.text.strip())
                logger.debug(f"Raw response: {content[:200]}")
                
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}. Content: {content[:200]}")
                    return {"error": f"JSON parse: {str(e)}", "is_live_game": False, "raw": content[:200]}
                
                # Normalize response with safe defaults
                broadcast_state = data.get("broadcast_state", "OTHER") if data else "OTHER"
                game_state = data.get("game_state") if data else None
                
                result = {
                    "broadcast_state": broadcast_state,
                    "is_live_game": broadcast_state == "LIVE_GAME",
                    "game_state": game_state,
                    "model_used": model_name
                }
                
                logger.info(f"Gemini({model_name}): {result['broadcast_state']} @ {game_state.get('game_time', 'N/A') if game_state else 'N/A'}")
                return result
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    logger.warning(f"Quota exceeded for {self.AVAILABLE_MODELS[self.current_model_idx]}. Rotating...")
                    self._get_model(rotate=True)
                    continue
                logger.error(f"Gemini API error: {e}")
                return {"error": str(e), "is_live_game": False}

        return {"error": "All models exhausted", "is_live_game": False}

    def set_min_interval(self, seconds: int):
        """Adjust minimum interval between API calls."""
        self.min_call_interval = max(10, seconds)  # At least 10s
        logger.info(f"VLM call interval set to {self.min_call_interval}s")
    
    def analyze_with_prediction(self, frame, blue_comp: str, red_comp: str, 
                                  history: str = "") -> Dict[str, Any]:
        """
        Combined analysis: state extraction + winner prediction in ONE API call.
        
        Rotates models on 429.
        """
        for _ in range(len(self.AVAILABLE_MODELS)):
            model = self._get_model()
            if not model:
                return {"error": "Gemini not configured", "is_live_game": False}
            
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_call_interval:
                wait_time = self.min_call_interval - elapsed
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            try:
                import cv2
                from PIL import Image
                
                small = cv2.resize(frame, (1280, 720))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                
                prompt = f"""Analyze this League of Legends broadcast frame. Extract ALL visible information and provide strategic analysis.

TEAM COMPOSITIONS:
- BLUE (Left scoreboard): {blue_comp}
- RED (Right scoreboard): {red_comp}

GAME CONTEXT:
{history}

=== EXTRACTION TASKS ===

1. BROADCAST STATE: LIVE_GAME / DRAFT_PHASE / LOADING_SCREEN / CASTER_DESK / OTHER

2. SCOREBOARD (read exact values from HUD):
   - Game clock (top center)
   - Kills, Gold (in thousands), Towers destroyed, Dragons for each side

3. MINIMAP ANALYSIS (bottom-right corner):
   - Champion positions (grouped? split? invading?)
   - Vision control (wards visible?)
   - Objective setup (near dragon/baron?)

4. CHAMPION STATUS (if visible):
   - Health/mana states
   - Ultimate availability (green dots = ready)
   - Key cooldowns or summoner spells

=== STRATEGIC ANALYSIS ===

5. POWER SPIKE ASSESSMENT:
   - Which team has stronger item timings right now?
   - Any champion about to hit key level (6, 11, 16)?
   - Core item completions visible?

6. WIN CONDITION ANALYSIS:
   - Blue's path to victory (e.g., "Scale with Jinx, avoid fights until 3 items")
   - Red's path to victory (e.g., "Force early fights with dive comp")
   - Which team is executing their win condition better?

7. NEXT FIGHT PREDICTION:
   - If a fight happens in the next 2 minutes, who wins and why?
   - Key abilities/ultimates that would decide the fight?

8. GAME WINNER PREDICTION:
   - Based on current trajectory, gold leads, and team compositions

Return ONLY valid JSON:
{{
  "broadcast_state": "STATE",
  "game_state": null OR {{
    "game_time": "MM:SS",
    "left_team": {{"kills": 0, "gold_k": 0.0, "towers": 0, "dragons": 0, "baron": false}},
    "right_team": {{"kills": 0, "gold_k": 0.0, "towers": 0, "dragons": 0, "baron": false}},
    "minimap_summary": "positioning description",
    "visible_ultimates": {{"blue": 0, "red": 0}},
    "next_objective": "dragon/baron/tower"
  }},
  "analysis": {{
    "blue_win_condition": "how blue wins",
    "red_win_condition": "how red wins", 
    "power_spike_advantage": "BLUE" or "RED" or "EVEN",
    "power_spike_reason": "why",
    "next_fight_winner": "BLUE" or "RED",
    "fight_reasoning": "key abilities and positioning factors"
  }},
  "prediction": {{
    "winner": "BLUE" or "RED",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation citing specific advantages"
  }}
}}"""

                self.last_call_time = time.time()
                
                response = model.generate_content(
                    [img, prompt],
                    generation_config={'temperature': 0.2, 'max_output_tokens': 65536}
                )
                
                if not response or not response.text:
                    return {"error": "Empty response", "is_live_game": False}
                
                # Increment usage stats
                model_name = self.AVAILABLE_MODELS[self.current_model_idx]
                self.usage_stats[model_name] += 1
                
                content = response.text.strip()
                if "```" in content:
                    content = re.sub(r"```json|```", "", content).strip()
                
                data = json.loads(content)
                broadcast_state = data.get("broadcast_state", "OTHER")
                
                result = {
                    "broadcast_state": broadcast_state,
                    "is_live_game": broadcast_state == "LIVE_GAME",
                    "game_state": data.get("game_state"),
                    "prediction": data.get("prediction", {}),
                    "model_used": model_name
                }
                
                gs = result.get("game_state", {})
                pred = result.get("prediction", {})
                if gs:
                    logger.info(f"Gemini({model_name}): {broadcast_state} @ {gs.get('game_time', 'N/A')} | "
                               f"Pred: {pred.get('winner', '?')} ({pred.get('confidence', 0):.0%})")
                
                return result
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    logger.warning(f"Quota exceeded for {self.AVAILABLE_MODELS[self.current_model_idx]}. Rotating...")
                    self._get_model(rotate=True)
                    continue
                logger.error(f"Gemini error: {e}")
                return {"error": str(e), "is_live_game": False}

        return {"error": "All models exhausted", "is_live_game": False}

    def analyze_sequence_with_prediction(self, frames: list, blue_comp: str, red_comp: str, 
                                       history: str = "") -> Dict[str, Any]:
        """
        Deep Macro Analysis: Analyze a sequence of frames (video-like) for strategic insight.
        Uses the 1M+ token context window to process multiple historical frames + text context.
        """
        if not frames:
            return {"error": "No frames provided"}
            
        for _ in range(len(self.AVAILABLE_MODELS)):
            model = self._get_model()
            if not model:
                return {"error": "Gemini not configured"}
            
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_call_interval:
                wait_time = self.min_call_interval - elapsed
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            try:
                import cv2
                from PIL import Image
                
                # Convert all frames to PIL Images (downscaled)
                pil_images = []
                for f in frames:
                    small = cv2.resize(f, (1280, 720))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(rgb))
                
                # Metadata for prompt
                duration = len(frames) * 30 / 60
                
                prompt = f"""You are the Head Analyst for a professional League of Legends team (LPL/LCK level).
The user is providing a CHRONOLOGICAL SEQUENCE of broadcast frames, captured approximately every 30 seconds.
Your task is to perform a DEEP MACRO ANALYSIS of the game's evolution over these {duration:.1f} minutes.

=== CONTEXT ===
TEAM COMPOSITIONS:
- BLUE: {blue_comp}
- RED: {red_comp}

GAME HISTORY & STATIC DATA:
{history}

=== ANALYSIS OBJECTIVES ===

1. 🌊 FLOW & MOMENTUM:
   - Do not just describe static states. Describe the *change* and *flow* across the frames.
   - Who is dictating the pace? Is the game stalling or accelerating?

2. 🗺️ MACRO & VISION:
   - Analyze the shape of vision lines (wards on minimap).
   - Look at wave states (pushing/freezing). Who has "priority" to move first?
   - Identify any "cross-map plays" (e.g., trading Top Tower for Dragon).

3. ⚡ POWER SPIKES & SCALING (Critical):
   - Reference the Champion Abilities/Context provided above.
   - Which team is stronger *right now* vs *in 10 minutes*?
   - Are key ultimates (e.g., Global Executions) online and threatening?

4. 🏆 WIN CONDITION CHECKLIST:
   - Evaluated based on the visual evidence + gold trends.
   - BLUE WIN CON: Is it met? (e.g., "Scale to late", "Snowball early")
   - RED WIN CON: Is it met?
   - VERDICT: Which team is executing their condition better?

5. 🔮 PREDICTION:
   - Based *strictly* on the trajectory shown in these frames.
   - Confidence must reflect the volatility of the game state.

Return ONLY valid JSON:
{{
  "broadcast_state": "LIVE_GAME",
  "is_live_game": true,
  "game_state": {{ "game_time": "current", "note": "extracted from last frame" }},
  "deep_analysis": {{
    "trend_summary": "Concise narrative of the sequence (2 sentences)",
    "momentum": "BLUE" or "RED" or "EVEN",
    "key_factor": "The single most important strategic element (e.g. 'Baron Vision', 'Jinx Items')",
    "lane_pressure": {{"top": "Blue/Red", "mid": "Blue/Red", "bot": "Blue/Red"}},
    "win_conditions": [ "Blue: [Status]", "Red: [Status]" ]
  }},
  "prediction": {{
    "winner": "BLUE" or "RED",
    "confidence": 0.0-1.0,
    "reasoning": "Professional analyst explanation citing specific macro details from the sequence."
  }}
}}"""
                
                self.last_call_time = time.time()
                
                # Pass list of images + prompt
                content = pil_images + [prompt]
                
                response = model.generate_content(
                    content,
                    generation_config={'temperature': 0.2, 'max_output_tokens': 4000} # Increased for detailed analysis
                )
                
                if not response or not response.text:
                    return {"error": "Empty response"}
                
                self.usage_stats[self.AVAILABLE_MODELS[self.current_model_idx]] += 1
                
                content_text = self._clean_json_string(response.text.strip())
                
                data = json.loads(content_text)
                return data
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    logger.warning(f"Quota exceeded (Sequence) for {self.AVAILABLE_MODELS[self.current_model_idx]}. Rotating...")
                    self._get_model(rotate=True)
                    continue
                logger.error(f"Sequence analysis error: {e}")
                return {"error": str(e)}

        return {"error": "All models exhausted"}

    def summarize_narrative(self, events_history: str) -> str:
        """
        Generates a concise narrative summary of the game so far based on event logs.
        Uses text-only prompt for speed and cost efficiency.
        """
        for _ in range(len(self.AVAILABLE_MODELS)):
            model = self._get_model()
            if not model: return "Error: No model configured."
            
            # Rate limiting check (shared with vision calls)
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_call_interval:
                wait_time = self.min_call_interval - elapsed
                logger.info(f"Rate limit (Narrative): waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            try:
                prompt = f"""You are a professional League of Legends color commentator and analyst.
Summarize the following game events into a concise 2-3 sentence narrative that explains the "story" of the game so far.
Focus on momentum shifts, win conditions, and key decisive moments. Do not just list events.

Events History:
{events_history}

Narrative Summary:"""

                self.last_call_time = time.time()
                response = model.generate_content(
                    prompt,
                    generation_config={'temperature': 0.7, 'max_output_tokens': 200}
                )
                
                if response and response.text:
                    self.usage_stats[self.AVAILABLE_MODELS[self.current_model_idx]] += 1
                    return response.text.strip()
                    
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    logger.warning(f"Quota exceeded (Narrative) for {self.AVAILABLE_MODELS[self.current_model_idx]}. Rotating...")
                    self._get_model(rotate=True)
                    continue
                logger.error(f"Narrative summarization error: {e}")
                return "Narrative generation failed."
                
        return "All models exhausted for narrative."
