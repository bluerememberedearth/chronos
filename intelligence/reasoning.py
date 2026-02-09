
import logging
import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Re-use the OpenAI client setup from perception or make a new one?
# It's cleaner to have its own client or pass initialized one.
# For V2, we'll instantiate a new client here for simplicity as they might use different keys/models.

try:
    from openai import OpenAI
    SILICONFLOW_AVAILABLE = True
except ImportError:
    SILICONFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class ReasoningEngine:
    def __init__(self, model: str = "Qwen/Qwen2.5-72B-Instruct"):
        load_dotenv()
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.model = model
        
        if SILICONFLOW_AVAILABLE and self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.siliconflow.com/v1"
            )
        else:
            self.client = None
            logger.warning("ReasoningEngine: SiliconFlow client not initialized.")

    def predict_winner(self, game_state: Dict[str, Any], history_summary: str = "No history available") -> Dict[str, Any]:
        """
        Analyzes the current game state and predicts the winner, considering history and minimap.
        
        Args:
            game_state: A dictionary containing game metrics (gold, kills, time, minimap etc.)
            history_summary: Text summary of recent changes.
            
        Returns:
            Dict containing:
            - winner: "BLUE" | "RED"
            - confidence: float (0.0 - 1.0)
            - reasoning: str
            - fight_prediction: "Blue likely to win next fight" (optional)
        """
        if not self.client:
            return {"error": "Client not initialized", "winner": "UNKNOWN", "confidence": 0.0}

        try:
            # Construct a prompt that asks for strategic analysis
            minimap = json.dumps(game_state.get('minimap', {}), indent=2)
            
            prompt = f"""
            You are an expert League of Legends analyst. 
            Analyze the following game state at time {game_state.get('game_time', 'UNKNOWN')}:
            
            Historical Context (Last 2 mins):
            {history_summary}
            
            Current State (Left=Blue, Right=Red):
            Left Team (Blue):
            - Kills: {game_state.get('left_team', {}).get('kills', 0)}
            - Gold: {game_state.get('left_team', {}).get('gold_k', 0)}k
            - Towers: {game_state.get('left_team', {}).get('towers', 0)}
            - Dragons: {game_state.get('left_team', {}).get('dragons', 0)}
            - Ultimate Count: {game_state.get('left_team', {}).get('ult_count', 'Unknown')} (Green Dots)
            
            Right Team (Red):
            - Kills: {game_state.get('right_team', {}).get('kills', 0)}
            - Gold: {game_state.get('right_team', {}).get('gold_k', 0)}k
            - Towers: {game_state.get('right_team', {}).get('towers', 0)}
            - Dragons: {game_state.get('right_team', {}).get('dragons', 0)}
            - Ultimate Count: {game_state.get('right_team', {}).get('ult_count', 'Unknown')} (Green Dots)
            
            Key Objective Spawns:
            - Next Objective: {game_state.get('next_objective', {}).get('type', 'Unknown')}
            - Timer: {game_state.get('next_objective', {}).get('timer', 'Unknown')} (Usually fight happens < 1:00)
            
            Minimap & Positional Analysis:
            {minimap}
            
            Task:
            1. Who wins the NEXT fight based on positioning and recent momentum?
            2. Who wins the GAME overall?
            
            Return ONLY a valid JSON object with:
            {{
                "next_fight_winner": "BLUE" or "RED" or "NONE",
                "winner": "BLUE" or "RED",
                "confidence": <float between 0.0 and 1.0>,
                "reasoning": "<explanation citing specific minimap movements or gold leads>"
            }}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
                # response_format={"type": "json_object"} # Valid? Let's skip to be safe and parse manually
            )

            content = response.choices[0].message.content.strip()
            
            # Clean markdown
            if "```" in content:
                import re
                content = re.sub(r"```json|```", "", content).strip()
            
            return json.loads(content)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e), "winner": "UNKNOWN", "confidence": 0.0}
