
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)

class GameStateHistory:
    def __init__(self, max_history_size=20):
        # Store (timestamp_unix, state_dict)
        self.history = deque(maxlen=max_history_size)

    def add_state(self, state, timestamp):
        """
        Adds a parsed game state to history.
        """
        if "error" in state:
            return
            
        # Optional: Deduplicate if state hasn't changed meaningfully?
        # For now, just append.
        self.history.append({
            "timestamp": timestamp,
            "state": state
        })

    def get_summary(self, seconds_lookback=120):
        """
        Returns a text summary of key events in the last N seconds.
        Focuses on Delta (Change) in Gold, Kills, and Objectives.
        """
        if len(self.history) < 2:
            return "No sufficient history yet."

        now = self.history[-1]["timestamp"]
        start_time = now - seconds_lookback
        
        # Find start state
        start_state = None
        for entry in self.history:
            if entry["timestamp"] >= start_time:
                start_state = entry["state"]
                break
        
        if not start_state:
            start_state = self.history[0]["state"]

        current_state = self.history[-1]["state"]
        
        # Calculate Deltas
        try:
            b_gold_delta = current_state.get("left_team", {}).get("gold_k", 0) - start_state.get("left_team", {}).get("gold_k", 0)
            r_gold_delta = current_state.get("right_team", {}).get("gold_k", 0) - start_state.get("right_team", {}).get("gold_k", 0)
            
            b_kills_delta = current_state.get("left_team", {}).get("kills", 0) - start_state.get("left_team", {}).get("kills", 0)
            r_kills_delta = current_state.get("right_team", {}).get("kills", 0) - start_state.get("right_team", {}).get("kills", 0)
            
            towers_delta = (current_state.get("left_team", {}).get("towers", 0) - start_state.get("left_team", {}).get("towers", 0)) + \
                           (current_state.get("right_team", {}).get("towers", 0) - start_state.get("right_team", {}).get("towers", 0))

            summary = [
                f"Validation Window: {seconds_lookback}s",
                f"Gold Change: Left {b_gold_delta:+.1f}k, Right {r_gold_delta:+.1f}k",
                f"Kills Added: Left +{b_kills_delta}, Right +{r_kills_delta}",
                f"Towers Taken: {towers_delta}",
                f"Current Minimap: {current_state.get('minimap_summary', 'N/A')}"
            ]
            
            return "\n".join(summary)

        except Exception as e:
            logger.error(f"History Summary Error: {e}")
            return "Error generating history."
