import time
import logging
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class ContextWindow:
    """
    The 'Hippocampus' of Chronos.
    Aggregates high-frequency sensor data into meaningful 'Episodes' and 'Trends'.
    
    Silicon Advantage: Perfect retention of exact timestamps and values (Gold, XP).
    Human Inspiration: Structuring data into 'Events' rather than just a stream of numbers.
    """
    def __init__(self, max_history_seconds=600):
        self.max_history_seconds = max_history_seconds
        
        # 1. Raw Timeline (High Frequency)
        # Stores every 'tick' of data for detailed reconstruction if needed.
        # Format: (timestamp, {data_snapshot})
        self.timeline = deque()
        
        # 2. Event Log (Semantic Memory)
        # Significant state changes triggers.
        # e.g., "Gold Diff flipped positive", "Dragon Taken".
        self.events: List[Dict] = []
        
        # 3. Derived State (The Concept)
        # Running averages / trends
        self.gold_velocity = 0.0 # Gold change per second (last 30s)
        self.last_gold_diff = 0
        self.last_update_time = time.time()

    def update(self, perception_bundle: Dict[str, Any]):
        """
        Ingest a bundle of data from the Perception Cockpit.
        bundle = {
            'timestamp': float,
            'scoreboard': {...},
            'minimap': [...],
            'objectives': {...},
            'champions': [...]
        }
        """
        now = perception_bundle.get('timestamp', time.time())
        self.timeline.append(perception_bundle)
        
        # Prune old history
        while self.timeline and (now - self.timeline[0]['timestamp'] > self.max_history_seconds):
            self.timeline.popleft()
            
        # Analyze for Events (The "Lizard Brain" trigger for memory formation)
        self._analyze_derivatives(perception_bundle)
        
    def _analyze_derivatives(self, current: Dict):
        """
        Calculates rates of change to detect events.
        """
        # Example: Gold Swing
        current_gold_diff = current.get('scoreboard', {}).get('gold_diff', 0)
        dt = current['timestamp'] - self.last_update_time
        
        if dt > 1.0: # Update derivatives every second approx
            delta = current_gold_diff - self.last_gold_diff
            self.gold_velocity = delta / dt
            
            # Semantic Event: Huge Swing
            if abs(delta) > 2000: # 2k gold swing in short time? Fight happened.
                self.add_event("HUGE_GOLD_SWING", {"delta": delta, "new_diff": current_gold_diff})
            
            self.last_gold_diff = current_gold_diff
            self.last_update_time = current['timestamp']

    def add_event(self, type: str, payload: Dict):
        event = {
            "timestamp": time.time(),
            "type": type,
            "data": payload
        }
        self.events.append(event)
        logger.info(f"Context Event: {type} - {payload}")

    def get_context_summary(self, window_seconds: int = 60) -> str:
        """
        Synthesizes a text prompt for the LLM describing the recent game state.
        This covers the gap between "Raw Data" and "Reasoning".
        """
        # 1. Get relevant slice
        # In a real impl, we'd filter self.timeline
        
        # 2. Formulate Narrative
        summary = f"--- GAME STATE SUMMARY (Last {window_seconds}s) ---\n"
        
        # Global
        current = self.timeline[-1] if self.timeline else {}
        sb = current.get('scoreboard', {})
        time_str = sb.get('time', 'Unknown')
        gold = sb.get('gold_diff', 0)
        
        summary += f"Time: {time_str} | Gold Diff: {gold} (Velocity: {self.gold_velocity:.1f}/s)\n"
        
        # Events
        recent_events = [e for e in self.events if (time.time() - e['timestamp']) < window_seconds]
        if recent_events:
            summary += "Recent Significant Events:\n"
            for e in recent_events:
                summary += f"- [{e['timestamp']:.1f}] {e['type']}: {e['data']}\n"
        else:
            summary += "No major events detected recently.\n"
            
        # Minimap (Attention)
        minimap_alerts = current.get('minimap', [])
        if minimap_alerts:
            summary += f"Map Activity: {len(minimap_alerts)} active alerts (Fighting/Ping).\n"
            
        return summary
