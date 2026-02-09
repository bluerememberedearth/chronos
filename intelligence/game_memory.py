"""
Game Memory - Short-term and Medium-term Context

Maintains structured context that builds up over time:

SHORT-TERM (last 2-3 observations):
  - Recent gold changes
  - Recent fights/kills
  - Latest minimap positions

MEDIUM-TERM (entire game):
  - Champion compositions (input at start)
  - Cumulative trends (gold velocity, kill patterns)
  - Key events (dragon takes, tower falls, item spikes)
  - Fight history

This context is passed to VLM for better predictions.
"""

import time
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TeamComp:
    """Champion composition for a team."""
    top: str
    jungle: str
    mid: str
    adc: str
    support: str
    
    @classmethod
    def from_string(cls, champs: str) -> 'TeamComp':
        """Parse 'Gnar,LeeSin,Ahri,Jinx,Thresh' format."""
        parts = [c.strip() for c in champs.split(',')]
        if len(parts) != 5:
            raise ValueError(f"Expected 5 champions, got {len(parts)}")
        return cls(top=parts[0], jungle=parts[1], mid=parts[2], adc=parts[3], support=parts[4])
    
    def __str__(self):
        return f"{self.top}/{self.jungle}/{self.mid}/{self.adc}/{self.support}"
    
    def as_dict(self) -> dict:
        return {"top": self.top, "jungle": self.jungle, "mid": self.mid, 
                "adc": self.adc, "support": self.support}


@dataclass
class GameEvent:
    """A notable game event."""
    timestamp: float
    game_time: str
    event_type: str  # "kill", "dragon", "tower", "fight", "item_spike"
    description: str


@dataclass  
class GameSnapshot:
    """Single point-in-time game state."""
    timestamp: float
    game_time: str
    blue_gold: float
    red_gold: float
    blue_kills: int
    red_kills: int
    blue_towers: int
    red_towers: int
    blue_dragons: int
    red_dragons: int
    minimap_summary: str = ""
    prediction: str = ""
    confidence: float = 0.0

@dataclass
class GameChapter:
    """A compressed narrative summary of a game phase."""
    start_time: str # "00:00"
    end_time: str   # "15:00"
    summary: str    # "Early game saw Blue side dominance..."
    key_events: List[str]

class GameNarrative:
    """
    Maintains the 'story' of the game in compressed chapters.
    """
    def __init__(self):
        self.chapters: List[GameChapter] = []
        self.current_chapter_summary: str = ""
        
    def add_chapter(self, summary: str, start_time: str, end_time: str, events: List[str]):
        self.chapters.append(GameChapter(start_time, end_time, summary, events))
        
    def get_full_narrative(self) -> str:
        if not self.chapters:
            return "No narrative history yet."
            
        lines = ["Game Narrative:"]
        for i, chap in enumerate(self.chapters):
            lines.append(f"Chapter {i+1} ({chap.start_time}-{chap.end_time}):")
            lines.append(f"  {chap.summary}")
        return "\n".join(lines)



    
@dataclass
class ChampionState:
    """State of a specific champion."""
    name: str
    last_seen_time: float
    last_seen_location: str  # "Bot Lane", "Dragon Pit", etc. from minimap_summary
    status: str = "UNKNOWN"  # "VISIBLE", "MIA", "DEAD"

from intelligence.data_dragon import DataDragon

class GameMemory:
    def __init__(self, blue_comp: TeamComp, red_comp: TeamComp):
        # Medium-term: Static context
        self.blue_comp = blue_comp
        self.red_comp = red_comp
        self.game_start_time: Optional[float] = None
        
        # Static Knowledge Base (DataDragon)
        try:
            self.static_data = DataDragon()
        except Exception as e:
            print(f"Warning: Static data load failed: {e}")
            self.static_data = None
        
        # Fog of War: Tracker
        self.champion_tracker: Dict[str, ChampionState] = {}
        # Initialize tracker
        for champ in blue_comp.as_dict().values():
            self.champion_tracker[champ] = ChampionState(champ, 0, "Base", "UNKNOWN")
        for champ in red_comp.as_dict().values():
            self.champion_tracker[champ] = ChampionState(champ, 0, "Base", "UNKNOWN")

        # Short-term: Recent observations (last 5)
        self.recent_snapshots: deque = deque(maxlen=5)
        
        # Medium-term: All observations for trend analysis
        self.all_snapshots: List[GameSnapshot] = []
        
        # Medium-term: Key events
        self.events: List[GameEvent] = []
        
        # Long-term: Narrative Memory
        self.narrative = GameNarrative()
        
        # Computed trends
        self.gold_velocity: float = 0.0  # Gold diff change per minute
        self.momentum: str = "NEUTRAL"  # BLUE, RED, or NEUTRAL
    
    def get_static_context(self) -> str:
        """Get static data context for all champions in the game."""
        if not self.static_data:
            return ""
            
        lines = ["Champion Abilities & Context:"]
        
        # Blue Team
        lines.append("BLUE TEAM:")
        for role, name in self.blue_comp.as_dict().items():
            ctx = self.static_data.get_champion_context(name)
            lines.append(f"- {role.upper()}: {ctx}")
            
        # Red Team
        lines.append("RED TEAM:")
        for role, name in self.red_comp.as_dict().items():
            ctx = self.static_data.get_champion_context(name)
            lines.append(f"- {role.upper()}: {ctx}")
            
        return "\n".join(lines)

    def update_champion_tracker(self, minimap_summary: str, visible_ultimates: dict, timestamp: float):
        """Update champion states based on new observation."""
        if not minimap_summary: return
        
        # Simple heuristic: If champion name appears in summary, mark as seen
        summary_lower = minimap_summary.lower()
        # Create a normalized version for matching strict names like "LeeSin" vs "Lee Sin"
        summary_normalized = summary_lower.replace(" ", "")
        
        for champ_name, state in self.champion_tracker.items():
            # Check basic name or normalized name (e.g. "LeeSin" in "leesinvading")
            # Ideally we want robust matching, but stripping spaces helps "LeeSin" find "Lee Sin"
            name_clean = champ_name.lower().replace(" ", "")
            
            # Match against original text OR normalized text
            # "Ahri" in "ahri visible" -> True
            # "LeeSin" (clean) in "leesinvading" (normalized) -> True
            # "LeeSin" (clean) in "lee sin invading" (normalized -> leesininvading) -> True
            
            is_visible = (champ_name.lower() in summary_lower) or (name_clean in summary_normalized)
            
            if is_visible:
                state.last_seen_time = timestamp
                state.status = "VISIBLE"
                # Try to extract location context (very basic)
                # "Ahri visible in mid lane" -> "mid lane"
                state.last_seen_location = "Map" # Default
                
                # Update location if specific keywords found near name could be future work
                if "bot" in summary_lower: state.last_seen_location = "Bot Side/Lane"
                elif "top" in summary_lower: state.last_seen_location = "Top Side/Lane"
                elif "mid" in summary_lower: state.last_seen_location = "Mid Lane"
                elif "dragon" in summary_lower: state.last_seen_location = "Dragon Pit"
                elif "baron" in summary_lower: state.last_seen_location = "Baron Pit"
            else:
                # If not seen in a while, status becomes MIA
                if timestamp - state.last_seen_time > 30 and state.last_seen_time > 0:
                    state.status = "MIA"

    def get_fog_of_war_context(self, current_time: float) -> str:
        """Return MIA status for champions not seen recently."""
        lines = ["Fog of War / MIA Status:"]
        matches = 0
        for champ_name, state in self.champion_tracker.items():
            delta = current_time - state.last_seen_time
            if delta > 30 and state.last_seen_time > 0: # Missing for > 30s
                lines.append(f"- {champ_name}: MIA (Last seen {int(delta)}s ago at {state.last_seen_location})")
                matches += 1
        
        if matches == 0: return ""
        return "\n".join(lines)
    
    def add_narrative_chapter(self, summary: str):
        """Add a summarized chapter to long-term memory."""
        if not self.all_snapshots: return
        start = self.all_snapshots[0].game_time
        end = self.all_snapshots[-1].game_time
        events = [e.description for e in self.events]
        self.narrative.add_chapter(summary, start, end, events)
        # Clear detailed snapshots/events after summarization?
        # For now, we keep them but could prune to save memory.

    def get_narrative_context(self) -> str:
        """Get high-level story context."""
        return self.narrative.get_full_narrative()

    def validate_state(self, snapshot: GameSnapshot) -> bool:
        """
        Cognitive Reflection: Check if new state is physically possible.
        Returns True if valid, False if impossible (hallucination).
        """
        if not self.all_snapshots:
            return True
            
        prev = self.all_snapshots[-1]
        
        # Rule 1: Gold should not decrease significantly (allow -1k for sell/undo/noise)
        if snapshot.blue_gold < prev.blue_gold - 1.0:
            print(f"⚠️ REJECTED: Blue gold drop {prev.blue_gold}->{snapshot.blue_gold}")
            return False
        if snapshot.red_gold < prev.red_gold - 1.0:
            print(f"⚠️ REJECTED: Red gold drop {prev.red_gold}->{snapshot.red_gold}")
            return False
            
        # Rule 2: Objectives cannot be un-taken (monotonic increasing)
        # Note: Towers = destroyed towers (0-11)
        if snapshot.blue_towers < prev.blue_towers:
            print(f"⚠️ REJECTED: Blue towers decreased {prev.blue_towers}->{snapshot.blue_towers}")
            return False
        if snapshot.red_towers < prev.red_towers:
            print(f"⚠️ REJECTED: Red towers decreased {prev.red_towers}->{snapshot.red_towers}")
            return False
            
        if snapshot.blue_dragons < prev.blue_dragons:
            print(f"⚠️ REJECTED: Blue dragons decreased {prev.blue_dragons}->{snapshot.blue_dragons}")
            return False
        if snapshot.red_dragons < prev.red_dragons:
            print(f"⚠️ REJECTED: Red dragons decreased {prev.red_dragons}->{snapshot.red_dragons}")
            return False
            
        # Rule 3: Kills cannot decrease
        if snapshot.blue_kills < prev.blue_kills:
            print(f"⚠️ REJECTED: Blue kills decreased {prev.blue_kills}->{snapshot.blue_kills}")
            return False
        if snapshot.red_kills < prev.red_kills:
            return False
            
        return True

    def add_observation(self, game_state: dict, prediction: dict, timestamp: float) -> bool:
        """Add a new observation to memory. Returns False if rejected."""
        if self.game_start_time is None:
            self.game_start_time = timestamp
        
        left = game_state.get('left_team', {})
        right = game_state.get('right_team', {})
        
        snapshot = GameSnapshot(
            timestamp=timestamp,
            game_time=game_state.get('game_time', '00:00'),
            blue_gold=left.get('gold_k', 0),
            red_gold=right.get('gold_k', 0),
            blue_kills=left.get('kills', 0),
            red_kills=right.get('kills', 0),
            blue_towers=left.get('towers', 0),
            red_towers=right.get('towers', 0),
            blue_dragons=left.get('dragons', 0),
            red_dragons=right.get('dragons', 0),
            minimap_summary=game_state.get('minimap_summary', ''),
            prediction=prediction.get('winner', ''),
            confidence=prediction.get('confidence', 0)
        )
        
        # Cognitive Reflection
        if not self.validate_state(snapshot):
            return False
        
        self.recent_snapshots.append(snapshot)
        self.all_snapshots.append(snapshot)
        
        # Update trends
        self._update_trends()
        
        # Detect events
        self._detect_events(snapshot)
        
        # Update Fog of War Tracker
        self.update_champion_tracker(
            snapshot.minimap_summary,
            prediction.get("visible_ultimates", {}),
            timestamp
        )
        return True
    
    def _update_trends(self):
        """Update computed trends from snapshots."""
        if len(self.all_snapshots) < 2:
            return
        
        # Gold velocity: change in gold diff over last 3 observations
        recent = list(self.recent_snapshots)[-3:]
        if len(recent) >= 2:
            old_diff = recent[0].blue_gold - recent[0].red_gold
            new_diff = recent[-1].blue_gold - recent[-1].red_gold
            time_delta = (recent[-1].timestamp - recent[0].timestamp) / 60  # minutes
            if time_delta > 0:
                self.gold_velocity = (new_diff - old_diff) / time_delta
        
        # Momentum based on gold velocity and recent kills
        if self.gold_velocity > 0.5:
            self.momentum = "BLUE"
        elif self.gold_velocity < -0.5:
            self.momentum = "RED"
        else:
            self.momentum = "NEUTRAL"
    
    def _detect_events(self, snapshot: GameSnapshot):
        """Detect notable events from state changes."""
        if len(self.all_snapshots) < 2:
            return
        
        prev = self.all_snapshots[-2]
        
        # Kill event
        blue_kills_diff = snapshot.blue_kills - prev.blue_kills
        red_kills_diff = snapshot.red_kills - prev.red_kills
        if blue_kills_diff + red_kills_diff >= 3:
            self.events.append(GameEvent(
                timestamp=snapshot.timestamp,
                game_time=snapshot.game_time,
                event_type="fight",
                description=f"Teamfight: Blue +{blue_kills_diff}, Red +{red_kills_diff}"
            ))
        
        # Dragon event
        if snapshot.blue_dragons > prev.blue_dragons:
            self.events.append(GameEvent(
                timestamp=snapshot.timestamp,
                game_time=snapshot.game_time,
                event_type="dragon",
                description=f"Blue secured dragon #{snapshot.blue_dragons}"
            ))
        if snapshot.red_dragons > prev.red_dragons:
            self.events.append(GameEvent(
                timestamp=snapshot.timestamp,
                game_time=snapshot.game_time,
                event_type="dragon",
                description=f"Red secured dragon #{snapshot.red_dragons}"
            ))
        
        # Tower event
        if snapshot.blue_towers > prev.blue_towers:
            self.events.append(GameEvent(
                timestamp=snapshot.timestamp,
                game_time=snapshot.game_time,
                event_type="tower",
                description=f"Blue destroyed tower #{snapshot.blue_towers}"
            ))
        if snapshot.red_towers > prev.red_towers:
            self.events.append(GameEvent(
                timestamp=snapshot.timestamp,
                game_time=snapshot.game_time,
                event_type="tower",
                description=f"Red destroyed tower #{snapshot.red_towers}"
            ))
    
    def get_short_term_context(self) -> str:
        """Get context from last 2-3 observations."""
        if not self.recent_snapshots:
            return "No observations yet."
        
        recent = list(self.recent_snapshots)[-3:]
        lines = []
        for snap in recent:
            gold_diff = snap.blue_gold - snap.red_gold
            lines.append(f"- {snap.game_time}: Gold diff {gold_diff:+.1f}k, "
                        f"Kills {snap.blue_kills}-{snap.red_kills}")
        
        return "Recent observations:\n" + "\n".join(lines)
    
    def get_medium_term_context(self) -> str:
        """Get cumulative game context."""
        lines = [
            f"Team Compositions:",
            f"  BLUE: {self.blue_comp}",
            f"  RED: {self.red_comp}",
        ]
        
        if self.all_snapshots:
            latest = self.all_snapshots[-1]
            gold_diff = latest.blue_gold - latest.red_gold
            lines.extend([
                f"\nGame State at {latest.game_time}:",
                f"  Gold: BLUE {latest.blue_gold:.1f}k vs RED {latest.red_gold:.1f}k (diff: {gold_diff:+.1f}k)",
                f"  Kills: {latest.blue_kills}-{latest.red_kills}",
                f"  Objectives: Blue {latest.blue_dragons}D/{latest.blue_towers}T, "
                f"Red {latest.red_dragons}D/{latest.red_towers}T",
            ])
        
        if self.gold_velocity != 0:
            lines.append(f"\nTrends:")
            lines.append(f"  Gold velocity: {self.gold_velocity:+.1f}k/min (momentum: {self.momentum})")
        
        if self.events:
            recent_events = self.events[-5:]
            lines.append(f"\nRecent Events:")
            for event in recent_events:
                lines.append(f"  [{event.game_time}] {event.description}")
        
        return "\n".join(lines)
        
    def get_full_context(self) -> str:
        """Get complete context for VLM prompt."""
        current_time = time.time()
        # Include Static Data at the TOP for grounding
        return (f"{self.get_static_context()}\n\n"
                f"{self.get_medium_term_context()}\n\n"
                f"{self.get_fog_of_war_context(current_time)}\n\n"
                f"{self.get_short_term_context()}")
    
    def get_stats(self) -> dict:
        """Get summary statistics."""
        return {
            "observations": len(self.all_snapshots),
            "events": len(self.events),
            "momentum": self.momentum,
            "gold_velocity": self.gold_velocity
        }


# Test
if __name__ == "__main__":
    blue = TeamComp.from_string("Gnar,LeeSin,Ahri,Jinx,Thresh")
    red = TeamComp.from_string("Jax,Viego,Syndra,Kaisa,Nautilus")
    
    memory = GameMemory(blue, red)
    
    # Simulate observations
    t = time.time()
    memory.add_observation(
        {"game_time": "05:00", "left_team": {"gold_k": 8.5, "kills": 2, "towers": 0, "dragons": 0},
         "right_team": {"gold_k": 7.9, "kills": 1, "towers": 0, "dragons": 0}},
        {"winner": "BLUE", "confidence": 0.55}, t
    )
    memory.add_observation(
        {"game_time": "08:00", "left_team": {"gold_k": 14.2, "kills": 5, "towers": 1, "dragons": 1},
         "right_team": {"gold_k": 12.1, "kills": 2, "towers": 0, "dragons": 0}},
        {"winner": "BLUE", "confidence": 0.68}, t + 180
    )
    
    print("=== FULL CONTEXT ===")
    print(memory.get_full_context())
    print("\n=== STATS ===")
    print(memory.get_stats())
