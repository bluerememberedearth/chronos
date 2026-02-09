import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import time
from intelligence.game_memory import GameMemory, TeamComp

class TestCognitiveReflection(unittest.TestCase):
    def setUp(self):
        blue = TeamComp.from_string("Gnar,LeeSin,Ahri,Jinx,Thresh")
        red = TeamComp.from_string("Jax,Viego,Syndra,Kaisa,Nautilus")
        self.memory = GameMemory(blue, red)
        self.start_time = time.time()

    def test_valid_progression(self):
        """Test that normal game progression is accepted."""
        # Initial state
        state1 = {
            "game_time": "05:00",
            "left_team": {"gold_k": 8.0, "kills": 2, "towers": 0, "dragons": 0},
            "right_team": {"gold_k": 7.5, "kills": 1, "towers": 0, "dragons": 0}
        }
        self.assertTrue(self.memory.add_observation(state1, {}, self.start_time))
        
        # Progression: Gold increases, kills increase
        state2 = {
            "game_time": "06:00",
            "left_team": {"gold_k": 9.5, "kills": 3, "towers": 0, "dragons": 0},
            "right_team": {"gold_k": 8.8, "kills": 1, "towers": 0, "dragons": 1}
        }
        self.assertTrue(self.memory.add_observation(state2, {}, self.start_time + 60))
        
        # Verify state was added
        self.assertEqual(len(self.memory.all_snapshots), 2)
        latest = self.memory.all_snapshots[-1]
        self.assertEqual(latest.blue_kill, 3) if hasattr(latest, 'blue_kill') else None
        
    def test_invalid_gold_drop(self):
        """Test rejection of significant gold drop."""
        # Initial state
        self.memory.add_observation({
            "game_time": "10:00",
            "left_team": {"gold_k": 15.0},
            "right_team": {"gold_k": 14.0}
        }, {}, self.start_time)
        
        # Invalid: Blue gold drops to 12k (impossible)
        invalid_state = {
            "game_time": "10:30",
            "left_team": {"gold_k": 12.0}, # Drop of 3k
            "right_team": {"gold_k": 14.5}
        }
        self.assertFalse(self.memory.add_observation(invalid_state, {}, self.start_time + 30))
        
        # Verify state NOT added
        self.assertEqual(len(self.memory.all_snapshots), 1)
        self.assertEqual(self.memory.all_snapshots[-1].blue_gold, 15.0)

    def test_invalid_objective_reversal(self):
        """Test rejection of objectives disappearing."""
        # Initial: Blue has 2 dragons
        self.memory.add_observation({
            "game_time": "20:00",
            "left_team": {"dragons": 2, "towers": 3},
            "right_team": {"dragons": 1, "towers": 1}
        }, {}, self.start_time)
        
        # Invalid: Blue dragons back to 1
        invalid_state = {
            "game_time": "20:30",
            "left_team": {"dragons": 1, "towers": 3}, # Drop
            "right_team": {"dragons": 1, "towers": 1}
        }
        self.assertFalse(self.memory.add_observation(invalid_state, {}, self.start_time + 30))
        
        # Invalid: Red tower count drops? Assuming towers = destroyed count
        # In GameMemory logic: "if snapshot.red_towers < prev.red_towers: REJECT"
        # Since I assumed towers=destroyed, they should only go up.
        invalid_towers = {
             "game_time": "20:30",
            "left_team": {"dragons": 2, "towers": 3},
            "right_team": {"dragons": 1, "towers": 0} # Drop from 1 to 0
        }
        self.assertFalse(self.memory.add_observation(invalid_towers, {}, self.start_time + 30))

if __name__ == '__main__':
    unittest.main()
