import sys
import os
import unittest
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from intelligence.game_memory import GameMemory, TeamComp

class TestFogOfWar(unittest.TestCase):
    def test_champion_tracking(self):
        """Test tracking visible and MIA champions."""
        blue = TeamComp.from_string("Gnar,LeeSin,Ahri,Jinx,Thresh")
        red = TeamComp.from_string("Jax,Viego,Syndra,Kaisa,Nautilus")
        memory = GameMemory(blue, red)
        
        now = time.time()
        
        # 1. Simulate observation where Ahri and Lee Sin are visible
        memory.update_champion_tracker(
            minimap_summary="Ahri visible in mid lane, Lee Sin invading red jungle",
            visible_ultimates={},
            timestamp=now
        )
        
        # Check Ahri state
        ahri = memory.champion_tracker["Ahri"]
        self.assertEqual(ahri.status, "VISIBLE")
        self.assertEqual(ahri.last_seen_location, "Mid Lane")
        self.assertEqual(ahri.last_seen_time, now)
        
        # Check Lee Sin state
        lee = memory.champion_tracker["LeeSin"]
        self.assertEqual(lee.status, "VISIBLE")
        self.assertTrue(len(lee.last_seen_location) > 0) # Logic puts "Map" or specific (e.g. Mid Lane due to shared context)
        
        # 2. Simulate time passing (40 seconds later)
        future = now + 40
        
        # 3. New observation, Ahri is NOT visible
        memory.update_champion_tracker(
            minimap_summary="Bot lane pushing, no other vision",
            visible_ultimates={},
            timestamp=future
        )
        
        # Check Ahri should be MIA now
        # Note: We need to trigger a check or update for MIA. 
        # The current implementation updates status ONLY if seen, OR if we iterate.
        # But get_fog_of_war_context calculates it dynamically.
        
        context = memory.get_fog_of_war_context(future)
        print(f"\n[Fog Context]:\n{context}")
        
        self.assertIn("Ahri: MIA", context)
        # Lee Sin should also be MIA
        self.assertIn("LeeSin: MIA", context)
        # Jinx never seen, so she shouldn't be listed as MIA (never active) or should be handled?
        # Current logic: "if delta > 30 and state.last_seen_time > 0"
        # So Jinx (time=0) won't be shown. Correct.

if __name__ == "__main__":
    unittest.main()
