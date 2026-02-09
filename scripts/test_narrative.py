import sys
import os
import unittest
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from intelligence.game_memory import GameMemory, TeamComp, GameEvent
from ingest.gemini_perception import GeminiPerceptionEngine

class TestNarrativeCompression(unittest.TestCase):
    def test_game_narrative_structure(self):
        """Test GameNarrative class functionality."""
        blue = TeamComp.from_string("Top,Jng,Mid,ADC,Sup")
        red = TeamComp.from_string("Top,Jng,Mid,ADC,Sup")
        memory = GameMemory(blue, red)
        
        # Simulate a full narrative flow
        memory.narrative.add_chapter(
            summary="Blue team dominated early game with a level 1 invade.",
            start_time="00:00",
            end_time="05:00",
            events=["Blue First Blood", "Red Jungle Invaded"]
        )
        
        context = memory.get_narrative_context()
        print(f"\n[Narrative Context Output]:\n{context}")
        
        self.assertIn("Chapter 1 (00:00-05:00):", context)
        self.assertIn("Blue team dominated early", context)

    @patch('google.generativeai.GenerativeModel')
    def test_summarize_narrative_call(self, mock_gen_model):
        """Test the LLM call for summarization."""
        engine = GeminiPerceptionEngine(api_key="fake")
        engine.min_call_interval = 0
        
        # Mock LLM response
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value.text = "Mocked Narrative Summary."
        mock_gen_model.return_value = mock_instance
        
        # Test input events
        events = "- 02:30: First Blood (Blue)\n- 05:00: Dragon (Red)"
        summary = engine.summarize_narrative(events)
        
        print(f"\n[Generated Summary]: {summary}")
        self.assertEqual(summary, "Mocked Narrative Summary.")
        
        # Verify prompt contained history
        args, _ = mock_instance.generate_content.call_args
        self.assertIn(events, args[0])

if __name__ == "__main__":
    unittest.main()
