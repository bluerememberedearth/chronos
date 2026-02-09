import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingest.gemini_perception import GeminiPerceptionEngine

class TestModelRotation(unittest.TestCase):
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_rotation_on_429(self, mock_configure, mock_gen_model):
        # Setup engine
        engine = GeminiPerceptionEngine(api_key="fake_key")
        engine.min_call_interval = 0  # speed up test
        
        # Mock the model's generate_content to throw 429 twice, then succeed
        mock_model_instance = MagicMock()
        
        # Create a side effect that raises 429 for the first two models, then returns a valid response
        def side_effect(*args, **kwargs):
            if engine.current_model_idx == 0:
                 # First model fails
                 raise Exception("Resource has been exhausted (e.g. check quota). [429]")
            if engine.current_model_idx == 1:
                 # Second model fails (after rotation)
                 raise Exception("Resource has been exhausted (e.g. check quota). [429]")
            
            # Third model succeeds
            mock_res = MagicMock()
            mock_res.text = '{"broadcast_state": "LIVE_GAME", "game_state": {"game_time": "10:00"}}'
            return mock_res

        mock_model_instance.generate_content.side_effect = side_effect
        mock_gen_model.return_value = mock_model_instance
        
        # We need to ensure _get_model returns our mock_model_instance
        # The engine caches models in self.models
        # So we'll just let it create them and it will use the mocked GenerativeModel class
        
        import numpy as np
        fake_frame = np.zeros((1080, 1920, 3), dtype='uint8')
        
        # Initial model should be idx 0
        self.assertEqual(engine.current_model_idx, 0)
        
        # Run analysis
        result = engine.analyze_frame(fake_frame)
        
        # Verify it rotated to index 2 (third model)
        self.assertEqual(engine.current_model_idx, 2)
        self.assertEqual(result['model_used'], engine.AVAILABLE_MODELS[2])
        print(f"✅ Success! Engine rotated from {engine.AVAILABLE_MODELS[0]} -> {engine.AVAILABLE_MODELS[1]} and settled on {engine.AVAILABLE_MODELS[2]}")

if __name__ == "__main__":
    unittest.main()
