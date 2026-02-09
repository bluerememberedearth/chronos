
import cv2
import os
import sys
import logging
import json
from dotenv import load_dotenv

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try imports
try:
    from chronos_v3.ingest.perception import HybridPerceptionEngine
except ImportError:
    # If package structure fails, try direct import (e.g. if running as script inside dir)
    try:
        from ingest.perception import HybridPerceptionEngine
    except ImportError:
        raise

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("GameStateVerify")

def verify_game_state():
    load_dotenv()
    engine = HybridPerceptionEngine()
    
    # Path to our captured mid-game frame
    # We used sample_seek1500_000.jpg
    # Assuming user didn't rename
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_frame_path = os.path.join(base_dir, "tests/fixtures/huya_vod/midgame/sample_seek1500_000.jpg")
    
    if not os.path.exists(target_frame_path):
        logger.error(f"Test frame not found: {target_frame_path}")
        return

    logger.info(f"Loading frame: {target_frame_path}")
    frame = cv2.imread(target_frame_path)
    
    if frame is None:
        logger.error("Failed to load image")
        return

    logger.info("Calling analyze_game_state...")
    state = engine.analyze_game_state(frame)
    
    logger.info("--- Extracted Game State ---")
    print(json.dumps(state, indent=2))
    
    # Simple validation against known values (from manual inspection)
    # Start Time: 1500s -> ~25 mins? Wait, frame says 17:28
    # 1500s is 25 mins. The seek might have been approximate or stream buffering affected timing.
    # Ah, the frame shows "17:28". So the VOD time is different from game time? Yes, likely draft + pause + buffer.
    # Let's validate against what we SEE in the image.
    
    if "error" in state:
        logger.error("Extraction Failed")
        return

    # Check approximate correctness
    try:
        # Check basic structure
        blue = state.get("blue_team", {})
        red = state.get("red_team", {})
        
        logger.info(f"Game Time: {state.get('game_time')}")
        logger.info(f"Blue Gold: {blue.get('gold_k')}k (Exp: ~30.1)")
        logger.info(f"Red Gold: {red.get('gold_k')}k (Exp: ~30.8)")
        
        # Validations
        if abs(blue.get('gold_k', 0) - 30.1) < 1.0:
            logger.info("✅ Blue Gold Accurate")
        else:
            logger.warning("⚠️ Blue Gold Mismatch")
            
        if abs(red.get('gold_k', 0) - 30.8) < 1.0:
            logger.info("✅ Red Gold Accurate")
        else:
             logger.warning("⚠️ Red Gold Mismatch")
             
        if blue.get('kills') == 1 and red.get('kills') == 3:
             logger.info("✅ Kills Accurate (1-3)")
        else:
             logger.warning(f"⚠️ Kills Mismatch: {blue.get('kills')}-{red.get('kills')}")

    except Exception as e:
        logger.error(f"Validation Error: {e}")

    # --- Phase 3: Reasoning ---
    logger.info("--- Reasoning Engine Prediction ---")
    try:
        try:
            from chronos_v3.intelligence.reasoning import ReasoningEngine
        except ImportError:
            from intelligence.reasoning import ReasoningEngine
            
        reasoner = ReasoningEngine() # Uses default model
        
        prediction = reasoner.predict_winner(state)
        print(json.dumps(prediction, indent=2))
        
        if prediction.get("winner") in ["RED", "BLUE"]:
             logger.info(f"✅ Prediction Successful: {prediction['winner']} ({prediction['confidence']:.2f})")
        else:
             logger.warning("⚠️ Prediction inconclusive")

    except ImportError as e:
        logger.error(f"ReasoningEngine Import Failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        logger.error(f"Reasoning Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_game_state()
