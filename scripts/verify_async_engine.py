
import logging
import sys
import os
import time
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from chronos_v3.ingest.perception import HybridPerceptionEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("AsyncVerify")

def verify_async():
    load_dotenv()
    engine = HybridPerceptionEngine()
    
    # Mock frame (black)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    logger.info("Starting Async Verification Loop...")
    
    # Simulate a game loop
    start_time = time.time()
    for i in range(100):
        loop_start = time.time()
        
        # This call should be INSTANT (<1ms) mostly, except when triggering thread (should still be fast)
        result = engine.process_frame(frame, time.time())
        
        duration = (time.time() - loop_start) * 1000
        
        has_smart = result.get("smart") is not None
        
        # Log slow frames
        if duration > 10: 
            logger.warning(f"Frame {i}: Slow processing ({duration:.2f}ms)")
        
        if has_smart:
            logger.info(f"Frame {i}: 💡 Smart Result Available! Content: {result['smart'].get('raw_response')[:50]}...")
            
        # Simulate 60 FPS
        time.sleep(0.016)
        
    logger.info(f"Finished 100 frames in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    verify_async()
