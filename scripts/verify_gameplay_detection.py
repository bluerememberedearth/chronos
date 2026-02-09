import cv2
import os
import sys
import logging
from dotenv import load_dotenv

# Ensure project root is in path
# The script is in /Users/solana/chronos_v3/scripts
# We need to add /Users/solana/ so that 'import chronos_v3' works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from chronos_v3.monitors.broadcast import BroadcastStateMonitor
from chronos_v3.ingest.perception import HybridPerceptionEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("verification_results.txt", mode='w')
    ]
)
logger = logging.getLogger("Verification")

def verify_detection():
    load_dotenv()
    
    # Initialize Engine
    logger.info("Initializing HybridPerceptionEngine...")
    engine = HybridPerceptionEngine()
    if not engine.client:
        logger.error("SiliconFlow Client not initialized. Check .env")
        return

    # Initialize Monitor
    monitor = BroadcastStateMonitor(perception_engine=engine)
    
    # Calculate absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # chronos_v3 dir
    
    test_cases = [
        {"path": os.path.join(base_dir, "tests/fixtures/huya_vod/frame_000.jpg"), "expected": ["DRAFT_PHASE", "CHAMPION_SELECT"]}, 
        {"path": os.path.join(base_dir, "tests/fixtures/huya_vod/frame_1200.jpg"), "expected": ["LIVE_GAME"]},
    ]
    
    for case in test_cases:
        path = case["path"]
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        frame = cv2.imread(path)
        if frame is None:
            logger.warning(f"Failed to read image: {path}")
            continue
            
        logger.info(f"Testing {path}...")
        
        # Manually invoke the cognitive check (bypass threading/interval)
        # We need to hook into the logic. 
        # BroadcastStateMonitor stores state in self.state
        
        # Reset state for clean test
        monitor.state = "UNKNOWN" 
        monitor.consecutive_live_counter = 0 if "LIVE_GAME" in case["expected"] else 100 # Force reset logic
        
        # The monitor logic requires 3 consecutive "LIVE_GAME" to switch TO Live Game.
        # But only 1 non-Live to switch FROM Live Game.
        # To test single-frame classification accuracy, we should patch the method or just look at logs/internal state logic.
        # Let's override `REQUIRED_STREAK` for this test.
        monitor.REQUIRED_STREAK = 1
        
        monitor._run_cognitive_check(frame)
        
        result_state = monitor.state
        logger.info(f"Result: {result_state} | Expected: {case['expected']}")
        
        if result_state in case["expected"]:
            logger.info("✅ PASS")
        else:
            logger.error("❌ FAIL")

if __name__ == "__main__":
    verify_detection()
