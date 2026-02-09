import logging
import time
import sys
import os
import cv2

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingest.stream_loader import GenericStreamLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("TwitchVerify")

def verify_twitch(url, num_frames=10):
    logger.info(f"🚀 Starting verification for: {url}")
    loader = GenericStreamLoader(url=url)
    
    try:
        loader.start()
        logger.info("📡 Stream Loader started, waiting for first frame...")
        
        frames_captured = 0
        timeout = 60  # 1 minute timeout
        start_time = time.time()
        
        while frames_captured < num_frames:
            if time.time() - start_time > timeout:
                logger.error("❌ Timeout waiting for frames.")
                break
                
            frame_data = loader.get_frame()
            if frame_data:
                frame, count = frame_data
                frames_captured += 1
                logger.info(f"✅ Captured frame {frames_captured}/{num_frames} (Internal count: {count})")
                
                # Verify frame shape
                h, w, c = frame.shape
                if h == 1080 and w == 1920:
                    logger.info(f"📏 Frame resolution correct: {w}x{h}")
                else:
                    logger.warning(f"⚠️ Unexpected resolution: {w}x{h}")
            else:
                time.sleep(1)
        
        if frames_captured >= num_frames:
            logger.info("✨ SUCCESS: Twitch ingestion is working perfectly!")
        else:
            logger.error("Failed to capture required number of frames.")
            
    except Exception as e:
        logger.error(f"💥 Verification failed: {e}")
    finally:
        loader.close()
        logger.info("🛑 Loader closed.")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.twitch.tv/lcs"
    verify_twitch(url)
