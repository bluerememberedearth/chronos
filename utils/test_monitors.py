import cv2
import time
import logging
from chronos_v3.ingest.ingest_stream import VideoIngest
from chronos_v3.monitors.minimap import MinimapMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Setup Ingestion (VOD)
    # Using the test vod we downloaded earlier
    source = "data/test_vod.mp4" 
    ingest = VideoIngest(source=source, target_fps=30)
    
    # 2. Setup Monitor
    monitor = MinimapMonitor()
    monitor.start()
    
    logger.info("Starting Monitor Test...")
    
    try:
        for timestamp, frame in ingest.stream():
            # Feed frame to monitor
            monitor.update(frame)
            
            # --- Visualization for Debugging ---
            # 1. Draw ROI on main frame
            x, y, w, h = monitor.roi_x, monitor.roi_y, monitor.roi_w, monitor.roi_h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 2. Show Output using cv2.imshow (if local) 
            # or just log alerts
            
            # Since we are likely headless or remote, let's just log high activity frames
            # But for visual verification (if user looks at artifacts), we could save snapshots.
            
            # Access monitor internals for debug
            alerts = monitor.get_alerts()
            if alerts:
                logger.info(f"ALERTS: {alerts}")
            
            # Optional: throttling to watch it unfold
            # time.sleep(0.01) 

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()
