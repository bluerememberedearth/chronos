import time
import logging
import sys
import os

# Core Pipeline
from chronos_v3.ingest.ingest_stream import VideoIngest
from chronos_v3.ingest.perception import HybridPerceptionEngine
from chronos_v3.reasoning.context import ContextWindow

# Monitors
from chronos_v3.monitors.minimap import MinimapMonitor
from chronos_v3.monitors.scoreboard import ScoreboardMonitor
from chronos_v3.monitors.champion_state import ChampionStateMonitor
from chronos_v3.monitors.objectives import ObjectiveMonitor
from chronos_v3.monitors.broadcast import BroadcastStateMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChronosWatcher")

def main():
    logger.info("Starting Chronos v3 Watcher...")
    
    # 1. Configuration
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    target_fps = 30
    
    logger.info(f"Source: {source} | Target FPS: {target_fps}")
    ingest = VideoIngest(source=source, target_fps=target_fps)
    
    # 2. Initialize Perception Cockpit (The "Eyes")
    logger.info("Initializing Perception Monitors...")
    
    hybrid_engine = HybridPerceptionEngine() # The Smart/Fast VLM engine
    
    monitors = {
        "minimap": MinimapMonitor(),
        "scoreboard": ScoreboardMonitor(),
        "champions": ChampionStateMonitor(),
        "objectives": ObjectiveMonitor(),
        "broadcast": BroadcastStateMonitor(perception_engine=hybrid_engine), # Integrated VLM
        "vision": hybrid_engine 
    }
    
    # Start Threaded Monitors
    for name, monitor in monitors.items():
        if hasattr(monitor, 'start'):
            monitor.start()
            
    # 3. Initialize Context (The "Memory")
    context = ContextWindow(max_history_seconds=600)
    
    logger.info("Starting Main Loop. Press Ctrl+C to stop.")
    try:
        start_time = time.time()
        for timestamp, frame in ingest.stream():
            
            # --- A. Perception Update ---
            # 1. Feed frame to threaded monitors (Non-blocking)
            for name, monitor in monitors.items():
                if name != "vision": # Vision engine is manual for now
                    monitor.update(frame)
            
            # 2. Check Broadcast State (Gatekeeper)
            # If we are in Replay or Ad, we do NOT want to update Context or run expensive VLM.
            broadcast_state = monitors['broadcast'].get_state()
            
            if broadcast_state != "LIVE_GAME":
                # Only log occasionally to avoid spam
                if timestamp % 5.0 < 0.1:
                    m, s = divmod(int(timestamp), 60)
                    time_str = f"{m:02d}:{s:02d}"
                    logger.info(f"VOD Time [{time_str}] Broadcast State: {broadcast_state}. STANDBY.")
                # We optionally still update ingest to keep pipe clear? Yes, ingest is driving loop.
                continue

            # --- B. Data Aggregation ---
            # Collect current state from all sensors
            
            # 3. Run Vision Engine (Fast Path - YOLO)
            # We run this synchronously in the loop as it drives the 'frame tick' logic often.
            # OPTIMIZATION: Skip YOLO if not in LIVE_GAME to fast-forward pre-game.
            vision_result = {}
            if broadcast_state == "LIVE_GAME":
                 vision_result = monitors['vision'].process_frame(frame, timestamp)
            
            snapshot = {
                "timestamp": timestamp,
                "broadcast": broadcast_state,
                "minimap": monitors['minimap'].get_alerts(),
                "scoreboard": monitors['scoreboard'].get_state(),
                "champions": monitors['champions'].get_states(),
                "objectives": monitors['objectives'].get_state(),
                "vision_fast": vision_result.get('fast'),
                "vision_smart": vision_result.get('smart') # Only populated if VLM ran
            }
            
            # --- C. Context Update ---
            context.update(snapshot)
            
            # --- D. Logging / Debugging ---
            if timestamp % 1.0 < 0.05: # Timestamp is now Video Time in seconds
                summary = context.get_context_summary(window_seconds=10)
                # Parse summary to concise log
                log_line = summary.split('\n')[1] # Time/Gold line
                
                m, s = divmod(int(timestamp), 60)
                time_str = f"{m:02d}:{s:02d}"
                
                logger.info(f"VOD Time [{time_str}]: {log_line}")
                
            # Demo Limit / Transition Check
            # Only stop if we are way past the expected start time (e.g. 10 mins)
            if timestamp > 600: 
                logger.info("Passed 10 minute mark. Stopping.")
                break
            
            # Hard Timeout (30 minutes to catch the 8-minute mark)
            if time.time() - start_time > 1800:
                logger.info("Time limit reached (1800s).")
                break
                
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.exception(f"Crash: {e}")
    finally:
        logger.info("Shutting down monitors...")
        for name, monitor in monitors.items():
             if hasattr(monitor, 'stop'):
                 monitor.stop()
        ingest.stop()

if __name__ == "__main__":
    main()
