
import logging
import sys
import os
import time
import json
import cv2
from dotenv import load_dotenv

# Path setup for correct imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try importing components
    try:
    # Try importing components
        from chronos_v3.ingest.ingest_stream import VideoIngest
        from chronos_v3.monitors.broadcast import BroadcastStateMonitor
        from chronos_v3.ingest.perception import HybridPerceptionEngine
        from chronos_v3.intelligence.reasoning import ReasoningEngine
        from chronos_v3.intelligence.history import GameStateHistory
    except ImportError:
    # Use fallback direct imports
        sys.path.insert(0, os.path.dirname(project_root)) # Point to parent for package
        from chronos_v3.ingest.ingest_stream import VideoIngest
        from chronos_v3.monitors.broadcast import BroadcastStateMonitor
        from chronos_v3.ingest.perception import HybridPerceptionEngine
        from chronos_v3.intelligence.reasoning import ReasoningEngine
        from chronos_v3.intelligence.history import GameStateHistory

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("SimulateLiveGame")

def resolve_huya_url(url):
    import shutil
    import subprocess
    yt_dlp_cmd = "yt-dlp"
    if not shutil.which(yt_dlp_cmd):
         # Try local venv
         potential = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
         if os.path.exists(potential):
             yt_dlp_cmd = potential
         else:
             return None
             
    cmd = [yt_dlp_cmd, '-g', url]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return res.stdout.strip().split('\n')[0]
    except:
        pass
    return None

def simulate_game(vod_url):
    load_dotenv()
    logger.info(f"Starting Simulation on VOD: {vod_url}")
    
    # Resolve URL
    stream_url = resolve_huya_url(vod_url)
    if not stream_url:
        logger.error("Failed to resolve stream URL")
        return

    # Initialize Components
    perception_engine = HybridPerceptionEngine()
    reasoning_engine = ReasoningEngine()
    history = GameStateHistory()
    
    cap = cv2.VideoCapture(stream_url)
    
    # Test Points: 15min, 16min (to show flow), 20min, 25min, 30min
    # Adding consecutive points to test history
    timestamps = [900, 960, 1200, 1500, 1800]
    
    for ts in timestamps:
        logger.info(f"Seeking to timestamp: {ts}s ...")
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        
        # Read a few frames to settle buffer
        for _ in range(10):
            cap.grab()
            
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            continue
            
        logger.info(f"Analyzing Frame at {ts}s...")
        
        # 1. Extract State
        state = perception_engine.analyze_game_state(frame)
        
        if "error" in state:
            logger.warning(f"Extraction Error: {state['error']}")
            continue
            
        # 2. Update History
        history.add_state(state, ts)
        history_summary = history.get_summary(seconds_lookback=300)
            
        # 3. Predict Winner
        prediction = reasoning_engine.predict_winner(state, history_summary)
        pred_winner = prediction.get("next_fight_winner", "UNCERTAIN")
        
        # 4. Verify Outcome (Peek ahead 2 minutes)
        logger.info(f"Peeking ahead to {ts + 120}s to verify prediction...")
        cap.set(cv2.CAP_PROP_POS_MSEC, (ts + 120) * 1000)
        for _ in range(5): cap.grab() # Settle
        ret_future, frame_future = cap.read()
        
        outcome_log = "Could not verify."
        if ret_future:
            future_state = perception_engine.analyze_game_state(frame_future)
            if "error" not in future_state:
                 # Determine who won the interval
                 # Logic: Who got more kills? Who gained more gold relative to the other?
                 
                 # Helper
                 def get_team_stats(s, side): return s.get(side) or {}
                 
                 # Current
                 curr_l = get_team_stats(state, 'left_team')
                 curr_r = get_team_stats(state, 'right_team')
                 
                 # Future
                 fut_l = get_team_stats(future_state, 'left_team')
                 fut_r = get_team_stats(future_state, 'right_team')
                 
                 # Deltas
                 l_kills = fut_l.get('kills', 0) - curr_l.get('kills', 0)
                 r_kills = fut_r.get('kills', 0) - curr_r.get('kills', 0)
                 
                 l_gold = fut_l.get('gold_k', 0) - curr_l.get('gold_k', 0)
                 r_gold = fut_r.get('gold_k', 0) - curr_r.get('gold_k', 0)
                 
                 actual_winner = "NONE"
                 if l_kills > r_kills + 1: actual_winner = "LEFT" # Significant kill lead
                 elif r_kills > l_kills + 1: actual_winner = "RIGHT"
                 elif l_gold > r_gold + 1.0: actual_winner = "LEFT" # Significant gold swing
                 elif r_gold > l_gold + 1.0: actual_winner = "RIGHT"
                 elif l_kills > r_kills: actual_winner = "LEFT" # Slight edge
                 elif r_kills > l_kills: actual_winner = "RIGHT"
                 
                 # Accuracy
                 is_correct = (pred_winner == "BLUE" and actual_winner == "LEFT") or \
                              (pred_winner == "RED" and actual_winner == "RIGHT") or \
                              (pred_winner == "NONE" and actual_winner == "NONE")
                              
                 match_icon = "✅" if is_correct else "❌"
                 outcome_log = f"{match_icon} Actual: {actual_winner} (Kills: L+{l_kills}/R+{r_kills}, Gold: L+{l_gold:.1f}k/R+{r_gold:.1f}k)"

        # 5. Output
        blue_stats = state.get('left_team') or {}
        red_stats = state.get('right_team') or {}
        
        print(f"\n==========================================")
        print(f"⏱️ Time: {state.get('game_time')} (+2min verification)")
        print(f"💰 Gold: L {blue_stats.get('gold_k')}k - R {red_stats.get('gold_k')}k")
        print(f"🟢 Ults: L {blue_stats.get('ult_count')} - R {red_stats.get('ult_count')}")
        print(f"🐉 Next Obj: {state.get('next_objective', {}).get('type')} ({state.get('next_objective', {}).get('timer')})")
        print(f"🗺️  Minimap: {json.dumps(state.get('minimap', {}))}")
        print(f"🔮 Prediction: {pred_winner} ({prediction.get('winner')} Game)")
        print(f"🕵️ Verification: {outcome_log}")
        print(f"📝 Reasoning: {prediction.get('reasoning')}")
        print(f"==========================================\n")
        
    cap.release()

if __name__ == "__main__":
    vod_url = "https://www.huya.com/video/play/1093845088.html"
    simulate_game(vod_url)
