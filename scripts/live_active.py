#!/usr/bin/env python3
"""
Live Game Prediction - Active Perception + Multi-Model

Brain-inspired architecture:
1. Peripheral (CV): Runs every frame, detects changes (~5ms, $0)
2. Foveal (VLM): Only triggered when CV detects something (~2s, consumes quota)
3. Multi-Model: Rotates between Gemini models to maximize 20 RPD/model limit

This conserves API quota by only calling VLM when the game state actually changes.
"""

import logging
import sys
import os
import time
import asyncio
import subprocess
import shutil

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
from dotenv import load_dotenv

from ingest.active_perception import ActivePerceptionEngine, PeripheralVision
from ingest.gemini_perception import GeminiPerceptionEngine
from intelligence.reasoning import ReasoningEngine
from intelligence.history import GameStateHistory

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ActiveLive")


def get_yt_dlp_path():
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    potential = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    return potential if os.path.exists(potential) else None


def resolve_stream_url(huya_url: str) -> str:
    yt_dlp = get_yt_dlp_path()
    if not yt_dlp:
        raise RuntimeError("yt-dlp not found")
    
    result = subprocess.run([yt_dlp, '-g', huya_url], capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        return result.stdout.strip().split('\n')[0]
    raise RuntimeError(f"yt-dlp failed: {result.stderr}")


class StreamCapture:
    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
    
    def get_frame(self):
        for _ in range(3):
            self.cap.grab()
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        if self.cap:
            self.cap.release()


class ActivePredictor:
    """Prediction with VLM-call awareness."""
    
    def __init__(self):
        self.observations = []
        self.locked = None
        self.vlm_calls = 0
    
    def add(self, game_state: dict, prediction: dict) -> dict:
        self.observations.append({
            'time': time.time(),
            'game_time': game_state.get('game_time'),
            'pred': prediction
        })
        self.vlm_calls += 1
        
        winner = prediction.get('winner')
        if len(self.observations) >= 2:
            recent = self.observations[-5:]
            agrees = sum(1 for o in recent if o['pred'].get('winner') == winner)
            consistency = agrees / len(recent)
        else:
            consistency = 0.5
        
        base = prediction.get('confidence', 0.5)
        cumulative = base * 0.6 + consistency * 0.4
        prediction['cumulative_confidence'] = min(cumulative, 0.95)
        prediction['vlm_calls'] = self.vlm_calls
        
        if cumulative >= 0.80 and not self.locked:
            self.locked = {
                'winner': winner,
                'confidence': cumulative,
                'game_time': game_state.get('game_time'),
                'vlm_calls': self.vlm_calls
            }
            logger.info(f"🔒 PREDICTION LOCKED after {self.vlm_calls} VLM calls")
        
        return prediction


async def run_active_pipeline(stream_url: str, seek_seconds: int = 0):
    """
    Main loop with Active Perception.
    """
    load_dotenv()
    
    logger.info("=" * 60)
    logger.info("CHRONOS - ACTIVE PERCEPTION MODE")
    logger.info("CV triggers VLM only on state changes")
    logger.info("=" * 60)
    
    direct_url = resolve_stream_url(stream_url)
    stream = StreamCapture(direct_url)
    
    # Seek if requested (for testing)
    if seek_seconds > 0:
        stream.cap.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000)
        logger.info(f"Seeked to {seek_seconds}s")
    
    # Initialize components
    vlm = GeminiPerceptionEngine()
    active = ActivePerceptionEngine(vlm_engine=vlm)
    reasoning = ReasoningEngine()
    history = GameStateHistory()
    predictor = ActivePredictor()
    
    frame_count = 0
    start_time = time.time()
    game_detected = False
    
    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                await asyncio.sleep(1)
                continue
            
            frame_count += 1
            
            # Active Perception: CV-triggered VLM
            result = active.process_frame(frame)
            
            if not result.get('vlm_called'):
                # No VLM call - show CV status occasionally
                if frame_count % 100 == 0:
                    alerts = result.get('alerts', [])
                    logger.info(f"Frame {frame_count}: CV monitoring... ({len(alerts)} alerts)")
                await asyncio.sleep(0.03)  # ~30fps CV loop
                continue
            
            # VLM was called - process result
            logger.info(f"Frame {frame_count}: VLM triggered by: {result.get('trigger_context', 'unknown')}")
            
            if result.get('error'):
                logger.error(f"VLM error: {result['error']}")
                await asyncio.sleep(1)
                continue
            
            state = result.get('broadcast_state', 'OTHER')
            
            if not result.get('is_live_game'):
                logger.info(f"State: {state}")
                continue
            
            if not game_detected:
                game_detected = True
                logger.info("🎮 GAME DETECTED - Beginning predictions")
            
            game_state = result.get('game_state', {})
            if not game_state:
                continue
            
            # Add to history and predict
            history.add_state(game_state, time.time())
            history_summary = history.get_summary(300)
            
            prediction = reasoning.predict_winner(game_state, history_summary)
            prediction = predictor.add(game_state, prediction)
            
            # Display
            left = game_state.get('left_team', {})
            right = game_state.get('right_team', {})
            
            elapsed = time.time() - start_time
            vlm_efficiency = frame_count / max(predictor.vlm_calls, 1)
            
            print(f"\n{'='*60}")
            print(f"⏱️  Game: {game_state.get('game_time', '?')} | Elapsed: {elapsed:.0f}s")
            print(f"💰 Gold: L {left.get('gold_k', '?')}k vs R {right.get('gold_k', '?')}k")
            print(f"⚔️  Kills: L {left.get('kills', '?')} vs R {right.get('kills', '?')}")
            print(f"🔮 Prediction: {prediction.get('winner')} ({prediction.get('cumulative_confidence', 0):.0%})")
            print(f"📊 VLM Calls: {predictor.vlm_calls} | Frames: {frame_count} | Efficiency: {vlm_efficiency:.0f}x")
            
            if predictor.locked:
                print(f"\n🔒 LOCKED: {predictor.locked['winner']} wins!")
                print(f"   Only {predictor.locked['vlm_calls']} VLM calls needed!")
                break
            
            print(f"{'='*60}")
            
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        stream.release()
        
        print("\n" + "=" * 60)
        print("FINAL STATUS")
        print(f"Frames processed: {frame_count}")
        print(f"VLM calls: {predictor.vlm_calls}")
        print(f"Efficiency: {frame_count / max(predictor.vlm_calls, 1):.0f}x frames per VLM call")
        if predictor.locked:
            print(f"Winner: {predictor.locked['winner']}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?",
                        default="https://www.huya.com/video/play/1093845088.html")
    parser.add_argument("--seek", type=int, default=600,
                        help="Seek to N seconds (default: 600 = 10 min)")
    
    args = parser.parse_args()
    asyncio.run(run_active_pipeline(args.url, args.seek))
