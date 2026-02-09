#!/usr/bin/env python3
"""
Live Game Prediction - Gemini Rate-Limit Optimized

This script connects to a Huya stream and predicts the game winner
using Gemini with severe rate limiting in mind.

Design:
- ONE VLM call every 60 seconds (combines state detection + extraction)
- Builds prediction confidence over time
- Locks prediction when confidence > 80%
"""

import logging
import sys
import os
import time
import json
import asyncio
import subprocess
import shutil

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
parent_root = os.path.abspath(os.path.join(project_root, '..'))
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

import cv2
from dotenv import load_dotenv

try:
    from ingest.gemini_perception import GeminiPerceptionEngine
    from intelligence.reasoning import ReasoningEngine
    from intelligence.history import GameStateHistory
except ImportError:
    from chronos_v3.ingest.gemini_perception import GeminiPerceptionEngine
    from chronos_v3.intelligence.reasoning import ReasoningEngine
    from chronos_v3.intelligence.history import GameStateHistory

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("LiveGameGemini")


def get_yt_dlp_path():
    """Find yt-dlp executable."""
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    potential = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    if os.path.exists(potential):
        return potential
    return None


def resolve_stream_url(huya_url: str) -> str:
    """Resolve Huya URL to direct stream URL."""
    yt_dlp = get_yt_dlp_path()
    if not yt_dlp:
        raise RuntimeError("yt-dlp not found")
    
    logger.info(f"Resolving: {huya_url}")
    result = subprocess.run(
        [yt_dlp, '-g', huya_url],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode == 0:
        url = result.stdout.strip().split('\n')[0]
        logger.info(f"Resolved: {url[:60]}...")
        return url
    raise RuntimeError(f"yt-dlp failed: {result.stderr}")


class StreamCapture:
    """Simple stream frame capture."""
    
    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
        logger.info("Stream connected")
    
    def get_frame(self):
        # Skip buffered frames to get latest
        for _ in range(5):
            self.cap.grab()
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        if self.cap:
            self.cap.release()


class SimplePredictor:
    """Simple prediction accumulator."""
    
    def __init__(self):
        self.observations = []
        self.locked = None
    
    def add(self, game_state: dict, prediction: dict) -> dict:
        """Add observation and calculate cumulative confidence."""
        self.observations.append({
            'time': time.time(),
            'game_time': game_state.get('game_time'),
            'state': game_state,
            'pred': prediction
        })
        
        # Calculate consistency
        winner = prediction.get('winner')
        if len(self.observations) >= 3:
            recent = self.observations[-5:]
            agrees = sum(1 for o in recent if o['pred'].get('winner') == winner)
            consistency = agrees / len(recent)
        else:
            consistency = 0.5
        
        # Cumulative confidence
        base = prediction.get('confidence', 0.5)
        cumulative = base * 0.6 + consistency * 0.4
        
        prediction['cumulative_confidence'] = min(cumulative, 0.95)
        prediction['observations'] = len(self.observations)
        
        # Lock at 80%
        if cumulative >= 0.80 and not self.locked:
            self.locked = {
                'winner': winner,
                'confidence': cumulative,
                'game_time': game_state.get('game_time'),
                'observations': len(self.observations)
            }
            logger.info(f"🔒 PREDICTION LOCKED: {winner} @ {game_state.get('game_time')}")
        
        return prediction


async def run_pipeline(stream_url: str, poll_interval: int = 60):
    """
    Main prediction loop.
    
    Args:
        stream_url: Huya URL (VOD or live)
        poll_interval: Seconds between VLM calls (minimum 30)
    """
    load_dotenv()
    
    # Resolve URL
    direct_url = resolve_stream_url(stream_url)
    
    # Initialize
    stream = StreamCapture(direct_url)
    perception = GeminiPerceptionEngine()
    perception.set_min_interval(poll_interval)
    reasoning = ReasoningEngine()
    history = GameStateHistory()
    predictor = SimplePredictor()
    
    logger.info("=" * 50)
    logger.info("CHRONOS - GEMINI RATE-LIMITED MODE")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info("=" * 50)
    
    game_started = False
    
    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                logger.warning("No frame, retrying...")
                await asyncio.sleep(5)
                continue
            
            # Combined VLM call (state + extraction)
            result = perception.analyze_frame(frame)
            
            if "error" in result:
                logger.error(f"VLM error: {result['error']}")
                await asyncio.sleep(poll_interval)
                continue
            
            broadcast_state = result.get("broadcast_state", "OTHER")
            
            # Wait for game to start
            if not result.get("is_live_game"):
                logger.info(f"State: {broadcast_state} (waiting for LIVE_GAME)")
                await asyncio.sleep(poll_interval)
                continue
            
            if not game_started:
                game_started = True
                logger.info("🎮 GAME DETECTED - Starting predictions")
            
            # Extract game state
            game_state = result.get("game_state", {})
            if not game_state:
                logger.warning("No game state extracted")
                await asyncio.sleep(poll_interval)
                continue
            
            # Add to history
            history.add_state(game_state, time.time())
            history_summary = history.get_summary(300)
            
            # Get prediction
            prediction = reasoning.predict_winner(game_state, history_summary)
            prediction = predictor.add(game_state, prediction)
            
            # Display
            left = game_state.get('left_team', {})
            right = game_state.get('right_team', {})
            
            print(f"\n{'='*50}")
            print(f"⏱️  {game_state.get('game_time', '?')}")
            print(f"💰 Gold: L {left.get('gold_k', '?')}k vs R {right.get('gold_k', '?')}k")
            print(f"⚔️  Kills: L {left.get('kills', '?')} vs R {right.get('kills', '?')}")
            print(f"🗺️  {game_state.get('minimap_summary', 'N/A')}")
            print(f"🔮 Prediction: {prediction.get('winner')} ({prediction.get('cumulative_confidence', 0):.0%})")
            print(f"📊 Observations: {prediction.get('observations', 0)}")
            
            if predictor.locked:
                print(f"\n🔒 LOCKED: {predictor.locked['winner']} wins!")
                print(f"   Confidence: {predictor.locked['confidence']:.0%}")
                print(f"   Locked at: {predictor.locked['game_time']}")
                break
            
            print(f"{'='*50}")
            
            await asyncio.sleep(poll_interval)
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        stream.release()
        
        print("\n" + "=" * 50)
        print("FINAL STATUS")
        print(f"Observations: {len(predictor.observations)}")
        if predictor.locked:
            print(f"Winner: {predictor.locked['winner']}")
        print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?",
                        default="https://www.huya.com/video/play/1093845088.html",
                        help="Huya URL")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between predictions (default: 60)")
    
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.url, args.interval))
