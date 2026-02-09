#!/usr/bin/env python3
"""
Live Game Prediction Pipeline
Connects to a Huya livestream and predicts game winner in real-time.
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

# Also add parent for package imports
parent_root = os.path.abspath(os.path.join(project_root, '..'))
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

import cv2
from dotenv import load_dotenv

# Try both import styles
try:
    from chronos_v3.ingest.perception import HybridPerceptionEngine
    from chronos_v3.intelligence.reasoning import ReasoningEngine
    from chronos_v3.intelligence.history import GameStateHistory
    from chronos_v3.monitors.broadcast import BroadcastStateMonitor
except ImportError:
    from ingest.perception import HybridPerceptionEngine
    from intelligence.reasoning import ReasoningEngine
    from intelligence.history import GameStateHistory
    from monitors.broadcast import BroadcastStateMonitor

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("LiveGame")


def get_yt_dlp_path():
    """Find yt-dlp executable."""
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    potential = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    if os.path.exists(potential):
        return potential
    return None


def resolve_stream_url(huya_url: str) -> str:
    """Resolve Huya URL to direct stream URL using yt-dlp."""
    yt_dlp = get_yt_dlp_path()
    if not yt_dlp:
        raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")
    
    logger.info(f"Resolving stream URL: {huya_url}")
    try:
        result = subprocess.run(
            [yt_dlp, '-g', huya_url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            urls = result.stdout.strip().split('\n')
            stream_url = urls[0]
            logger.info(f"Resolved to: {stream_url[:80]}...")
            return stream_url
        else:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timed out")


class LiveStreamCapture:
    """Captures frames from a live stream."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap = None
        self._connect()
    
    def _connect(self):
        """Open video capture."""
        logger.info("Connecting to stream...")
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
        logger.info("Stream connected!")
    
    def get_frame(self):
        """Get the latest frame from the stream."""
        if not self.cap or not self.cap.isOpened():
            self._connect()
        
        # Skip ahead to get most recent frame (live streams buffer)
        for _ in range(5):
            self.cap.grab()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame, reconnecting...")
            self._connect()
            ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def release(self):
        """Release the capture."""
        if self.cap:
            self.cap.release()


class GamePredictor:
    """Accumulates observations and builds prediction confidence over time."""
    
    def __init__(self):
        self.observations = []  # (timestamp, state, prediction)
        self.locked_prediction = None
        self.reasoning_engine = ReasoningEngine()
        self.history = GameStateHistory()
    
    def add_observation(self, state: dict, game_time_str: str) -> dict:
        """Add a game state observation and update prediction."""
        timestamp = time.time()
        
        # Add to history
        self.history.add_state(state, timestamp)
        history_summary = self.history.get_summary(seconds_lookback=300)
        
        # Get prediction from reasoning engine
        prediction = self.reasoning_engine.predict_winner(state, history_summary)
        
        # Store observation
        self.observations.append({
            'timestamp': timestamp,
            'game_time': game_time_str,
            'state': state,
            'prediction': prediction
        })
        
        # Calculate cumulative confidence
        confidence = self._calculate_cumulative_confidence(prediction)
        prediction['cumulative_confidence'] = confidence
        
        # Lock prediction if confident enough
        if confidence >= 0.80 and not self.locked_prediction:
            self.locked_prediction = {
                'winner': prediction.get('winner'),
                'confidence': confidence,
                'game_time': game_time_str,
                'observations': len(self.observations)
            }
            logger.info(f"🔒 PREDICTION LOCKED: {prediction.get('winner')} @ {game_time_str}")
        
        return prediction
    
    def _calculate_cumulative_confidence(self, current_pred: dict) -> float:
        """
        Calculate confidence based on:
        - Consistency of predictions over time
        - Size of gold lead
        - Number of observations
        """
        if len(self.observations) < 2:
            return current_pred.get('confidence', 0.5) * 0.5  # Low confidence early
        
        # Count how many recent predictions agree
        recent = self.observations[-5:] if len(self.observations) >= 5 else self.observations
        current_winner = current_pred.get('winner')
        
        agreeing = sum(1 for obs in recent if obs['prediction'].get('winner') == current_winner)
        consistency = agreeing / len(recent)
        
        # Base confidence from LLM
        base_confidence = current_pred.get('confidence', 0.5)
        
        # Boost for consistency
        cumulative = base_confidence * 0.6 + consistency * 0.4
        
        # Cap at 0.95
        return min(cumulative, 0.95)
    
    def get_status(self) -> dict:
        """Get current prediction status."""
        if not self.observations:
            return {'status': 'waiting', 'observations': 0}
        
        latest = self.observations[-1]['prediction']
        return {
            'status': 'locked' if self.locked_prediction else 'predicting',
            'winner': latest.get('winner'),
            'confidence': latest.get('cumulative_confidence', 0),
            'observations': len(self.observations),
            'locked': self.locked_prediction
        }


async def run_live_pipeline(stream_url: str, poll_interval: int = 30):
    """
    Main live game prediction loop.
    
    Args:
        stream_url: Huya stream URL
        poll_interval: Seconds between predictions (default 30)
    """
    load_dotenv()
    
    # Resolve stream URL
    direct_url = resolve_stream_url(stream_url)
    
    # Initialize components
    stream = LiveStreamCapture(direct_url)
    perception = HybridPerceptionEngine()
    state_monitor = BroadcastStateMonitor(perception_engine=perception)
    predictor = GamePredictor()
    
    logger.info("=" * 60)
    logger.info("CHRONOS LIVE PREDICTION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Stream: {stream_url}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info("Waiting for game to start...")
    logger.info("=" * 60)
    
    game_started = False
    start_time = None
    frame_count = 0
    
    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                logger.warning("No frame received, waiting...")
                await asyncio.sleep(5)
                continue
            
            frame_count += 1
            
            # Phase 2: Detect game state
            if not game_started:
                # Update monitor (triggers VLM check every 60 frames internally)
                state_monitor.update(frame)
                state_name = state_monitor.get_state()
                
                if state_name == 'LIVE_GAME':
                    game_started = True
                    start_time = time.time()
                    logger.info("🎮 GAME STARTED! Beginning prediction...")
                else:
                    if frame_count % 30 == 0:  # Log every ~1 second
                        logger.info(f"Current state: {state_name} (waiting for LIVE_GAME)")
                    await asyncio.sleep(0.5)
                    continue
            
            # Phase 3: Extract state and predict
            game_state = perception.analyze_game_state(frame)
            
            if "error" in game_state:
                logger.warning(f"Extraction error: {game_state['error']}")
                await asyncio.sleep(poll_interval)
                continue
            
            game_time = game_state.get('game_time', 'Unknown')
            prediction = predictor.add_observation(game_state, game_time)
            
            # Display status
            left_stats = game_state.get('left_team', {})
            right_stats = game_state.get('right_team', {})
            
            print(f"\n{'='*50}")
            print(f"⏱️  Game Time: {game_time}")
            print(f"💰 Gold: L {left_stats.get('gold_k', '?')}k - R {right_stats.get('gold_k', '?')}k")
            print(f"⚔️  Kills: L {left_stats.get('kills', '?')} - R {right_stats.get('kills', '?')}")
            print(f"🔮 Prediction: {prediction.get('winner')} ({prediction.get('cumulative_confidence', 0):.0%})")
            print(f"📝 {prediction.get('reasoning', '')[:100]}...")
            
            if predictor.locked_prediction:
                locked = predictor.locked_prediction
                print(f"\n🔒 LOCKED PREDICTION: {locked['winner']} wins!")
                print(f"   Locked at: {locked['game_time']} with {locked['confidence']:.0%} confidence")
                print(f"   Based on: {locked['observations']} observations")
                print(f"{'='*50}\n")
                break
            
            print(f"{'='*50}\n")
            
            await asyncio.sleep(poll_interval)
    
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        stream.release()
        
        # Final summary
        status = predictor.get_status()
        print("\n" + "=" * 60)
        print("FINAL STATUS")
        print("=" * 60)
        print(f"Observations: {status['observations']}")
        if status.get('locked'):
            print(f"Prediction: {status['locked']['winner']} WINS")
            print(f"Confidence: {status['locked']['confidence']:.0%}")
        elif status['observations'] > 0:
            print(f"Current lean: {status.get('winner')} ({status.get('confidence', 0):.0%})")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chronos Live Game Prediction")
    parser.add_argument("url", nargs="?", 
                        default="https://www.huya.com/660168",
                        help="Huya stream URL (live or VOD)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between predictions (default: 30)")
    
    args = parser.parse_args()
    
    asyncio.run(run_live_pipeline(args.url, args.interval))
