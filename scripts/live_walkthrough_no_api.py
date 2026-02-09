#!/usr/bin/env python3
"""
Live Walkthrough (No API) - Safe Verification Script

This script mirrors predict_live.py but mocks VLM calls to allow 
a full infrastructure walkthrough without spending API credits.
"""

import logging
import sys
import os
import time
import asyncio
import argparse
import textwrap
from collections import deque
import numpy as np

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Components (Real logic)
from ingest.stream_loader import GenericStreamLoader
from intelligence.game_memory import GameMemory, TeamComp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("ChronosSim")

class MockAsyncVLM:
    """Mocks the AsyncVLMWrapper to avoid API calls."""
    def __init__(self):
        self._pending_result = None
        self._last_call_time = 0
        self.total_calls = 0
        self.busy = False

    def is_busy(self):
        return self.busy

    def submit(self, frame, blue, red, context):
        self.busy = True
        self.total_calls += 1
        self._last_call_time = time.time()
        # Simulate 3s VLM processing time
        logger.info(f"🧪 [MOCK VLM] Analyzing frame... (Target: {blue} vs {red})")
        
        # Create a fake result to be returned in 3 seconds
        self._pending_result = {
            'is_live_game': True,
            'broadcast_state': 'LIVE',
            'game_state': {
                'game_time': f"{10 + self.total_calls}:24",
                'left_team': {'gold_k': 20.5 + self.total_calls, 'kills': self.total_calls},
                'right_team': {'gold_k': 21.2, 'kills': 2}
            },
            'prediction': {
                'winner': 'RED' if self.total_calls < 5 else 'BLUE',
                'confidence': 0.65 + (self.total_calls * 0.05),
                'reasoning': "Mocked reasoning for walkthrough purposes."
            }
        }

    def check_result(self):
        if self.busy and (time.time() - self._last_call_time > 3):
            result = self._pending_result
            self._pending_result = None
            self.busy = False
            return result
        return None

class MotionDetector:
    def __init__(self, threshold: float = 0.04):
        self.prev_frame = None
        self.threshold = threshold
        self.history = deque(maxlen=30)
    
    def check(self, frame: np.ndarray) -> bool:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        if self.prev_frame is None:
            self.prev_frame = gray
            return True
        diff = cv2.absdiff(self.prev_frame, gray)
        motion = np.mean(diff) / 255.0
        self.history.append(motion)
        self.prev_frame = gray
        avg = np.mean(list(self.history)) if len(self.history) > 5 else motion
        return motion > avg * 2.0 and motion > self.threshold

async def run_walkthrough(url, blue_comp, red_comp):
    logger.info(f"🚀 Starting Live Walkthrough (NO-API MODE)")
    logger.info(f"Target: {url}")
    
    stream = GenericStreamLoader(url=url)
    try:
        stream.start()
        logger.info("📡 Ingestion Pipe established.")
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        return

    # Real logic components (except VLM)
    motion = MotionDetector()
    async_vlm = MockAsyncVLM()
    memory = GameMemory(blue_comp, red_comp)
    last_vlm_time = 0
    start_time = time.time()
    
    print("\n" + "="*60)
    print("🕵️  CHRONOS LIVE WALKTHROUGH (SIMULATED COGNITION)")
    print("="*60)
    print(f"Ingesting: {url}")
    print("Note: VLM calls are intercepted and mocked.")
    print("="*60 + "\n")

    try:
        # Run for 2 minutes to demonstrate stability
        while time.time() - start_time < 120:
            frame_data = stream.get_frame()
            if frame_data is None:
                await asyncio.sleep(1)
                continue
            
            frame, count = frame_data
            now = time.time()
            
            # 1. Motion Detection (Real)
            motion_triggered = motion.check(frame)
            
            # 2. Result Processing (Real flow, Mock data)
            result = async_vlm.check_result()
            if result:
                last_vlm_time = now
                logger.info(f"✅ Received VLM analysis (Simulated)")
                
                game_state = result.get('game_state', {})
                prediction = result.get('prediction', {})
                
                # Update memory
                memory.add_observation(game_state, prediction, now)
                
                # Print tactical update
                left = game_state.get('left_team', {})
                right = game_state.get('right_team', {})
                print(f"⏱️  {game_state['game_time']} | 💰 {left['gold_k']:.1f}k vs {right['gold_k']:.1f}k | ⚔️ {left['kills']}-{right['kills']} | 🔮 {prediction['winner']} {prediction['confidence']:.0%}")
                
                # Reasoning Display
                reasoning = prediction.get('reasoning', "Simulated reasoning for walkthrough.")
                wrapped = textwrap.fill(reasoning, width=80, initial_indent="   📝 ", subsequent_indent="      ")
                print(f"{wrapped}\n")

            # 3. Submission Logic (Real triggers)
            elapsed = now - last_vlm_time
            should_call = (not async_vlm.is_busy() and elapsed >= 20 and (motion_triggered or elapsed >= 45))
            
            if should_call:
                async_vlm.submit(frame, blue_comp, red_comp, "Mock Context")
                logger.info(f"📡 VLM Analysis Triggered (Motion: {motion_triggered})")

            await asyncio.sleep(0.1)
            
        logger.info("🏁 Walkthrough duration reached.")
        
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        logger.info("🛑 Walkthrough complete. Infrastructure holds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", default="https://www.twitch.tv/lcs", nargs="?")
    args = parser.parse_args()
    
    blue = TeamComp.from_string("Kante,Viego,Azir,Ashe,Braum")
    red = TeamComp.from_string("Renekton,Sejuani,Taliyah,Varus,Nautilus")
    
    asyncio.run(run_walkthrough(args.url, blue, red))
