#!/usr/bin/env python3
"""
Chronos Live Predictor - Production Version (Cadence Mode)

Usage:
  python predict_live.py URL --blue "Gnar,LeeSin,Ahri,Jinx,Thresh" --red "Pax,Viego,Syndra,Kaisa,Nautilus"

Features:
- Cadence-Based Triggering (120s snapshots, 300s deep dives)
- Triple-Thread Architecture (Zero-blocking ingestion)
- Budget-aware (15 calls per game)
- Multi-Model Rotation (handle 21 RPD limit)
"""

import logging
import sys
import os
import time
import json
import asyncio
import subprocess
import shutil
import argparse
import textwrap
from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from dotenv import load_dotenv

# Components
from ingest.gemini_perception import GeminiPerceptionEngine
from ingest.stream_loader import GenericStreamLoader
from ingest.async_vlm import AsyncVLMWrapper, FrameBuffer
from intelligence.game_memory import GameMemory, TeamComp

# Logging - clean output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("Chronos")


class LivePredictor:
    """Prediction state machine."""
    
    def __init__(self, blue_comp: TeamComp, red_comp: TeamComp):
        self.blue_comp = blue_comp
        self.red_comp = red_comp
        self.observations = []
        self.locked = None
        self.vlm_calls = 0
    
    def add_observation(self, game_state: dict, prediction: dict) -> dict:
        """Add observation and calculate confidence."""
        self.vlm_calls += 1
        
        self.observations.append({
            'time': time.time(),
            'game_time': game_state.get('game_time'),
            'winner': prediction.get('winner'),
            'confidence': prediction.get('confidence', 0.5)
        })
        
        # Calculate cumulative confidence from consistency
        winner = prediction.get('winner')
        recent = self.observations[-5:]
        agrees = sum(1 for o in recent if o['winner'] == winner)
        consistency = agrees / len(recent) if recent else 0.5
        
        base_conf = prediction.get('confidence', 0.5)
        cumulative = base_conf * 0.5 + consistency * 0.5
        
        prediction['cumulative'] = min(cumulative, 0.95)
        prediction['observations'] = len(self.observations)
        prediction['consistency'] = consistency
        
        # Lock at 85%
        if cumulative >= 0.85 and not self.locked:
            self.locked = {
                'winner': winner,
                'confidence': cumulative,
                'game_time': game_state.get('game_time'),
                'calls': self.vlm_calls
            }
        elif self.locked:
             # Unlock if confidence drops or winner flips (Game is not over!)
             if winner != self.locked['winner'] or cumulative < 0.70:
                 self.locked = None
        
        return prediction


def resolve_stream(url: str) -> str:
    """Resolve stream URL via yt-dlp."""
    yt_dlp = shutil.which("yt-dlp") or ".venv/bin/yt-dlp"
    result = subprocess.run([yt_dlp, '-g', url], capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        return result.stdout.strip().split('\n')[0]
    raise RuntimeError(f"Failed to resolve stream: {result.stderr}")


async def run_prediction(
    stream_url: str,
    blue_comp: TeamComp,
    red_comp: TeamComp,
    target_calls: int = 15
):
    """
    Main prediction loop.
    Cadence: Standard call every 120s, Deep Dive every 300s.
    """
    load_dotenv()
    
    logger.info(f"Starting stream...")
    stream = GenericStreamLoader(url=stream_url)
    try:
        stream.start()
        logger.info("Stream started!")
    except Exception as e:
        logger.error(f"Stream failed: {e}")
        raise
    
    perception = GeminiPerceptionEngine()
    perception.min_call_interval = 30
    async_vlm = AsyncVLMWrapper(perception)
    frame_buffer = FrameBuffer(max_frames=150)
    memory = GameMemory(blue_comp, red_comp)
    predictor = LivePredictor(blue_comp, red_comp)
    
    print("\n" + "="*60)
    print("🎮 CHRONOS LIVE PREDICTOR (CADENCE MODE)")
    print("="*60)
    print(f"BLUE: {blue_comp}")
    print(f"RED:  {red_comp}")
    print(f"Budget: {target_calls} Standard Calls")
    print("Intervals: Standard (120s), Deep Macro (300s)")
    print("="*60 + "\n")
    
    game_started = False
    last_vlm_time = 0
    frame_count = 0
    long_term_frames = []
    last_deep_dive_time = 0
    
    try:
        while True:
            frame_data = stream.get_frame()
            if frame_data is None:
                await asyncio.sleep(2)
                continue
            
            frame, read_count = frame_data
            now = time.time()
            frame_buffer.add(frame, now)
            frame_count += 1
            
            # Deep Macro Analysis Trigger (Every 5 minutes)
            if game_started and (now - last_deep_dive_time >= 300) and not async_vlm.is_busy():
                if long_term_frames:
                    frames_to_send = [f for t, f in long_term_frames]
                    if async_vlm.submit_sequence(
                        frames_to_send,
                        str(blue_comp), 
                        str(red_comp),
                        memory.get_full_context()
                    ):
                        logger.info(f"🔭 Triggering Deep Macro Analysis with {len(frames_to_send)} frames")
                        last_deep_dive_time = now
            
            # Check for completed VLM result
            result = async_vlm.check_result()
            if result:
                if 'deep_analysis' not in result:
                    last_vlm_time = now
                
                if result.get('error'):
                    logger.error(f"VLM Error: {result['error']}")
                elif 'deep_analysis' in result:
                    analysis = result['deep_analysis']
                    pred = result.get('prediction', {})
                    logger.info("🔭 DEEP MACRO ANALYSIS RECEIVED!")
                    print("\n" + "="*60)
                    print(f"🔭 DEEP MACRO ANALYSIS ({len(long_term_frames)} frames analyzed)")
                    print("="*60)
                    print(f"📈 TREND: {analysis.get('trend_summary', 'N/A')}")
                    print(f"🌊 MOMENTUM: {analysis.get('momentum', 'UNKNOWN')} | KEY: {analysis.get('key_factor', 'N/A')}")
                    
                    lanes = analysis.get('lane_pressure', {})
                    print(f"🗺️  LANES: TOP:{lanes.get('top','?')} MID:{lanes.get('mid','?')} BOT:{lanes.get('bot','?')}")
                    
                    winlines = analysis.get('win_conditions', [])
                    if winlines:
                        print("🏆 WIN CONDITIONS:")
                        for w in winlines: print(f"   - {w}")
                            
                    print(f"🔮 PREDICTION: {pred.get('winner', '?')} ({pred.get('confidence', 0):.1%})")
                    print(f"📝 REASONING:\n{textwrap.fill(pred.get('reasoning', ''), width=80, initial_indent='   ', subsequent_indent='   ')}")
                    print("="*60 + "\n")
                    
                    if result.get('game_state'):
                         memory.add_observation(result['game_state'], pred, now)
                else:
                    if not result.get('is_live_game'):
                        logger.info(f"Waiting for game... ({result.get('broadcast_state', 'UNKNOWN')})")
                    else:
                        if not game_started:
                            game_started = True
                            logger.info("🎮 GAME STARTED!")
                            last_deep_dive_time = now
                        
                        if not long_term_frames or (now - long_term_frames[-1][0] >= 30):
                            frame_small = cv2.resize(frame, (640, 360))
                            long_term_frames.append((now, frame_small))

                        game_state = result.get('game_state', {})
                        if game_state:
                            prediction = result.get('prediction', {})
                            if memory.add_observation(game_state, prediction, now):
                                prediction = predictor.add_observation(game_state, prediction)
                                context_str = ""
                                if predictor.locked:
                                    context_str = f" | 🔒 LOCKED"
                                
                                left, right = game_state.get('left_team', {}), game_state.get('right_team', {})
                                winner, conf = prediction.get('winner', '?'), prediction.get('cumulative', 0)
                                
                                print(f"⏱️  {game_state.get('game_time', '??:??'):>6} | "
                                      f"💰 {left.get('gold_k', 0):.1f}k vs {right.get('gold_k', 0):.1f}k | "
                                      f"⚔️ {left.get('kills', 0)}-{right.get('kills', 0)} | "
                                      f"🔮 {winner} {conf:.0%}{context_str} | "
                                      f"📊 Call {predictor.vlm_calls}/{target_calls}")
                                
                                reasoning = prediction.get('reasoning', result.get('reasoning', ''))
                                if reasoning:
                                    wrapped = textwrap.fill(reasoning, width=80, initial_indent="   📝 ", subsequent_indent="      ")
                                    print(f"{wrapped}\n")
                            
                            # (Removed block display)
                            
                            if predictor.vlm_calls >= target_calls:
                                logger.warning("Standard VLM budget reached. Switching to purely Deep Macro Analysis mode.")
            
            # Submit new VLM call on 120s cadence
            elapsed_since_vlm = now - last_vlm_time
            if not async_vlm.is_busy() and predictor.vlm_calls < target_calls and elapsed_since_vlm >= 120:
                if async_vlm.submit(frame, str(blue_comp), str(red_comp), memory.get_full_context() if game_started else ""):
                    logger.info(f"VLM Cadence Call {predictor.vlm_calls + 1}/{target_calls} submitted")
            
            await asyncio.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if game_started and long_term_frames and not async_vlm.is_busy():
            logger.info("🎬 Stream ended. Triggering final Endgame Deep Analysis...")
            async_vlm.submit_sequence([f for t,f in long_term_frames], str(blue_comp), str(red_comp), memory.get_full_context())
            time.sleep(5)
        stream.close()
        print("\n" + "="*60 + "\nFINAL SUMMARY\nVLM calls: " + str(predictor.vlm_calls) + "\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Chronos Live Predictor")
    parser.add_argument("url", help="Stream URL")
    parser.add_argument("--blue", required=True, help="Blue champions: 'T,J,M,A,S'")
    parser.add_argument("--red", required=True, help="Red champions: 'T,J,M,A,S'")
    parser.add_argument("--budget", type=int, default=15, help="Max VLM calls (default: 15)")
    args = parser.parse_args()
    
    try:
        blue, red = TeamComp.from_string(args.blue), TeamComp.from_string(args.red)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    asyncio.run(run_prediction(args.url, blue, red, args.budget))

if __name__ == "__main__":
    main()
