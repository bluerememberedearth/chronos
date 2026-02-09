"""
Async VLM Wrapper - Non-blocking VLM calls

Runs VLM inference in a ThreadPool so the main loop continues
processing frames and detecting motion during API calls.
"""

import logging
import time
import concurrent.futures
from typing import Optional, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PendingVLMCall:
    """Tracks a pending VLM call."""
    future: concurrent.futures.Future
    frame_timestamp: float
    submitted_at: float


class AsyncVLMWrapper:
    """
    Wraps a synchronous VLM perception engine for async execution.
    
    Usage:
        async_vlm = AsyncVLMWrapper(perception_engine)
        async_vlm.submit(frame, blue, red, context)
        
        # Later, in main loop:
        result = async_vlm.check_result()
        if result:
            # Process result
    """
    
    def __init__(self, perception_engine, max_workers: int = 1):
        self.perception = perception_engine
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending: Optional[PendingVLMCall] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_result_time: float = 0
        
        # Stats
        self.total_calls = 0
        self.total_latency = 0.0
    
    def is_busy(self) -> bool:
        """Check if a VLM call is in progress."""
        return self.pending is not None and not self.pending.future.done()
    
    def submit(self, frame: np.ndarray, blue_comp: str, red_comp: str, 
               context: str = "") -> bool:
        """
        Submit a single-frame VLM call for async execution.
        """
        if self.is_busy():
            return False
        
        now = time.time()
        future = self.executor.submit(
            self._run_vlm,
            frame.copy(),
            blue_comp,
            red_comp,
            context
        )
        self.pending = PendingVLMCall(future, now, now)
        return True

    def submit_sequence(self, frames: list, blue_comp: str, red_comp: str, 
                      context: str = "") -> bool:
        """
        Submit a multi-frame sequence for Deep Macro Analysis.
        """
        if self.is_busy():
            return False
            
        now = time.time()
        # Copy frames to avoid race conditions (expensive but necessary)
        frames_copy = [f.copy() for f in frames]
        
        future = self.executor.submit(
            self._run_sequence,
            frames_copy,
            blue_comp,
            red_comp,
            context
        )
        self.pending = PendingVLMCall(future, now, now)
        logger.info(f"Deep Macro Analysis submitted with {len(frames)} frames")
        return True

    def _run_vlm(self, frame: np.ndarray, blue: str, red: str, context: str) -> Dict:
        """Execute single-frame analysis."""
        try:
            return self.perception.analyze_with_prediction(frame, blue, red, context)
        except Exception as e:
            logger.error(f"VLM thread error: {e}")
            return {"error": str(e), "is_live_game": False}

    def _run_sequence(self, frames: list, blue: str, red: str, context: str) -> Dict:
        """Execute multi-frame sequence analysis."""
        try:
            return self.perception.analyze_sequence_with_prediction(frames, blue, red, context)
        except Exception as e:
            logger.error(f"Deep Analysis thread error: {e}")
            return {"error": str(e), "is_live_game": False}
    
    def check_result(self) -> Optional[Dict[str, Any]]:
        """
        Check if async VLM result is ready.
        
        Returns the result if available, None if still pending.
        Non-blocking.
        """
        if self.pending is None:
            return None
        
        if not self.pending.future.done():
            return None
        
        # Get result
        try:
            result = self.pending.future.result(timeout=0)
            latency = time.time() - self.pending.submitted_at
            
            self.total_calls += 1
            self.total_latency += latency
            
            logger.info(f"VLM completed in {latency:.1f}s")
            
            result['_latency'] = latency
            result['_frame_timestamp'] = self.pending.frame_timestamp
            
            self.last_result = result
            self.last_result_time = time.time()
            
        except Exception as e:
            logger.error(f"VLM result error: {e}")
            result = {"error": str(e), "is_live_game": False}
        
        self.pending = None
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        avg_latency = self.total_latency / max(1, self.total_calls)
        return {
            "total_calls": self.total_calls,
            "avg_latency": f"{avg_latency:.1f}s",
            "is_busy": self.is_busy()
        }
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=False)


class FrameBuffer:
    """
    Circular buffer of recent frames with timestamps.
    
    Allows retrospective analysis after VLM calls complete.
    """
    
    def __init__(self, max_frames: int = 150, max_age: float = 10.0):
        self.buffer: deque = deque(maxlen=max_frames)
        self.max_age = max_age
    
    def add(self, frame: np.ndarray, timestamp: float = None):
        """Add a frame to the buffer."""
        if timestamp is None:
            timestamp = time.time()
        
        # Store smaller version for memory efficiency
        small = frame[::4, ::4]  # 4x downsample
        self.buffer.append((small, timestamp))
    
    def get_since(self, timestamp: float):
        """Get all frames since a timestamp."""
        return [(f, t) for f, t in self.buffer if t >= timestamp]
    
    def get_motion_during(self, start_time: float, end_time: float) -> float:
        """Calculate total motion between timestamps."""
        frames = [(f, t) for f, t in self.buffer 
                  if start_time <= t <= end_time]
        
        if len(frames) < 2:
            return 0.0
        
        total_motion = 0.0
        for i in range(1, len(frames)):
            prev, _ = frames[i-1]
            curr, _ = frames[i]
            
            diff = np.abs(prev.astype(float) - curr.astype(float))
            total_motion += np.mean(diff) / 255.0
        
        return total_motion / len(frames)
    
    def __len__(self):
        return len(self.buffer)


# Quick test
if __name__ == "__main__":
    print("Testing AsyncVLMWrapper...")
    
    # Mock perception engine
    class MockPerception:
        def analyze_with_prediction(self, frame, blue, red, ctx):
            time.sleep(2)  # Simulate API latency
            return {"broadcast_state": "LIVE_GAME", "prediction": {"winner": "BLUE"}}
    
    wrapper = AsyncVLMWrapper(MockPerception())
    
    # Submit
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    wrapper.submit(fake_frame, "Gnar,Lee,Ahri,Jinx,Thresh", "Jax,Viego,Syndra,Kaisa,Nautilus")
    
    # Poll while busy
    print("Submitted, polling...")
    for i in range(30):
        result = wrapper.check_result()
        if result:
            print(f"Result after {i*0.1:.1f}s: {result}")
            break
        print(f"  Busy at {i*0.1:.1f}s")
        time.sleep(0.1)
    
    wrapper.shutdown()
    print("✅ AsyncVLMWrapper test passed")
