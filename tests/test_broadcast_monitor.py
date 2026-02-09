import pytest
import cv2
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chronos_v3.monitors.broadcast import BroadcastStateMonitor
from chronos_v3.ingest.perception import HybridPerceptionEngine

# Mock data
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=HybridPerceptionEngine)
    # Mock client and response structure
    engine.client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"state": "LIVE_GAME"}'
    mock_response.choices = [mock_choice]
    engine.client.chat.completions.create.return_value = mock_response
    engine.model = "mock-model"
    return engine

@pytest.fixture
def black_frame():
    path = os.path.join(FIXTURE_DIR, 'test_black.jpg')
    if not os.path.exists(path):
        # Create on fly if missing (fallback)
        import numpy as np
        img = np.zeros((1080,1920,3), np.uint8)
        cv2.imwrite(path, img)
    return cv2.imread(path)

def test_broadcast_monitor_initial_state(mock_engine):
    monitor = BroadcastStateMonitor(perception_engine=mock_engine)
    assert monitor.state == "UNKNOWN"

def test_broadcast_monitor_trigger(mock_engine, black_frame):
    monitor = BroadcastStateMonitor(perception_engine=mock_engine)
    
    # We need to trigger the update enough times to hit the frame interval
    # Interval is 60 frames
    monitor.frame_interval = 2 # Lower for test
    
    # Run update
    monitor.update(black_frame)
    monitor.update(black_frame) 
    
    # Since checking is threaded, we might not see immediate state change in unit test
    # unless we mock the thread or sleep. 
    # Better: Invoke _run_cognitive_check directly for unit testing logic
    
    monitor._run_cognitive_check(black_frame)
    
    # Default mock returns "LIVE_GAME"
    # But streak requirement is 3
    assert monitor.consecutive_live_counter == 1
    assert monitor.state == "UNKNOWN" 
    
    monitor._run_cognitive_check(black_frame)
    assert monitor.consecutive_live_counter == 2
    
    monitor._run_cognitive_check(black_frame)
    assert monitor.consecutive_live_counter == 3
    assert monitor.state == "LIVE_GAME"

def test_broadcast_monitor_fast_fail(mock_engine, black_frame):
    """
    Test that sends a different response to verify logic handles specific VLM output
    """
    monitor = BroadcastStateMonitor(perception_engine=mock_engine)
    
    # Mock "DRAFT_PHASE" response
    mock_engine.client.chat.completions.create.return_value.choices[0].message.content = '{"state": "DRAFT_PHASE"}'
    
    monitor._run_cognitive_check(black_frame)
    
    assert monitor.state == "DRAFT_PHASE"
    assert monitor.consecutive_live_counter == 0
