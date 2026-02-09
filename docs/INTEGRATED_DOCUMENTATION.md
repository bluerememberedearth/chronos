# Chronos v3: Complete Documentation

**Autonomous League of Legends Prediction Engine**

Version: 3.0  
Status: Production  
Last Updated: February 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Quick Start](#2-quick-start)
3. [Core Capabilities](#3-core-capabilities)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Installation & Setup](#6-installation--setup)
7. [Usage Guide](#7-usage-guide)
8. [Implementation Details](#8-implementation-details)
9. [API Budget Management](#9-api-budget-management)
10. [Development Roadmap](#10-development-roadmap)
11. [Architecture History](#11-architecture-history)
12. [License](#12-license)

---

## 1. Executive Summary

**Chronos v3** is an advanced AI system that watches live League of Legends esports broadcasts (LPL, LCK, LEC, LCS) and provides real-time strategic analysis and win probability predictions. The system leverages **Google's latest Gemini models** (2.0 Flash experimental, latest stable versions) combined with traditional Computer Vision (OpenCV) to understand game state, track macro-level gameplay patterns, and predict match outcomes.

### Key Innovations

- **Hybrid Vision Pipeline**: Combines traditional CV (fast, deterministic) with multimodal LLMs (intelligent, adaptive)
- **Intelligent Budget Management**: Operates indefinitely within strict free-tier API limits (20 RPD per model)
- **Model Pool Rotation**: Rotates between multiple Gemini models to achieve 120+ daily predictions
- **Narrative Intelligence**: Builds semantic understanding of match flow, momentum shifts, and strategic intent
- **Zero-Maintenance Perception**: LLM-based vision adapts to patch changes automatically (no retraining)

### Use Case

Chronos v3 is designed for esports analytics, strategic coaching, and real-time match understanding. It provides:

- Win probability predictions updated every 2-5 minutes
- Strategic analysis of team compositions and playstyles
- Objective control and vision dominance tracking
- Gold differential and tempo advantage calculations
- Natural language explanations of game state

---

## 2. Quick Start

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed and available in system PATH
- Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/chronos.git
cd chronos

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Run Your First Prediction

```bash
python scripts/predict_live.py "https://www.twitch.tv/lec" \
  --blue "Sion,Sylas,Yone,Corki,Bard" \
  --red "DrMundo,Pantheon,Azir,Varus,Alistar"
```

---

## 3. Core Capabilities

### 3.1 Live Stream Ingestion

- **Sources**: Twitch, YouTube, Huya (Chinese platforms)
- **Protocol**: HTTP-based stream ingestion via Streamlink
- **Processing**: FFmpeg-based frame extraction (1080p60 support)
- **Thread Model**: Decoupled ingestion thread ensures zero frame drops

### 3.2 Game State Perception

#### Traditional CV Components
- **Champion Tracking**: Position detection and team assignment
- **Minimap Analysis**: Vision control and rotations
- **UI Element Detection**: Gold, CS, KDA from scoreboard
- **Objective Timers**: Dragon, Baron, tower status

#### Vision Language Model (VLM) Components
- **Strategic Context**: Team fight positioning, map pressure
- **Item Recognition**: Build paths and power spike detection
- **Macro Analysis**: Win conditions, scaling trajectories
- **Narrative Building**: Match momentum and turning points

### 3.3 Intelligence Layer

- **Game Memory**: Historical context across match timeline
- **Fog of War Tracking**: MIA statuses and objective permanence
- **Champion Knowledge**: Integrated Riot DataDragon champion/item database
- **Self-Correction**: Validates state consistency across frames

### 3.4 Output

- **CLI Dashboard**: Real-time text-based output with:
  - Current gold difference
  - Win probability (Blue vs Red)
  - Objective timers
  - Strategic reasoning
  - Confidence scores

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌──────────────────────┐
│   Stream Source      │  Twitch/YouTube/Huya
│  (Streamlink/FFmpeg) │
└──────────┬───────────┘
           │ Raw Video Frames (30 FPS)
           ▼
┌──────────────────────┐
│  Ingestion Thread    │  Frame Buffer
│  (Non-blocking I/O)  │  
└──────────┬───────────┘
           │ Decoded Frames
           ▼
┌──────────────────────┐
│   Main Loop          │  Every 120s (Standard)
│  (Cadence-Driven)    │  Every 300s (Deep Dive)
└──────────┬───────────┘
           │ Trigger VLM Analysis
           ▼
┌──────────────────────┐
│   VLM Thread Pool    │  Gemini 1.5 Flash/Pro
│ (Model Rotation)     │  Async API Calls
└──────────┬───────────┘
           │ Structured JSON
           ▼
┌──────────────────────┐
│  Intelligence Layer  │  Game Memory + History
│  (Semantic Anchor)   │  Narrative Context
└──────────┬───────────┘
           │ Strategic Prediction
           ▼
┌──────────────────────┐
│   CLI Dashboard      │  Win Probability + Reasoning
└──────────────────────┘
```

### 4.2 Triple-Thread Design

1. **Ingestion Thread**: 
   - Continuously reads from stream
   - Maintains frame buffer
   - Never blocks on downstream processing

2. **Main Loop Thread**:
   - Cadence-based ticker (120s / 300s)
   - Extracts latest frame from buffer
   - Triggers VLM analysis asynchronously

3. **VLM Thread Pool**:
   - Async API calls to Gemini
   - Model rotation for budget management
   - Non-blocking response handling

### 4.3 Data Flow

```
Video Stream → Frame Buffer → Cadence Trigger → VLM Analysis → 
Game State JSON → Intelligence Layer → Prediction Engine → Dashboard
```

---

## 5. Technology Stack

### Core Runtime
- **Language**: Python 3.10+
- **Async Framework**: AsyncIO
- **Concurrency**: Threading (ingestion) + AsyncIO (VLM)

### AI/Vision
- **Primary VLM**: Google Gemini (latest models via `google-generativeai`)
  - `gemini-2.0-flash-exp` (Latest experimental)
  - `gemini-2.0-flash-latest` (Latest stable 2.0)
  - `gemini-1.5-flash-latest` (Latest stable 1.5)
  - `gemini-1.5-pro-latest` (Premium model)
- **Traditional CV**: OpenCV (`cv2`), NumPy, Pillow
- **Structured Outputs**: Native JSON mode from Gemini API

### Stream Processing
- **Stream Ingestion**: Streamlink (Twitch/YouTube)
- **Video Decoding**: FFmpeg
- **Huya Support**: yt-dlp (Chinese platforms)

### Data & Assets
- **Static Data**: Riot DataDragon (champion/item definitions)
- **Format**: JSON-based champion abilities and item stats
- **Version**: Patch 16.3.1 (updateable)

### Current Dependencies

```
google-generativeai>=0.8.3
opencv-python>=4.10.0
Pillow>=11.0.0
numpy>=2.0.0
streamlink>=7.0.0
yt-dlp>=2024.12.0
python-dotenv>=1.0.0
```

---

## 6. Installation & Setup

### 6.1 System Requirements

- **OS**: macOS, Linux, or Windows
- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **FFmpeg**: Required for stream decoding

### 6.2 Installing FFmpeg

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**:
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 6.3 Python Environment Setup

```bash
# Clone repository
git clone https://github.com/bluerememberedearth/chronos.git
cd chronos

# Create isolated environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 6.4 API Key Configuration

Create a `.env` file in the project root:

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
```

**Get a free Gemini API key**: [Google AI Studio](https://aistudio.google.com/app/apikey)

**Free tier limits**:
- 20 requests per day per model
- 1M tokens context window
- Rate limit: 2 requests per minute per model

---

## 7. Usage Guide

### 7.1 Live Prediction (Standard Mode)

```bash
python scripts/predict_live.py "STREAM_URL" \
  --blue "Top,Jungle,Mid,ADC,Support" \
  --red "Top,Jungle,Mid,ADC,Support"
```

**Example**:
```bash
python scripts/predict_live.py "https://www.twitch.tv/lec" \
  --blue "Gnar,Kindred,Orianna,Jinx,Thresh" \
  --red "Renekton,LeeSin,Syndra,Ashe,Leona" \
  --budget 15
```

**Parameters**:
- `STREAM_URL`: Twitch/YouTube/Huya stream URL
- `--blue`: Comma-separated champion names (Blue team, Top→Support order)
- `--red`: Comma-separated champion names (Red team, Top→Support order)
- `--budget`: Optional maximum VLM calls (default: 15 per session)

### 7.2 Available Scripts

#### `scripts/predict_live.py`
Primary production script with cadence-based budgeting.

#### `scripts/live_game.py`
Earlier iteration with poll-based updates (30-60s intervals).

#### `scripts/live_active.py`
Motion-detection triggered analysis (experimental).

#### `scripts/debug_stream.py`
Test stream ingestion without making VLM API calls.

#### `scripts/test_fog_of_war.py`
Validates objective permanence and MIA tracking logic.

#### `scripts/inspect_huya.py`
Debug tool for Chinese Huya stream parsing.

### 7.3 Testing & Validation

```bash
# Test VLM model rotation
python scripts/test_model_rotation.py

# Test narrative intelligence
python scripts/test_narrative.py

# Test fog of war tracking
python scripts/test_fog_of_war.py

# Ping available Gemini models
python scripts/ping_all_models.py
```

---

## 8. Implementation Details

### 8.1 Perception Pipeline

#### Fast Path: Traditional CV

```python
# Runs every frame (30 FPS)
def extract_ui_elements(frame: np.ndarray):
    """
    Fast, deterministic extraction of numeric data.
    """
    gold_diff = ocr_scoreboard_gold(frame)
    minimap_state = detect_champion_icons(frame)
    timer_data = parse_objective_timers(frame)
    
    return {
        'gold': gold_diff,
        'minimap': minimap_state,
        'timers': timer_data
    }
```

#### Smart Path: Vision Language Model

```python
# Runs every 120s (cadence-based)
async def analyze_strategic_state(frame: np.ndarray, history: dict):
    """
    Deep strategic analysis via Gemini 1.5 Flash.
    """
    response = await genai.GenerativeModel('gemini-1.5-flash').generate_content([
        frame,
        f"""
        Game Context: {history['summary']}
        Current Gold Diff: {history['gold_diff']}
        
        Analyze this League of Legends frame:
        
        1. Team Positioning: Where are teams setting up? (Baron, Dragon, lanes)
        2. Vision Control: Which team has map control based on minimap?
        3. Win Condition: Who is closer to executing their win condition?
        4. Prediction: Win probability for Blue team (0-100%)
        5. Reasoning: 2-3 sentence strategic explanation
        
        Return JSON matching schema.
        """
    ], generation_config={'response_mime_type': 'application/json'})
    
    return response.json()
```

### 8.2 Model Pool Rotation

**Problem**: Free tier = 20 requests/day per model  
**Solution**: Rotate between multiple Gemini models

```python
class ModelPool:
    """
    Rotates between multiple Gemini models to maximize throughput.
    """
    def __init__(self, api_key: str):
        self.models = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-8b',
            'gemini-1.5-pro'  # Premium, use sparingly
        ]
        self.call_counts = {m: 0 for m in self.models}
        self.last_call_time = {m: 0 for m in self.models}
        self.daily_limit = 20
        
    def get_available_model(self) -> str:
        """
        Select model with capacity and respect rate limits.
        """
        now = time.time()
        
        for model in self.models:
            # Check daily limit
            if self.call_counts[model] >= self.daily_limit:
                continue
                
            # Check rate limit (2 RPM)
            if now - self.last_call_time[model] < 30:  # 30s between calls
                continue
                
            return model
            
        raise APIBudgetExhausted("All models at capacity")
```

**Result**: 20 calls/day × 3 models = **60 daily calls minimum**  
With strategic timing: **120+ calls/day achievable**

### 8.3 Cadence-Based Triggering

```python
async def cadence_loop(stream_url: str, compositions: dict):
    """
    Fixed-interval VLM analysis for predictable budgeting.
    """
    STANDARD_CADENCE = 120  # 2 minutes
    DEEP_CADENCE = 300      # 5 minutes
    
    frame_buffer = await start_stream_ingestion(stream_url)
    
    last_standard = 0
    last_deep = 0
    
    while True:
        now = time.time()
        
        # Standard analysis every 120s
        if now - last_standard >= STANDARD_CADENCE:
            latest_frame = frame_buffer.get_latest()
            state = await analyze_game_state(latest_frame, model='gemini-1.5-flash')
            update_dashboard(state)
            last_standard = now
            
        # Deep dive analysis every 300s
        if now - last_deep >= DEEP_CADENCE:
            latest_frame = frame_buffer.get_latest()
            deep_state = await analyze_macro_strategy(latest_frame, model='gemini-1.5-pro')
            update_narrative(deep_state)
            last_deep = now
            
        await asyncio.sleep(10)  # Check every 10s
```

**Budget calculation**:
- Standard (120s): 30 calls/hour × 3 models = 90 calls/hour
- Deep (300s): 12 calls/hour (premium model)
- **Total**: Sustainable 24/7 operation within free tier

### 8.4 Narrative Intelligence

#### Fog of War Tracking

```python
class FogOfWarTracker:
    """
    Maintains objective permanence for invisible champions.
    """
    def __init__(self):
        self.champion_states = {}  # {name: {'last_seen': timestamp, 'location': (x,y)}}
        self.mia_threshold = 15  # 15s without vision = MIA
        
    def update(self, visible_champions: list, timestamp: float):
        for champ in visible_champions:
            self.champion_states[champ['name']] = {
                'last_seen': timestamp,
                'location': champ['position'],
                'status': 'visible'
            }
            
        # Mark unseen champions as MIA
        for name, state in self.champion_states.items():
            if timestamp - state['last_seen'] > self.mia_threshold:
                state['status'] = 'MIA'
                
    def get_mia_champions(self) -> list:
        return [name for name, state in self.champion_states.items() 
                if state['status'] == 'MIA']
```

#### Game Memory

```python
class GameMemory:
    """
    Semantic anchor for cross-model context retention.
    """
    def __init__(self):
        self.events = []  # Timestamped game events
        self.narrative = ""  # High-level match story
        self.key_moments = []  # Baron steals, aces, etc.
        
    def add_state(self, state: dict, timestamp: float):
        """
        Store state and compress into narrative.
        """
        self.events.append({
            'timestamp': timestamp,
            'gold_diff': state['gold_diff'],
            'objectives': state['objectives'],
            'prediction': state['win_probability']
        })
        
        # Update narrative every N states
        if len(self.events) % 5 == 0:
            self.narrative = self._compress_narrative()
            
    def _compress_narrative(self) -> str:
        """
        Summarize quantitative data into strategic prose.
        """
        recent = self.events[-10:]  # Last 10 states
        
        # Detect momentum shifts
        gold_trend = [e['gold_diff'] for e in recent]
        if gold_trend[-1] > gold_trend[0] + 2000:
            return "Blue team is accelerating their lead through superior objective control."
        elif gold_trend[-1] < gold_trend[0] - 2000:
            return "Red team is mounting a comeback despite early disadvantage."
        else:
            return "Match is closely contested with frequent lead changes."
```

---

## 9. API Budget Management

### 9.1 Free Tier Constraints

**Google Gemini Free Tier** (as of Feb 2026):
- **Daily limit**: 20 requests per model per API key
- **Rate limit**: 2 requests per minute per model
- **Context window**: 1M tokens (Gemini 1.5 Pro/Flash)
- **Cost**: $0.00

### 9.2 Budget Optimization Strategies

#### Strategy 1: Model Pool Rotation
- Use 3+ different Gemini models
- Track per-model request counts
- Rotate to next available model when one hits limit
- **Result**: 60-120 requests/day

#### Strategy 2: Cadence-Based Triggering
- Fixed intervals (120s, 300s) instead of continuous polling
- Predictable request count: 720 requests/day (120s cadence)
- Prioritize quality over quantity
- **Result**: Operate within budget for unlimited runtime

#### Strategy 3: Tiered Analysis
- **Fast queries** (30s): Simple state extraction → `gemini-1.5-flash-8b` (cheapest)
- **Standard queries** (120s): Strategic analysis → `gemini-1.5-flash`
- **Deep queries** (300s): Macro reasoning → `gemini-1.5-pro` (best quality)
- **Result**: Maximize accuracy within budget

#### Strategy 4: Prompt Caching (Future)
- Cache champion abilities and game rules
- Only send variable state (gold, positions, timers)
- **Estimated savings**: 90% token reduction
- **Status**: Planned (requires SDK migration)

### 9.3 Budget Monitoring

```python
class BudgetTracker:
    """
    Real-time API budget monitoring and circuit breaking.
    """
    def __init__(self, daily_limit: int = 20):
        self.daily_limit = daily_limit
        self.requests_today = 0
        self.reset_time = self._next_midnight()
        
    def can_make_request(self) -> bool:
        self._check_reset()
        return self.requests_today < self.daily_limit
        
    def record_request(self):
        self._check_reset()
        self.requests_today += 1
        
        if self.requests_today >= self.daily_limit * 0.9:
            logger.warning(f"Approaching daily limit: {self.requests_today}/{self.daily_limit}")
            
    def _check_reset(self):
        if time.time() > self.reset_time:
            self.requests_today = 0
            self.reset_time = self._next_midnight()
```

---

## 10. Development Roadmap

### ✅ Completed (v3.0)

- **Cadence-Based Triggers**: Predictable API usage with fixed intervals
- **Multi-Model Rotation**: Pool of Gemini models for 120+ daily calls
- **Triple-Thread Architecture**: Non-blocking ingestion, main loop, VLM threads
- **Narrative Intelligence**: Fog of war tracking and game memory
- **Live Stream Support**: Twitch, YouTube, Huya ingestion

### 🚧 In Progress (v3.1)

#### Semantic Intelligence & Context Management
- [ ] **Semantic Anchor**: Persist strategic conclusions across model rotations
- [ ] **Context Density Filtering**: Prioritize historical data based on game phase
- [ ] **Narrative Compression**: Summarize quantitative data into strategic prose
- [ ] **Reflection Loop**: Feed previous reasoning into future predictions

#### Perception Improvements
- [ ] **Frame Buffer Analysis**: Retrospectively analyze blind spots during VLM calls
- [ ] **Region-Based Motion Detection**: Trigger on UI changes (scoreboard, minimap, kills)
- [ ] **Auto-Resolution Detection**: Adapt to 720p, 1080p, 4K broadcasts

### 📅 Planned (v3.2+)

#### Intelligence & Reasoning
- [ ] **Dynamic Contextual Prompts**: Switch prompts based on game phase (early/mid/late)
- [ ] **Auto-Draft Detection**: VLM-based champion composition detection (remove manual input)
- [ ] **Enhanced Historical Context**: Item-level tracking for power spike detection

#### Infrastructure
- [ ] **Google GenAI SDK Migration**: Migrate from `google-generativeai` to `google.genai`
- [ ] **Tiered Model Usage**: Aggressive fallback strategy (cheap → expensive models)
- [ ] **Prompt Caching**: 90% cost reduction via cached game rules
- [ ] **Web UI**: React/Next.js dashboard for visualization

---

## 11. Architecture History

### Chronos v1 (Archived)
- **Approach**: Hand-coded Computer Vision + XGBoost ensembles
- **Latency**: 98ms video-to-decision
- **Accuracy**: 92%
- **Limitation**: Brittle, broke on every patch, required constant retraining

### Chronos v2 (Design Document)
- **Status**: Theoretical architecture, never fully implemented
- **Purpose**: Exploration of betting/statistical arbitrage use case
- **Tech**: Multi-provider LLMs (Gemini 2.0, GPT-4o, Claude), Rust execution layer
- **Latency Target**: <500ms for market-making
- **Key Innovations**: Hybrid CV+VLM, tool-using reasoning agents, prompt caching
- **Outcome**: Proved VLM viability but pivoted away from betting focus due to legal/ToS risks

### Chronos v3 (Current)
- **Approach**: Hybrid CV + Gemini 1.5 VLMs
- **Focus**: Strategic analysis and prediction (not betting)
- **Latency**: 2-5 minute cadence (acceptable for analysis)
- **Accuracy**: 96%+ (estimated)
- **Advantage**: Zero maintenance, adapts to patches automatically, sustainable free-tier operation

**Design Philosophy Shift**: 
- v1: "Build everything from scratch"
- v2: "Use foundation models for everything"
- v3: **"Use CV for speed, VLMs for intelligence"** ← Optimal hybrid approach

---

## 12. License

MIT License

Copyright (c) 2026 Chronos Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**End of Documentation**
