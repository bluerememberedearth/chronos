# Chronos v3 Development Roadmap 🚀

This document outlines planned improvements and future features for the Chronos Live Prediction Pipeline.

**Last Updated**: February 2026  
**Current Version**: v3.0  
**Next Release**: v3.1 (Semantic Intelligence Focus)

---

## Release History

### ✅ v3.0 (Current - Released)

**Core Features**:
- ✅ Cadence-based VLM triggers (120s standard, 300s deep)
- ✅ Multi-model rotation (latest Gemini models with `-latest` suffix for future-proofing)
- ✅ Triple-thread architecture (ingestion, main loop, VLM pool)
- ✅ Narrative intelligence with fog of war tracking
- ✅ Live stream support (Twitch, YouTube, Huya)
- ✅ CLI dashboard with real-time predictions
- ✅ Free-tier budget management (120+ daily calls)

**Technology Stack**:
- Python 3.10+ with AsyncIO
- Google Gemini (latest models: 2.0 Flash, experimental variants, 1.5 Pro)
- OpenCV for traditional CV
- Streamlink + FFmpeg for ingestion

---

## 🚧 v3.1: Semantic Intelligence & Context Management

**Epic Theme**: Shifting from "data recording" to "semantic anchor" that manages context density for the VLM brain.

**Release Target**: Q2 2026

### Semantic Anchor System
- [ ] **Cross-Model Grounding**: Ensure high-level conclusions (e.g., "Jinx is the primary win-con") persist across physical model rotations so the "thought process" doesn't reset when switching between gemini-1.5-flash and gemini-1.5-pro
- [ ] **State Persistence Layer**: Maintain strategic narrative across API calls
- [ ] **Confidence Tracking**: Track prediction confidence over time to detect momentum shifts

**Technical Approach**:
```python
class SemanticAnchor:
    """
    Maintains strategic narrative across model rotations.
    """
    def __init__(self):
        self.core_insights = {}  # Persisted strategic conclusions
        self.confidence_history = []  # Track prediction stability
        
    def anchor_insight(self, key: str, value: str, confidence: float):
        """
        Store high-confidence strategic conclusions that survive model rotations.
        """
        if confidence > 0.8:
            self.core_insights[key] = {
                'value': value,
                'confidence': confidence,
                'timestamp': time.time()
            }
```

### Context Density Filtering
- [ ] **Dynamic Prioritization**: Select which historical data to pass based on current game events
  - During resets: Prioritize item timings and gold trajectories
  - During objective setup: Prioritize vision control and positioning
  - During team fights: Prioritize ultimate cooldowns and health states
- [ ] **Token Budget Optimization**: Compress context to fit within input limits while preserving critical information
- [ ] **Event Salience Scoring**: Rank historical events by strategic importance

**Example**:
```python
def filter_context_for_phase(history: list, current_phase: str) -> str:
    """
    Dynamically prioritize context based on game phase.
    """
    if current_phase == 'baron_setup':
        return focus_on(['vision_control', 'positioning', 'ultimate_cds'], history)
    elif current_phase == 'item_spike':
        return focus_on(['gold_diff', 'item_timings', 'power_spikes'], history)
```

### Narrative Compression
- [ ] **Impact Summarization**: Describe the *impact* of events rather than raw numbers
  - ❌ "Gold diff changed from +1200 to +2400"
  - ✅ "Blue's Baron take created a 5k gold swing, opening path to victory"
- [ ] **Strategic Prose Generation**: Convert quantitative data into high-density narrative paragraphs
- [ ] **Momentum Detection**: Automatically identify and articulate momentum shifts

### Reflection Loop
- [ ] **Prediction Validation**: Compare previous predictions to actual outcomes
- [ ] **Confidence Calibration**: Adjust future confidence based on historical accuracy
- [ ] **Error Analysis**: Feed prediction "misses" back into context
  - "Previously predicted Blue 70% at 15min, but Red won teamfight. Revising scaling assessment."

---

## 📅 v3.2: Advanced Perception

**Release Target**: Q3 2026

### Frame Buffer Analysis
- [ ] **Blind Spot Detection**: Retrospectively analyze the 2-5 second window during VLM calls
  - Problem: While waiting for API response, we might miss kills/objectives
  - Solution: Buffer frames during VLM call, analyze retroactively for major events
- [ ] **Event Detection**: Identify sudden kills, objective steals, aces in buffered frames
- [ ] **State Reconciliation**: Update game state if buffered analysis reveals missed events

**Architecture**:
```python
class BlindSpotAnalyzer:
    """
    Analyzes frames buffered during VLM API calls.
    """
    async def analyze_blind_spot(self, buffered_frames: list):
        # Quick CV-based event detection on buffered frames
        for frame in buffered_frames:
            if detect_major_event(frame):  # Kill feed, objective announcement
                await trigger_emergency_vlm_update(frame)
```

### Region-Based Motion Detection
- [ ] **UI Region Monitoring**: Track specific UI areas instead of full-frame pixel diffs
  - **Scoreboard**: Trigger on gold changes >500, KDA updates
  - **Minimap**: Trigger on team rotations, objective contests
  - **Kill Feed**: Trigger immediately on multi-kills or objective steals
- [ ] **Intelligent Triggering**: VLM calls only on relevant game state changes
- [ ] **Budget Efficiency**: Reduce wasted calls on static game states

**Regions of Interest**:
```python
ROI_DEFINITIONS = {
    'scoreboard': (0, 0, 1920, 120),      # Top of screen
    'minimap': (1620, 780, 1920, 1080),   # Bottom right
    'kill_feed': (1400, 100, 1900, 400),  # Upper right
    'objectives': (800, 1000, 1120, 1080) # Bottom center
}
```

### Auto-Resolution Detection
- [ ] **Adaptive Resolution Handling**: Automatically detect and adapt to broadcast resolution
  - Support: 720p, 1080p, 1440p, 4K
  - Dynamically adjust ROI coordinates
  - Scale analysis regions appropriately
- [ ] **Quality Fallback**: Degrade gracefully if stream quality drops
- [ ] **Multi-Source Support**: Handle co-streams with different resolutions

---

## 🔮 v3.3: Intelligence & Automation

**Release Target**: Q4 2026

### Dynamic Contextual Prompts
- [ ] **Phase-Aware Prompting**: Switch instruction sets based on game phase
  - **Early game (0-15min)**: Focus on lane positioning, jungle pathing, vision
  - **Mid game (15-25min)**: Focus on objective control, rotations, item spikes
  - **Late game (25min+)**: Focus on Baron/Elder setup, base defense, win conditions
- [ ] **Composition-Aware Analysis**: Tailor prompts to team composition archetypes
  - Scaling comp: Emphasize gold differential trends
  - Early game comp: Emphasize tempo and objective pressure
  - Team fight comp: Emphasize ultimate cooldowns and positioning

**Example**:
```python
def get_phase_prompt(game_time: int, comp_type: str) -> str:
    if game_time < 15:
        return EARLY_GAME_PROMPT_TEMPLATE.format(comp_type=comp_type)
    elif game_time < 25:
        return MID_GAME_PROMPT_TEMPLATE.format(comp_type=comp_type)
    else:
        return LATE_GAME_PROMPT_TEMPLATE.format(comp_type=comp_type)
```

### Auto-Draft Detection
- [ ] **Pre-Game VLM Call**: Analyze loading screen or draft phase to detect champion compositions
- [ ] **Remove Manual Input**: Eliminate `--blue` and `--red` command-line arguments
- [ ] **Champion Recognition**: OCR or VLM-based champion name extraction from draft UI
- [ ] **Automatic Role Assignment**: Infer Top/Jungle/Mid/ADC/Support from champion picks

**Technical Approach**:
```python
async def detect_draft_from_loading_screen(frame: np.ndarray):
    """
    Extract champion compositions from loading screen.
    """
    response = await genai.GenerativeModel('gemini-2.0-flash-latest').generate_content([
        frame,
        """
        Analyze this League of Legends loading screen.
        
        Extract:
        - Blue team: List 5 champion names (Top, Jungle, Mid, ADC, Support order)
        - Red team: List 5 champion names (Top, Jungle, Mid, ADC, Support order)
        
        Return JSON: {"blue": ["Champ1", ...], "red": ["Champ1", ...]}
        """
    ], generation_config={'response_mime_type': 'application/json'})
    
    return response.json()
```

### Enhanced Historical Context
- [ ] **Item-Level Tracking**: Track individual item purchases across all players
- [ ] **Power Spike Detection**: Automatically detect when champions hit key item breakpoints
  - Mythic completion, core items, legendary spikes
- [ ] **Build Path Analysis**: Evaluate item choices and predict future build direction
- [ ] **Gold Efficiency Scoring**: Calculate relative power based on item value

---

## 🏗️ v4.0: Infrastructure & Scale

**Release Target**: Q1 2027

### Google GenAI SDK Migration
- [ ] **Migrate from `google-generativeai` to `google.genai`**: 
  - Current SDK (`google-generativeai`) is deprecated
  - New SDK (`google.genai`) provides better long-term support
  - Migration required for future Gemini features
- [ ] **Update all imports**: Change `import google.generativeai as genai` → `from google import genai`
- [ ] **Test compatibility**: Ensure all existing functionality works with new SDK
- [ ] **Performance validation**: Benchmark latency and reliability

**Migration Status**: Planned (current SDK still functional)

### Tiered Model Usage
- [ ] **Aggressive Fallback Strategy**: Prioritize cheaper models for simple tasks
  - **Tier 1 (Fastest)**: `gemini-2.0-flash-exp` for state extraction
  - **Tier 2 (Standard)**: `gemini-2.0-flash-latest` for strategic analysis
  - **Tier 3 (Premium)**: `gemini-1.5-pro-latest` for complex macro reasoning
- [ ] **Dynamic Tier Selection**: Automatically choose model based on:
  - Query complexity
  - Available budget
  - Required latency
  - Recent prediction confidence

**Example Logic**:
```python
def select_model_tier(query_complexity: str, budget_remaining: int) -> str:
    if query_complexity == 'simple' or budget_remaining < 5:
        return 'gemini-2.0-flash-exp'      # Fastest experimental
    elif query_complexity == 'standard':
        return 'gemini-2.0-flash-latest'   # Latest stable
    else:
        return 'gemini-1.5-pro-latest'     # Best quality
```

### Prompt Caching
- [ ] **Cache Static Context**: Champion abilities, game rules, map geometry
  - **Cached once per game**: ~300k tokens (champion pool + mechanics)
  - **Variable per request**: ~5k tokens (current state)
- [ ] **Cost Reduction**: Estimated 90% token savings via caching
  - Without caching: 305k tokens × $0.40/M = $0.122/request
  - With caching: 300k cached × $0.04/M + 5k fresh × $0.40/M = $0.014/request
- [ ] **Implementation**: Requires Gemini API caching feature (check availability)

**Cached Context Structure**:
```python
CACHED_CONTEXT = f"""
<game_rules>
{LEAGUE_MECHANICS_GUIDE}  # 50k tokens
</game_rules>

<champion_pool>
{ALL_CHAMPION_ABILITIES}  # 200k tokens
</champion_pool>

<map_geometry>
{SUMMONERS_RIFT_DATA}     # 50k tokens
</map_geometry>
"""

# Only send fresh per request:
FRESH_STATE = f"""
Game Time: {timestamp}
Gold Diff: {gold}
Positions: {positions}
"""
```

### Web UI Dashboard
- [ ] **Frontend**: React/Next.js-based web dashboard
- [ ] **Real-Time Updates**: WebSocket connection to Python backend
- [ ] **Visualizations**:
  - Live win probability chart
  - Gold difference graph over time
  - Minimap overlay with champion positions
  - Objective timeline
  - Strategic reasoning panel
- [ ] **Multi-Game Support**: Monitor multiple streams simultaneously
- [ ] **Historical Replay**: Review past predictions and analysis

**Tech Stack**:
- Frontend: Next.js + TypeScript + TailwindCSS
- Backend: FastAPI (Python) with WebSocket support
- State Management: React Context or Zustand
- Charts: Recharts or Chart.js

---

## 🔬 Research & Experimental

**These are exploratory features without committed timelines:**

### Multi-Stream Analysis
- Analyze multiple concurrent games
- Cross-game pattern recognition
- Meta trend detection across regions (LPL vs LCK vs LEC)

### Community Features
- Public API for accessing predictions
- Discord/Twitch bot integration
- Live prediction leaderboards

### Advanced Analytics
- Champion synergy scoring
- Draft phase win probability prediction
- Player performance attribution
- Coaching insights and replay analysis

---

## 📊 Success Metrics

### v3.1 Goals
- **Context Efficiency**: Reduce average prompt tokens by 40% via context filtering
- **Prediction Stability**: <10% confidence variance between consecutive predictions
- **Narrative Quality**: User survey rating >4/5 for strategic explanations

### v3.2 Goals
- **Event Detection**: Catch 95%+ of major events (kills, objectives) in blind spots
- **Budget Efficiency**: Reduce unnecessary VLM calls by 30% via smart triggering
- **Auto-Draft**: 98%+ accuracy on champion recognition

### v4.0 Goals
- **Cost Reduction**: 90% token savings via prompt caching
- **Latency**: <2s average VLM response time
- **Uptime**: 99.5% availability over 30-day period

---

## 🤝 Contributing

Want to help build these features? Priority areas:

1. **Semantic Anchor Implementation** (v3.1)
2. **Auto-Draft Detection** (v3.3)
3. **Web UI Development** (v4.0)
4. **Prompt Optimization** (All versions)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 Notes

- **Version numbering**: v3.x = Python + Latest Gemini models architecture
- **Breaking changes**: Will be clearly marked and documented
- **Backward compatibility**: Maintained within major versions (3.0 → 3.x)
- **API changes**: SDK migration (v4.0) may require code updates

**For implementation details and current architecture, see [docs/INTEGRATED_DOCUMENTATION.md](docs/INTEGRATED_DOCUMENTATION.md)**
