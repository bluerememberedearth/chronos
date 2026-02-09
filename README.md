# Chronos v3: Autonomous League of Legends Prediction Engine 🔮

**Chronos v3** is an advanced AI system that watches live League of Legends esports broadcasts (LPL, LCK, LEC, LCS) and provides real-time strategic analysis and win probability predictions. It leverages **Google's latest Gemini models** (2.0 Flash, experimental variants, and 1.5 Pro) combined with traditional Computer Vision to "see" the game state, understand macro-level flow, and predict outcomes.

## 🚀 Key Features

- **Live Stream Ingestion**: Connects directly to Twitch, YouTube, and Huya streams via `streamlink` and `ffmpeg` (1080p60 support)
- **Hybrid Vision Pipeline**: Traditional CV for speed + Vision Language Models for strategic intelligence
- **Intelligent Budget Management**: Operates indefinitely within free-tier API limits via model pool rotation
- **Triple-Thread Architecture**: Decoupled ingestion, main loop, and VLM threads ensure zero frame drops
- **Narrative Intelligence**: Tracks fog of war, validates game state consistency, and builds match narrative
- **Real-Time Dashboard**: Clean CLI output with gold differences, objective timers, win probability, and strategic reasoning

## 🛠️ Tech Stack

- **Core**: Python 3.10+, AsyncIO, Threading
- **AI/VLM**: Google Gemini (latest models via `google-generativeai`)
- **Vision**: OpenCV (`cv2`), Pillow, NumPy
- **Stream**: Streamlink, FFmpeg, yt-dlp
- **Data**: Riot DataDragon (Static game assets)

## 📦 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bluerememberedearth/chronos.git
cd chronos

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*Note: Requires `ffmpeg` installed on your system path.*

### Configuration

Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_api_key_here
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

### Run Live Prediction

```bash
python scripts/predict_live.py "https://www.twitch.tv/lec" \
  --blue "Sion,Sylas,Yone,Corki,Bard" \
  --red "DrMundo,Pantheon,Azir,Varus,Alistar"
```

**Parameters:**
- `URL`: Stream URL (Twitch/YouTube/Huya)
- `--blue`, `--red`: Comma-separated list of 5 champions (Top, Jungle, Mid, ADC, Support)
- `--budget`: Optional max VLM calls (default 15)

## 🏗️ Architecture

```
Stream Source (Twitch/YouTube/Huya)
        │
        ▼
Ingestion Thread (Frame Buffer)
        │
        ▼
Main Loop (Cadence: 120s / 300s)
        │
        ▼
VLM Thread Pool (Model Rotation)
  ├─ gemini-2.0-flash-exp (Latest Experimental)
  ├─ gemini-2.0-flash-latest (Latest Stable)
  └─ gemini-1.5-pro-latest (Premium Deep Analysis)
        │
        ▼
Intelligence Layer (Game Memory + Narrative)
        │
        ▼
CLI Dashboard (Predictions + Reasoning)
```

### Design Philosophy

**Hybrid Approach**: Use traditional CV for fast, deterministic tasks (gold, timers) and Vision Language Models for intelligent, adaptive analysis (strategy, positioning, win conditions).

**Budget-First**: Designed to run 24/7 on free-tier Gemini API via:
- Model pool rotation (20 RPD × 3 models = 60+ calls/day)
- Cadence-based triggering (predictable usage)
- Tiered analysis (cheap models for simple tasks, premium for deep reasoning)

## 📚 Documentation

- **[Complete Documentation](docs/INTEGRATED_DOCUMENTATION.md)**: Full technical guide with implementation details
- **[Development Roadmap](ROADMAP.md)**: Planned features and improvements
- **[v2 Design Document](docs/chronos_v2_design.md)**: Archived theoretical architecture (not implemented)

## 🧪 Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/predict_live.py` | Production script with cadence-based budgeting |
| `scripts/live_game.py` | Poll-based updates (30-60s intervals) |
| `scripts/debug_stream.py` | Test stream ingestion without API calls |
| `scripts/test_fog_of_war.py` | Validate objective tracking logic |
| `scripts/inspect_huya.py` | Debug Chinese Huya stream parsing |
| `scripts/ping_all_models.py` | Check Gemini model availability |

## 🎯 Roadmap

### ✅ Completed
- [x] Cadence-based triggers for predictable API usage
- [x] Multi-model rotation (120+ daily calls)
- [x] Triple-thread architecture
- [x] Narrative intelligence and fog of war tracking

### 🚧 In Progress (v3.1)
- [ ] Semantic anchor for cross-model context retention
- [ ] Context density filtering based on game phase
- [ ] Region-based motion detection
- [ ] Frame buffer blind spot analysis

### 📅 Planned (v3.2+)
- [ ] Auto-draft detection (remove manual champion input)
- [ ] Google GenAI SDK migration (`google.genai`)
- [ ] Prompt caching for 90% cost reduction
- [ ] Web UI (React/Next.js dashboard)

See [ROADMAP.md](ROADMAP.md) for detailed feature descriptions.

## 🧠 How It Works

### Perception Pipeline

1. **Fast Path (Every Frame)**:
   - Extract gold, CS, KDA via OpenCV
   - Track minimap champion positions
   - Parse objective timers

2. **Smart Path (Every 120-300s)**:
   - Send frame to Gemini Vision API
   - Analyze team positioning and vision control
   - Evaluate win conditions and strategic intent
   - Generate win probability prediction

### Model Pool Rotation

Free tier = 20 requests/day per model. Solution:

```python
models = [
    'gemini-2.0-flash-exp',      # Latest experimental (fastest)
    'gemini-2.0-flash-latest',   # Latest stable 2.0
    'gemini-1.5-flash-latest',   # Latest stable 1.5
    'gemini-1.5-pro-latest'      # Latest premium (deep reasoning)
]

# Rotate to next available model when one hits limit
# Result: 80-120 requests/day sustainable with latest models
```

### Narrative Intelligence

The system maintains a semantic understanding of the match:

- **Fog of War**: Tracks MIA champions and objective permanence
- **Game Memory**: Historical context across match timeline
- **Self-Correction**: Validates state consistency between frames
- **Strategic Narrative**: Compresses quantitative data into high-level story

## ⚙️ API Budget Management

**Free Tier Constraints**:
- 20 requests/day per Gemini model
- 2 requests/minute rate limit per model
- 1M token context window

**Our Strategy**:
- **Cadence-based**: Fixed 120s/300s intervals (predictable usage)
- **Model rotation**: 3+ models × 20 RPD = 60+ requests minimum
- **Tiered analysis**: Match model cost to query complexity
- **Result**: Sustainable 24/7 operation at $0 cost

## 📖 Example Output

```
=== Game State (14:32) ===
Gold Difference: +2,450 (Blue advantage)
Dragon Control: Blue 2 - Red 1
Baron Timer: 5:28

Win Probability: Blue 68% | Red 32%

Strategic Analysis:
Blue team is leveraging their early gold lead through superior vision 
control around Baron pit. Red's scaling comp (Jinx/Azir) needs to 
stall for 10+ minutes, but Blue is setting up Baron siege. Expect 
forced fight in next 2 minutes.

Confidence: 0.74
Next Update: 120s
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Auto-draft detection via VLM
- Additional stream platform support
- Web UI development
- Prompt optimization for better predictions

## ⚖️ License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **Riot Games**: DataDragon static asset API
- **Google**: Gemini API and multimodal foundation models
- **Streamlink**: Stream ingestion infrastructure

---

**For complete technical documentation, see [docs/INTEGRATED_DOCUMENTATION.md](docs/INTEGRATED_DOCUMENTATION.md)**
