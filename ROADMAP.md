# Chronos v3 Roadmap 🚀

This document outlines the planned improvements and future features for the Chronos Live Prediction Pipeline.

## Strategic Epic: Semantic Intelligence & Context Management 🧠
*Shifting the Intelligence Layer from "Data Recording" to a "Semantic Anchor" that manages context density for the VLM brain.*

- [ ] **Semantic Anchor (Cross-Model Grounding)**: Ensure high-level VLM conclusions (e.g., "Jinx is the primary win-con") persist across physical model rotations (429 fallbacks) so the "thought process" doesn't reset.
- [ ] **Context Density Filtering**: Dynamically prioritize which historical data to pass based on current game events (e.g., prioritize item timings during resets, but prioritize positioning during objective setup).
- [ ] **Narrative Compression**: Summarize quantitative data into high-density strategic paragraphs that describe the *impact* of events rather than just the numbers.
- [ ] **Reflection Loop**: Feed previous reasoning "misses" into the history so the VLM can adjust its confidence and logic in real-time.

## Core Perception Improvements
- [ ] **Frame Buffer (Blind Spot Analysis)**: Retrospectively analyze the 2-5 second window during VLM calls to ensure no key events (like sudden kills or objective steals) are missed.
- [ ] **Region-Based Motion Detection**: Move from full-frame pixel diffs to specific UI region monitoring (Scoreboard, Minimap, Kill Feed) to trigger VLM calls only on relevant game state changes.
- [ ] **Auto-Resolution Detection**: Enable the stream loader to automatically detect and adapt to 720p, 1080p, and 4K broadcast resolutions.

## Intelligence & Reasoning
- [ ] **Dynamic Contextual Prompts**: Switch instruction sets based on game phase (e.g., specific focus on lane positioning in early game vs. high-ground defense/baron positioning in late game).
- [ ] **Auto-Draft Detection**: Implement a pre-game VLM call to identify champion compositions from the loading screen or draft phase, removing the need for manual champion input.
- [ ] **Enhanced Historical Context**: Expand the history summary to include "Item Level" tracking to detect power spikes across the whole team automatically.

## Infrastructure & Efficiency
- [ ] **Google GenAI SDK Migration**: Migrate from the deprecated `google.generativeai` package to the modern `google.genai` SDK for better long-term support.
- [ ] **Tiered Model Usage**: Implement a more aggressive fallback strategy that prioritizes cheaper models for simple state extraction and expensive models (Gemini 2.5 Pro) for high-stakes strategic predictions.
