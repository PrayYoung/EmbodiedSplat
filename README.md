# EmbodiedSplat

EmbodiedSplat is a photorealistic reinforcement learning environment for embodied agents built on 3D Gaussian Splatting (3DGS).

The project explores how agents can learn navigation and spatial reasoning directly from image observations rendered from real-world scenes. By combining world-model scene representations with reinforcement learning, EmbodiedSplat serves as a foundation for research in:

- Embodied AI
- Robotics navigation
- Spatial intelligence
- World models
- Sim-to-real transfer

---

## Features

- Goal-conditioned navigation environment (Gymnasium)
- Agent dynamics and reward shaping
- PPO training pipeline (Stable-Baselines3)
- Image-based observations
- Photorealistic rendering via Nerfstudio + 3D Gaussian Splatting
- Train scenes from phone videos or photos
- Renderer abstraction for future world model backends

---

## System Architecture

Agent Policy (PPO)
    ↓
Navigation Environment (Gym)
    ↓
Renderer Interface
    ↓
3D Gaussian Splatting Scene
    ↓
Photorealistic RGB Observations

This modular design allows replacing the rendering backend or extending to robotics pipelines.

---

## Quick Start

### 1. Clone Repository

git clone https://github.com/PrayYoung/EmbodiedSplat.git  
cd EmbodiedSplat

### 2. Create Environment (uv)

uv venv --python 3.11  
source .venv/bin/activate  
uv sync  
uv pip install -e .

### 3. Run PPO Training (Dummy Renderer)

uv run python scripts/rl/train_ppo.py

This verifies the RL environment and training pipeline.

---

## Creating a Photorealistic Scene

EmbodiedSplat supports:

- training from a phone video
- loading existing Nerfstudio models

### Install Dependencies (macOS)

brew install colmap ffmpeg  
uv pip install nerfstudio

---

### Step 1 — Process Video

Place your video:

scenes/living_room/raw/video.mp4

Run:

bash scripts/ns/process_video.sh living_room

This extracts frames and computes camera poses.

---

### Step 2 — Train 3DGS Scene

bash scripts/ns/train_splatfacto.sh living_room

This creates:

scenes/living_room/runs/splatfacto/

---

### Step 3 — Render a Test Image

uv run python scripts/render_one.py

You should see:

Saved to output.png

---

## Scene Directory Structure

scenes/<scene_name>/  
  raw/            # video or images  
  processed/      # ns-process-data output  
  runs/           # trained splatfacto model  

This structure supports reproducible pipelines and external model loading.

---

## Using an Existing Nerfstudio Model

If you already have a trained scene containing:

config.yml  
nerfstudio_models/

Simply set the model directory in render_one.py.

---

## Roadmap

Phase 1  
- RL navigation environment  
- PPO training pipeline  
- Dummy renderer  

Phase 2  
- 3DGS rendering integration  
- Replace dummy renderer in environment  

Phase 3  
- Collision & depth-based navigation  
- Goal randomization & curriculum learning  
- Domain randomization  

Phase 4  
- Sim-to-real experiments  
- Robotics integration  
- Semantic navigation  

---

## Research Motivation

Photorealistic world models are becoming essential for embodied intelligence. Training agents in visually realistic environments enables:

- spatial reasoning
- perception-action learning
- transfer to real-world robotics

EmbodiedSplat explores the intersection of world models, reinforcement learning, and spatial intelligence.

---

## Tech Stack

Python 3.11  
Gymnasium  
Stable-Baselines3  
Nerfstudio  
3D Gaussian Splatting  
PyTorch  

---

## License

MIT License

---

## Acknowledgments

Nerfstudio team  
3D Gaussian Splatting research community  
Embodied AI & robotics research pioneers
