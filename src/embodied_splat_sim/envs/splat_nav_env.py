from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml

from embodied_splat_sim.types import EnvConfig
from embodied_splat_sim.sim.state import AgentState
from embodied_splat_sim.sim.dynamics import step_state_discrete
from embodied_splat_sim.renderer import NerfstudioRenderer


def _load_config(config_path: str | Path) -> EnvConfig:
    p = Path(config_path)
    data = yaml.safe_load(p.read_text())

    width = int(data["render"]["width"])
    height = int(data["render"]["height"])

    dt = float(data["sim"]["dt"])
    step_penalty = float(data["sim"]["step_penalty"])
    goal_radius = float(data["sim"]["goal_radius"])

    action_type = str(data["action"]["type"])
    forward_step = float(data["action"]["forward_step"])
    turn_deg = float(data["action"]["turn_deg"])

    goal_xy = tuple(map(float, data["task"]["goal_xy"]))
    start_xy = tuple(map(float, data["task"]["start_xy"]))
    start_yaw_deg = float(data["task"]["start_yaw_deg"])

    return EnvConfig(
        width=width,
        height=height,
        dt=dt,
        step_penalty=step_penalty,
        goal_radius=goal_radius,
        action_type=action_type,
        forward_step=forward_step,
        turn_deg=turn_deg,
        goal_xy=(goal_xy[0], goal_xy[1]),
        start_xy=(start_xy[0], start_xy[1]),
        start_yaw_deg=start_yaw_deg,
    )


class DummyRenderer:
    """
    MVP renderer: returns a simple RGB observation encoding:
      - distance-to-goal (as intensity)
      - relative bearing (as gradient)
      - a small dot showing heading
    Output: uint8 HxWx3
    """

    def __init__(self, width: int, height: int):
        self.w = int(width)
        self.h = int(height)

    def render(self, state: AgentState, goal_xy: Tuple[float, float]) -> np.ndarray:
        gx, gy = goal_xy
        dx = gx - state.x
        dy = gy - state.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        bearing = math.atan2(dy, dx)  # world bearing to goal
        rel = bearing - state.yaw
        rel = (rel + math.pi) % (2 * math.pi) - math.pi  # [-pi, pi]

        # Base image: distance encoded as brightness
        # closer -> brighter
        closeness = 1.0 / (1.0 + dist)  # (0,1]
        base = int(np.clip(closeness * 255.0, 0, 255))

        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:, :, :] = base

        # Encode relative bearing as a left-right gradient in G channel
        # rel=-pi -> 0, rel=+pi -> 255
        gval = int(np.clip((rel + math.pi) / (2 * math.pi) * 255.0, 0, 255))
        img[:, :, 1] = gval

        # Heading dot in R channel
        cx, cy = self.w // 2, self.h // 2
        # dot position depends on yaw (just for visualization)
        px = int(cx + (self.w * 0.2) * math.cos(state.yaw))
        py = int(cy - (self.h * 0.2) * math.sin(state.yaw))
        px = int(np.clip(px, 0, self.w - 1))
        py = int(np.clip(py, 0, self.h - 1))
        img[max(0, py - 3) : min(self.h, py + 4), max(0, px - 3) : min(self.w, px + 4), 0] = 255

        return img


class SplatNavEnv(gym.Env):
    """
    Minimal Gymnasium environment:
      - Discrete actions
      - Observation is RGB image (dummy for now)
      - Reward: distance decrease - step_penalty
      - Termination: reached goal OR max_steps
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config_path: str = "configs/env.yaml",
        renderer_mode: str = "dummy",
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = _load_config(config_path)
        assert self.cfg.action_type == "discrete", "MVP only supports discrete actions"

        self.max_steps = int(max_steps)
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

        # Spaces
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.cfg.height, self.cfg.width, 3), dtype=np.uint8
        )

        # State + task
        self.goal_xy = self.cfg.goal_xy
        self.state = self._make_start_state()

        # Renderer
        if renderer_mode == "dummy":
            self.renderer = DummyRenderer(self.cfg.width, self.cfg.height)
        elif renderer_mode == "nerfstudio":
            from embodied_splat_sim.renderer.nerfstudio_renderer import NerfstudioRenderer
            from embodied_splat_sim.renderer.ns_config import load_nerfstudio_config, resolve_model_dir

            ns_cfg = load_nerfstudio_config("configs/nerfstudio.yaml")
            model_dir = resolve_model_dir(ns_cfg)

            self.renderer = NerfstudioRenderer(
                model_dir=model_dir,
                width=self.cfg.width,
                height=self.cfg.height,
                device=ns_cfg.device,
                origin_mode=ns_cfg.origin_mode,
            )
        else:
            raise ValueError(f"Unknown renderer_mode={renderer_mode}")

        # Cache
        self._prev_dist = self._dist_to_goal(self.state)

    def _make_start_state(self) -> AgentState:
        yaw = math.radians(self.cfg.start_yaw_deg)
        sx, sy = self.cfg.start_xy
        return AgentState(x=float(sx), y=float(sy), z=0.0, yaw=float(yaw))

    def _dist_to_goal(self, s: AgentState) -> float:
        gx, gy = self.goal_xy
        dx = gx - s.x
        dy = gy - s.y
        return math.sqrt(dx * dx + dy * dy)

    def _terminated(self) -> bool:
        return self._dist_to_goal(self.state) <= self.cfg.goal_radius

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self.state = self._make_start_state()

        # (Optional) randomize goal/start later; keep deterministic now
        self._prev_dist = self._dist_to_goal(self.state)

        obs = self.renderer.render(self.state, self.goal_xy)
        info = {"dist_to_goal": self._prev_dist}
        return obs, info

    def step(self, action: int):
        self._step_count += 1

        # Update state
        turn_rad = math.radians(self.cfg.turn_deg)
        self.state = step_state_discrete(
            self.state,
            int(action),
            forward_step=self.cfg.forward_step,
            turn_rad=turn_rad,
        )

        # Reward
        dist = self._dist_to_goal(self.state)
        progress = self._prev_dist - dist
        reward = float(progress - self.cfg.step_penalty)
        self._prev_dist = dist

        terminated = self._terminated()
        truncated = self._step_count >= self.max_steps

        obs = self.renderer.render(self.state, self.goal_xy)
        info = {"dist_to_goal": dist, "step_count": self._step_count}

        return obs, reward, terminated, truncated, info

    def render(self):
        # Gymnasium expects render() to return an image in rgb_array mode
        return self.renderer.render(self.state, self.goal_xy)
