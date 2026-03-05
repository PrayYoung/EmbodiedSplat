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
    goal_radius = float(data["sim"]["goal_radius"])
    enable_collision = bool(data["sim"].get("enable_collision", False))
    collision_min_depth = float(data["sim"].get("collision_min_depth", 0.25))

    action_type = str(data["action"]["type"])
    forward_step = float(data["action"]["forward_step"])
    turn_deg = float(data["action"]["turn_deg"])

    goal_xy = tuple(map(float, data["task"]["goal_xy"]))
    start_xy = tuple(map(float, data["task"]["start_xy"]))
    start_yaw_deg = float(data["task"]["start_yaw_deg"])
    randomize_goal = bool(data["task"].get("randomize_goal", False))
    goal_box_raw = data["task"].get("goal_box", [[goal_xy[0], goal_xy[1]], [goal_xy[0], goal_xy[1]]])
    randomize_start = bool(data["task"].get("randomize_start", False))
    start_box_raw = data["task"].get(
        "start_box",
        [[start_xy[0], start_xy[1], start_yaw_deg], [start_xy[0], start_xy[1], start_yaw_deg]],
    )
    goal_box = (
        (float(goal_box_raw[0][0]), float(goal_box_raw[0][1])),
        (float(goal_box_raw[1][0]), float(goal_box_raw[1][1])),
    )

    cfg = EnvConfig(
        width=width,
        height=height,
        dt=dt,
        goal_radius=goal_radius,
        enable_collision=enable_collision,
        collision_min_depth=collision_min_depth,
        action_type=action_type,
        forward_step=forward_step,
        turn_deg=turn_deg,
        goal_xy=(goal_xy[0], goal_xy[1]),
        start_xy=(start_xy[0], start_xy[1]),
        start_yaw_deg=start_yaw_deg,
        randomize_goal=randomize_goal,
        goal_box=goal_box,
    )
    cfg.randomize_start = randomize_start
    cfg.start_box = (
        (float(start_box_raw[0][0]), float(start_box_raw[0][1]), float(start_box_raw[0][2])),
        (float(start_box_raw[1][0]), float(start_box_raw[1][1]), float(start_box_raw[1][2])),
    )
    reward_cfg = data.get("reward", {})
    cfg.step_penalty = float(reward_cfg.get("step_penalty", -0.01))
    cfg.progress_scale = float(reward_cfg.get("progress_scale", 1.0))
    cfg.success_bonus = float(reward_cfg.get("success_bonus", 0.0))
    cfg.collision_penalty = float(reward_cfg.get("collision_penalty", 0.0))
    cfg.success_radius = float(data["task"].get("success_radius", goal_radius))
    cfg.max_steps = int(data["task"].get("max_steps", 200))
    return cfg


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

    def render(self, state: AgentState, return_depth: bool = False):
        """
        Render a simple RGB image encoding agent yaw and position in a stable way.

        Output: uint8 HxWx3
        """
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Encode yaw as a horizontal gradient in G channel.
        # yaw in [-pi, pi] -> [0, 255]
        yaw = (state.yaw + math.pi) % (2 * math.pi) - math.pi
        gval = int(np.clip((yaw + math.pi) / (2 * math.pi) * 255.0, 0, 255))
        img[:, :, 1] = gval

        # Encode (x, y) as low-frequency patterns for stability.
        # This makes the observation non-trivial but independent of the goal.
        xval = int(np.clip((state.x * 10.0 + 128.0), 0, 255))
        yval = int(np.clip((state.y * 10.0 + 128.0), 0, 255))
        img[:, :, 2] = xval
        img[:, :, 0] = yval

        # Draw a heading dot (for debugging/visual sanity).
        cx, cy = self.w // 2, self.h // 2
        px = int(cx + (self.w * 0.2) * math.cos(state.yaw))
        py = int(cy - (self.h * 0.2) * math.sin(state.yaw))
        px = int(np.clip(px, 0, self.w - 1))
        py = int(np.clip(py, 0, self.h - 1))
        img[max(0, py - 3) : min(self.h, py + 4), max(0, px - 3) : min(self.w, px + 4), :] = 255

        if not return_depth:
            return img
        depth = np.full((self.h, self.w), np.inf, dtype=np.float32)
        return img, depth


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

        self.max_steps = int(getattr(self.cfg, "max_steps", max_steps))
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

    def _sample_goal(self) -> Tuple[float, float]:
        (xmin, ymin), (xmax, ymax) = self.cfg.goal_box
        gx = float(self._rng.uniform(xmin, xmax))
        gy = float(self._rng.uniform(ymin, ymax))
        return gx, gy

    def _sample_start_state(self) -> AgentState:
        (xmin, ymin, tmin), (xmax, ymax, tmax) = self.cfg.start_box
        sx = float(self._rng.uniform(xmin, xmax))
        sy = float(self._rng.uniform(ymin, ymax))
        yaw = math.radians(float(self._rng.uniform(tmin, tmax)))
        return AgentState(x=sx, y=sy, z=0.0, yaw=yaw)

    def _terminated(self) -> bool:
        return self._dist_to_goal(self.state) <= self.cfg.goal_radius

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        if getattr(self.cfg, "randomize_start", False):
            self.state = self._sample_start_state()
        else:
            self.state = self._make_start_state()

        if self.cfg.randomize_goal:
            self.goal_xy = self._sample_goal()
        else:
            self.goal_xy = self.cfg.goal_xy
        self._prev_dist = self._dist_to_goal(self.state)

        obs = self.renderer.render(self.state)
        info = {"distance_to_goal": self._prev_dist, "collision": False}
        return obs, info

    def step(self, action: int):
        self._step_count += 1

        # Update state
        turn_rad = math.radians(self.cfg.turn_deg)
        candidate = step_state_discrete(
            self.state,
            int(action),
            forward_step=self.cfg.forward_step,
            turn_rad=turn_rad,
        )
        collision = False
        if self.cfg.enable_collision and int(action) in (0, 1, 2, 3):
            _, depth = self.renderer.render(candidate, return_depth=True)
            if depth is not None:
                h, w = depth.shape[:2]
                cy, cx = h // 2, w // 2
                r = 2
                patch = depth[max(0, cy - r) : min(h, cy + r + 1), max(0, cx - r) : min(w, cx + r + 1)]
                valid = np.isfinite(patch) & (patch > 0)
                valid_ratio = float(np.mean(valid))
                if valid_ratio >= 0.5:
                    p5 = float(np.percentile(patch[valid], 5))
                    if p5 < self.cfg.collision_min_depth:
                        collision = True
        if collision:
            candidate = self.state
        self.state = candidate

        # Reward
        dist = self._dist_to_goal(self.state)
        progress = self._prev_dist - dist
        reward = float(self.cfg.step_penalty + self.cfg.progress_scale * progress)
        self._prev_dist = dist
        terminated = dist <= self.cfg.success_radius
        if terminated:
            reward += float(self.cfg.success_bonus)
        if collision:
            reward += float(self.cfg.collision_penalty)
        truncated = self._step_count >= self.max_steps

        obs = self.renderer.render(self.state)
        info = {"distance_to_goal": dist, "step_count": self._step_count, "collision": collision}

        return obs, reward, terminated, truncated, info

    def render(self):
        # Gymnasium expects render() to return an image in rgb_array mode
        return self.renderer.render(self.state)
