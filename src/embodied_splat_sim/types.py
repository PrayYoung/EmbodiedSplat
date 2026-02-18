from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    width: int
    height: int

    dt: float
    step_penalty: float
    goal_radius: float

    action_type: str
    forward_step: float
    turn_deg: float

    goal_xy: Tuple[float, float]
    start_xy: Tuple[float, float]
    start_yaw_deg: float
