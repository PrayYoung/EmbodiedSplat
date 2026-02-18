from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentState:
    x: float
    y: float
    z: float
    yaw: float  # radians
