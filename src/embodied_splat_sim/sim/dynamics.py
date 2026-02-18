from __future__ import annotations

import math
from typing import Literal

from .state import AgentState

# Discrete actions (MVP)
# 0: forward, 1: back, 2: strafe_left, 3: strafe_right, 4: turn_left, 5: turn_right
Action = Literal[0, 1, 2, 3, 4, 5]


def step_state_discrete(
    s: AgentState,
    action: int,
    forward_step: float,
    turn_rad: float,
) -> AgentState:
    # Copy to avoid in-place mutation surprises
    x, y, z, yaw = s.x, s.y, s.z, s.yaw

    # Heading vectors in XY plane
    fx = math.cos(yaw)
    fy = math.sin(yaw)
    # Left vector
    lx = -math.sin(yaw)
    ly = math.cos(yaw)

    if action == 0:  # forward
        x += forward_step * fx
        y += forward_step * fy
    elif action == 1:  # back
        x -= forward_step * fx
        y -= forward_step * fy
    elif action == 2:  # strafe_left
        x += forward_step * lx
        y += forward_step * ly
    elif action == 3:  # strafe_right
        x -= forward_step * lx
        y -= forward_step * ly
    elif action == 4:  # turn_left
        yaw += turn_rad
    elif action == 5:  # turn_right
        yaw -= turn_rad
    else:
        raise ValueError(f"Unknown discrete action: {action}")

    # Normalize yaw to [-pi, pi]
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return AgentState(x=x, y=y, z=z, yaw=yaw)
