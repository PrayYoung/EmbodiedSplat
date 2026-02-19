from __future__ import annotations

import math
import numpy as np


def rotz(yaw_rad: float) -> np.ndarray:
    """Rotation matrix for a yaw (rotation about +Z) in radians."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a vector to unit length."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def make_c2w_from_forward_up(
    position_world: np.ndarray,
    forward_world: np.ndarray,
    up_world: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> np.ndarray:
    """
    Build a camera-to-world matrix (3x4) using nerfstudio camera conventions:
    - Camera space: +X right, +Y up, +Z back (away from camera)
    - Look direction is -Z

    We choose:
      z_cam (back axis) = -forward_world
      x_cam (right axis) = forward_world x up_world
      y_cam (up axis) = up_world

    Returns:
      c2w: (3,4) float32, [R|t], columns are (x_cam, y_cam, z_cam, t) in world.
    """
    f = normalize(forward_world)
    up = normalize(up_world)

    # Camera back axis (+Z in camera coords) points opposite the look direction.
    z_cam = normalize(-f)

    # Right axis: forward x up (gives right-handed basis with z-up world).
    x_cam = normalize(np.cross(f, up))

    # Up axis: ensure orthonormal by recomputing y as z x x.
    y_cam = normalize(np.cross(z_cam, x_cam))

    R = np.stack([x_cam, y_cam, z_cam], axis=1)  # (3,3)
    t = position_world.reshape(3, 1).astype(np.float32)  # (3,1)

    c2w = np.concatenate([R, t], axis=1).astype(np.float32)  # (3,4)
    return c2w
