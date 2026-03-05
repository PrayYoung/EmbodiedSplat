import argparse
import math
from pathlib import Path

import numpy as np

from embodied_splat_sim.envs.splat_nav_env import SplatNavEnv
from embodied_splat_sim.sim.state import AgentState


def depth_stats(depth: np.ndarray):
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None
    v = depth[valid]
    return {
        "min": float(np.min(v)),
        "p5": float(np.percentile(v, 5)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
        "valid_ratio": float(np.mean(valid)),
    }


def save_depth_pgm(path: Path, depth: np.ndarray):
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return
    v = depth.copy()
    v[~valid] = np.nan
    dmin = float(np.nanpercentile(v, 5))
    dmax = float(np.nanpercentile(v, 95))
    if dmax <= dmin:
        dmin = float(np.nanmin(v))
        dmax = float(np.nanmax(v))
    norm = (v - dmin) / max(dmax - dmin, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)

    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(img.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    env = SplatNavEnv(config_path=args.config, renderer_mode="nerfstudio", seed=0)
    env.reset(seed=0)

    poses = []
    base = env.state
    for i in range(args.poses):
        yaw = base.yaw + (i - args.poses // 2) * (math.pi / 12.0)
        poses.append(AgentState(x=base.x, y=base.y, z=base.z, yaw=yaw))

    for i, state in enumerate(poses):
        rgb, depth = env.renderer.render(state, return_depth=True)
        if depth is None:
            print(f"pose {i}: depth unavailable")
            continue
        stats = depth_stats(depth)
        print(f"pose {i}: {stats}")
        if args.save:
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            save_depth_pgm(out.with_suffix(f".pose{i}.pgm"), depth)


if __name__ == "__main__":
    main()
