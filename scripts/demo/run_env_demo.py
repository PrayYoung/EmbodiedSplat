import argparse
import subprocess
from pathlib import Path

import numpy as np

from embodied_splat_sim.envs.splat_nav_env import SplatNavEnv


def _start_ffmpeg(video_path: Path, width: int, height: int, fps: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", choices=["dummy", "nerfstudio"], default="dummy")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="demo.mp4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    env = SplatNavEnv(config_path="configs/env.yaml", renderer_mode=args.renderer, seed=0)
    obs, info = env.reset(seed=0)

    frames = []
    rng = np.random.default_rng(0)

    ffmpeg_proc = None
    if args.save_video:
        video_path = Path(args.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_proc = _start_ffmpeg(video_path, env.cfg.width, env.cfg.height, args.fps)

    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if args.renderer == "dummy":
            frame = rng.integers(0, 256, size=obs.shape, dtype=np.uint8)
        else:
            frame = env.render()

        frames.append(frame)
        if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.write(frame.tobytes())

        dist = info.get("distance_to_goal")
        collision = bool(info.get("collision", False))
        print(f"step={step} reward={reward:.4f} dist={dist:.4f} collision={collision}")

        if terminated or truncated:
            break

    if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()


if __name__ == "__main__":
    main()
