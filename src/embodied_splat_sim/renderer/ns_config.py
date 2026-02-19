from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class NerfstudioConfig:
    scene: str
    use_latest_run: bool
    model_dir: Optional[str]
    origin_mode: str
    device: Optional[str]


def load_nerfstudio_config(path: str | Path = "configs/nerfstudio.yaml") -> NerfstudioConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    ns = data.get("nerfstudio", {})

    return NerfstudioConfig(
        scene=str(ns.get("scene", "living_room")),
        use_latest_run=bool(ns.get("use_latest_run", True)),
        model_dir=ns.get("model_dir", None),
        origin_mode=str(ns.get("origin_mode", "dataset_cam0")),
        device=ns.get("device", None),
    )


def resolve_model_dir(cfg: NerfstudioConfig) -> Path:
    """
    Resolve the nerfstudio run directory that contains config.yml.

    Priority:
      1) cfg.model_dir if provided
      2) latest directory under scenes/<scene>/runs/
    """
    if cfg.model_dir:
        model_dir = Path(cfg.model_dir)
        if not (model_dir / "config.yml").exists():
            raise FileNotFoundError(f"config.yml not found in model_dir: {model_dir}")
        return model_dir

    runs_dir = Path("scenes") / cfg.scene / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {runs_dir}. Train first:\n"
            f"  bash scripts/ns/process_video.sh {cfg.scene}\n"
            f"  bash scripts/ns/train_splatfacto.sh {cfg.scene}"
        )

    latest = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    if not (latest / "config.yml").exists():
        raise FileNotFoundError(f"Latest run dir missing config.yml: {latest}")

    return latest
