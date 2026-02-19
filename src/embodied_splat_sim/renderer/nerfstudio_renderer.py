from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from embodied_splat_sim.sim.state import AgentState
from embodied_splat_sim.utils.se3 import make_c2w_from_forward_up, rotz, normalize


class NerfstudioRenderer:
    """
    Nerfstudio-based renderer that converts an AgentState (x,y,z,yaw)
    into a nerfstudio Cameras object and returns an RGB image.

    Design goals:
    - Keep RL env independent from nerfstudio camera APIs
    - Reuse intrinsics/distortion from a reference dataset camera
    - Align agent's yaw=0 to the dataset camera's forward direction
    """

    def __init__(
        self,
        model_dir: str | Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
        device: Optional[str] = None,
        origin_mode: str = "dataset_cam0",
    ):
        """
        Args:
            model_dir: Nerfstudio run directory containing config.yml and nerfstudio_models/
            width/height: Override render resolution. If None, reuse dataset camera resolution.
            device: Optional device override (e.g., "cpu", "mps"). If None, uses pipeline device.
            origin_mode:
                - "dataset_cam0": agent position is an offset from the first dataset camera center
                - "world": agent position is interpreted directly in nerfstudio world coordinates
        """
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yml not found in: {self.model_dir}")

        _, self.pipeline, _, _ = eval_setup(config_path, test_mode="inference")

        if device is not None:
            self.pipeline.to(device)

        self.device = self.pipeline.device

        # Reference camera: provides intrinsics/distortion/camera_type and also a "world alignment".
        ref_cam = self.pipeline.datamanager.train_dataset.cameras[0:1].to(self.device)

        # Cache intrinsics and camera model params from reference camera.
        self.fx = ref_cam.fx.clone()
        self.fy = ref_cam.fy.clone()
        self.cx = ref_cam.cx.clone()
        self.cy = ref_cam.cy.clone()
        self.distortion_params = (
            ref_cam.distortion_params.clone() if ref_cam.distortion_params is not None else None
        )
        self.camera_type = ref_cam.camera_type.clone()
        self.times = ref_cam.times.clone() if ref_cam.times is not None else None
        self.metadata = ref_cam.metadata

        # Determine output resolution.
        self.width = int(width) if width is not None else int(ref_cam.width.item())
        self.height = int(height) if height is not None else int(ref_cam.height.item())

        # Reference camera pose.
        ref_c2w = ref_cam.camera_to_worlds[0].detach().cpu().numpy()  # (3,4)
        ref_pos = ref_c2w[:, 3]
        ref_R = ref_c2w[:, :3]

        # Nerfstudio camera convention: +Z is back, so look direction is -Z.
        # The camera back axis in world is ref_R[:,2], so forward/look is -ref_R[:,2].
        ref_forward = normalize(-ref_R[:, 2])

        # Compute a baseline yaw offset so that agent yaw=0 matches ref camera's XY forward direction.
        # This stabilizes early tests and avoids "camera facing the wrong way" surprises.
        self.yaw0 = math.atan2(float(ref_forward[1]), float(ref_forward[0]))

        self.origin_mode = origin_mode
        self.origin_world = ref_pos.astype(np.float32)

    def _state_to_camera(self, state: AgentState) -> Cameras:
        """
        Convert agent state into a single-camera nerfstudio Cameras object.
        """
        if self.origin_mode == "dataset_cam0":
            pos_world = self.origin_world + np.array([state.x, state.y, state.z], dtype=np.float32)
        elif self.origin_mode == "world":
            pos_world = np.array([state.x, state.y, state.z], dtype=np.float32)
        else:
            raise ValueError(f"Unknown origin_mode: {self.origin_mode}")

        yaw = float(self.yaw0 + state.yaw)

        # Our agent "forward" in world XY plane.
        forward_world = rotz(yaw) @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

        c2w = make_c2w_from_forward_up(pos_world, forward_world)  # (3,4)

        camera_to_worlds = torch.from_numpy(c2w).to(self.device).unsqueeze(0)  # (1,3,4)

        cams = Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=torch.tensor([self.width], device=self.device),
            height=torch.tensor([self.height], device=self.device),
            distortion_params=self.distortion_params,
            camera_type=self.camera_type,
            times=self.times,
            metadata=self.metadata,
        )
        return cams

    @torch.no_grad()
    def render(self, state: AgentState) -> np.ndarray:
        """
        Render an RGB image for the given agent state.

        Returns:
            HxWx3 uint8 RGB image.
        """
        cams = self._state_to_camera(state)
        outputs = self.pipeline.model.get_outputs_for_camera(cams)

        rgb = outputs["rgb"].detach().cpu().numpy()  # (H,W,3), float in [0,1]
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb_u8
