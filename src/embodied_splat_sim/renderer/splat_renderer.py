import numpy as np
import torch

from .nerfstudio_loader import load_nerfstudio_pipeline


class SplatRenderer:
    def __init__(self, model_dir: str):
        self.pipeline = load_nerfstudio_pipeline(model_dir)
        self.device = self.pipeline.device

    def render(self, camera):
        """
        camera: nerfstudio camera object

        returns: HxWx3 uint8 image
        """
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)
        return rgb
