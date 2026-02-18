from pathlib import Path
import numpy as np
import cv2
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


def load_pipeline(model_dir: str):
    """
    Load nerfstudio pipeline for inference.

    model_dir must contain:
        config.yml
        nerfstudio_models/
    """
    config_path = Path(model_dir) / "config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found in {model_dir}")

    _, pipeline, _, _ = eval_setup(
        config_path,
        test_mode="inference",
    )

    return pipeline


def create_camera(pipeline):
    """
    Create a simple camera using the first training camera pose.

    This ensures the pose is valid and inside the reconstructed scene.
    """

    # Use an existing camera pose from the dataset
    camera = pipeline.datamanager.train_dataset.cameras[0:1]

    # Move to correct device
    camera = camera.to(pipeline.device)

    return camera


def render_rgb(pipeline, camera):
    """
    Render an RGB image from the given camera.
    """
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)

    rgb = outputs["rgb"].cpu().numpy()
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def main():
    model_dir = "scenes/living_room/runs/splatfacto"  # ← change if needed

    pipeline = load_pipeline(model_dir)
    print("Pipeline loaded")

    camera = create_camera(pipeline)
    print("Camera created")

    rgb = render_rgb(pipeline, camera)
    print("Rendered image")

    output_path = Path("output.png")
    cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
