from pathlib import Path
import numpy as np
import cv2

from embodied_splat_sim.renderer import SplatRenderer


def main():
    model_dir = "runs/nerfstudio/demo"

    renderer = SplatRenderer(model_dir)

    print("Renderer loaded successfully")


if __name__ == "__main__":
    main()
