from embodied_splat_sim.renderer import NerfstudioRenderer
from embodied_splat_sim.sim.state import AgentState
import cv2

def main():
    model_dir = "scenes/living_room/runs/<YOUR_RUN_DIR>"
    r = NerfstudioRenderer(model_dir, width=640, height=360)

    # Start at origin (dataset cam0) and rotate in place.
    for i in range(8):
        yaw = i * (3.14159 / 4.0)
        img = r.render(AgentState(x=0.0, y=0.0, z=0.0, yaw=yaw))
        cv2.imwrite(f"assets/demo/state_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
