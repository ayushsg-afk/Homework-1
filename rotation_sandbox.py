import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "../model/asymmetric_body.xml"

# Edit sequence frames here to compare ("current" vs "fixed")
rotation_sequence = [
    ("z", np.deg2rad(90), "fixed"),
    ("x", np.deg2rad(90), "fixed"),
]


def compose_sequence(sequence):
    """
    Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order.
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        if frame == "current":
            R = R @ R_step
        elif frame == "fixed":
            R = R_step @ R
        else:
            raise ValueError(f"Unknown frame specification: '{frame}'")
    return R


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open. Animating rotation sequence in a loop...")

        # Outer infinite loop keeps repeating the animation sequence
        while viewer.is_running():
            # Reset body orientation to starting Identity matrix (0 degrees)
            R_curr = np.eye(3)
            set_body_orientation(data, R_curr)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0)  # Pause briefly at the initial pose before starting

            # Animate each rotation in the sequence
            for axis, total_angle, frame in rotation_sequence:
                steps = 60
                for i in range(1, steps + 1):
                    if not viewer.is_running():
                        break
                    angle_step = total_angle * (i / steps)
                    R_step = ELEMENTARY_ROTATIONS[axis](angle_step)

                    if frame == "current":
                        R_anim = R_curr @ R_step
                    elif frame == "fixed":
                        R_anim = R_step @ R_curr

                    set_body_orientation(data, R_anim)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(1 / 60)

                R_curr = R_anim
                time.sleep(0.5)  # Brief pause between individual steps

            time.sleep(1.5)  # Hold final orientation before restarting loop


if __name__ == "__main__":
    main()
