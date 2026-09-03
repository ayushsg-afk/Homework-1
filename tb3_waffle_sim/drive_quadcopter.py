import os
import time
import sys
import tty
import termios
import select
import numpy as np
import mujoco
import mujoco.viewer

# Load Quadcopter Model
model_path = os.path.expanduser('~/tb3_waffle_sim/quadcopter.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

body_id = model.body('base_link').id

# Drone physical parameters
mass = 0.64
g = 9.81
hover_thrust_base = mass * g

# Target States (Position & Yaw Heading)
target_x = 0.0
target_y = 0.0
target_z = 0.8        # Hover height (m)
target_yaw = 0.0      # Heading angle in degrees

def get_key_nonblocking():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
        if rlist:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

print("=========================================================")
print("  QUADCOPTER 3D: UPRIGHT LOCKED WITH FULL YAW ROTATION   ")
print("=========================================================")
print("Controls:")
print("  [ I ] : Move Forward     |  [ K ] : Move Backward")
print("  [ J ] : Move Left        |  [ L ] : Move Right")
print("  [ A ] : Yaw/Rotate Left  |  [ D ] : Yaw/Rotate Right")
print("  [ U ] : Climb (+Z)       |  [ O ] : Descend (-Z)")
print("  [ M ] : Reset Position & Orientation")
print("=========================================================\n")

step_counter = 0
last_printed_yaw = 0.0
last_printed_pos = np.zeros(3)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    while viewer.is_running():
        key = get_key_nonblocking()
        if key:
            key = key.lower()
            # Yaw Heading Controls
            if key == 'a':
                target_yaw += 5.0    # Rotate counter-clockwise (Left)
            elif key == 'd':
                target_yaw -= 5.0    # Rotate clockwise (Right)
            
            # Local Directional Step Movements (Relative to Current Heading)
            yaw_rad_current = np.radians(target_yaw)
            if key == 'i':
                target_x += 0.05 * np.cos(yaw_rad_current)
                target_y += 0.05 * np.sin(yaw_rad_current)
            elif key == 'k':
                target_x -= 0.05 * np.cos(yaw_rad_current)
                target_y -= 0.05 * np.sin(yaw_rad_current)
            elif key == 'j':
                target_x -= 0.05 * np.sin(yaw_rad_current)
                target_y += 0.05 * np.cos(yaw_rad_current)
            elif key == 'l':
                target_x += 0.05 * np.sin(yaw_rad_current)
                target_y -= 0.05 * np.cos(yaw_rad_current)
            elif key == 'u':
                target_z += 0.05
            elif key == 'o':
                target_z = max(0.1, target_z - 0.05)
            elif key == 'm' or key == ' ':
                target_x, target_y, target_z = 0.0, 0.0, 0.8
                target_yaw = 0.0

        pos = data.xpos[body_id]
        vel = data.qvel[:3]

        # Convert Target Yaw Heading into Quaternion [w, x, y, z] (Roll=0, Pitch=0, Yaw=theta)
        half_yaw = np.radians(target_yaw) / 2.0
        qw = np.cos(half_yaw)
        qz = np.sin(half_yaw)
        
        # Hard-lock orientation to prevent pitching/rolling flips while maintaining true Yaw rotation
        data.qpos[3:7] = np.array([qw, 0.0, 0.0, qz])
        data.qvel[3:6] = 0.0  # Zero out angular rate instabilities

        # Position PID Controller
        kp_pos = 10.0
        kd_pos = 4.0

        error_x = target_x - pos[0]
        error_y = target_y - pos[1]
        error_z = target_z - pos[2]

        force_x = kp_pos * error_x - kd_pos * vel[0]
        force_y = kp_pos * error_y - kd_pos * vel[1]
        force_z = hover_thrust_base + (kp_pos * 1.5 * error_z) - (kd_pos * 1.5 * vel[2])

        # Apply translational forces
        data.xfrc_applied[body_id, :3] = np.array([
            np.clip(force_x, -3.0, 3.0),
            np.clip(force_y, -3.0, 3.0),
            np.clip(force_z, 0.0, 15.0)
        ])

        mujoco.mj_step(model, data)

        # Transformation Outputs
        R_current = data.xmat[body_id].reshape(3, 3)
        yaw_rad = np.arctan2(R_current[1, 0], R_current[0, 0])
        yaw_deg = np.degrees(yaw_rad)
        R_rounded = np.round(R_current, 3)

        # Log matrix output continuous step logging when yaw or position changes
        if (np.linalg.norm(pos - last_printed_pos) > 0.01) or (abs(yaw_deg - last_printed_yaw) >= 1.0):
            step_counter += 1
            print(f"--- [Step {step_counter}] Drone Transformation Data ---")
            print(f"Target Pos (X,Y,Z): X={target_x:5.2f}m | Y={target_y:5.2f}m | Z={target_z:5.2f}m")
            print(f"Actual Pos (X,Y,Z): X={pos[0]:5.2f}m | Y={pos[1]:5.2f}m | Z={pos[2]:5.2f}m")
            print(f"Heading Angle(Yaw): {yaw_deg:6.1f}°")
            print("Rotation Matrix R_body^world [x_b | y_b | z_b]:")
            print(f"  |  {R_rounded[0,0]:6.3f}  {R_rounded[0,1]:6.3f}  {R_rounded[0,2]:6.3f}  |")
            print(f"  |  {R_rounded[1,0]:6.3f}  {R_rounded[1,1]:6.3f}  {R_rounded[1,2]:6.3f}  |")
            print(f"  |  {R_rounded[2,0]:6.3f}  {R_rounded[2,1]:6.3f}  {R_rounded[2,2]:6.3f}  |")
            print("-" * 55 + "\n")
            
            last_printed_pos = pos.copy()
            last_printed_yaw = yaw_deg

        viewer.sync()
        time.sleep(0.005)
