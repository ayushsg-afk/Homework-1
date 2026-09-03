import os
import time
import sys
import tty
import termios
import select
import numpy as np
import mujoco
import mujoco.viewer

# Load model
model_path = os.path.expanduser('~/tb3_waffle_sim/waffle_pi.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

left_actuator_id = model.actuator('left_motor').id
right_actuator_id = model.actuator('right_motor').id
body_id = model.body('base_link').id

linear_vel = 0.0
angular_vel = 0.0

def get_key_nonblocking():
    """Reads single keypress without requiring Enter."""
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
print("      STEP 3: BODY FRAME KINEMATICS MONITOR             ")
print("=========================================================")
print("Controls:")
print("  [ I ] : Drive Forward    |  [ J ] : Turn Left")
print("  [ K ] : Drive Backward   |  [ L ] : Turn Right")
print("  [ M ] : Emergency Stop   |  [ R ] : Reset Position")
print("=========================================================\n")

step_counter = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Render ONLY site frames (the center body axis on the robot)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    
    last_printed_R = np.round(data.xmat[body_id].reshape(3, 3), 2)

    while viewer.is_running():
        key = get_key_nonblocking()
        if key:
            key = key.lower()
            if key == 'i':
                linear_vel = min(linear_vel + 0.1, 1.0)
            elif key == 'k':
                linear_vel = max(linear_vel - 0.1, -1.0)
            elif key == 'j':
                angular_vel = min(angular_vel + 0.2, 2.0)
            elif key == 'l':
                angular_vel = max(angular_vel - 0.2, -2.0)
            elif key == 'm' or key == ' ':
                linear_vel = 0.0
                angular_vel = 0.0
            elif key == 'r':
                linear_vel = 0.0
                angular_vel = 0.0
                mujoco.mj_resetData(model, data)
                print("\n[RESET] Robot frame returned to world origin (0,0,0).\n")

        # Differential drive geometry
        wheel_separation = 0.22
        wheel_radius = 0.027
        v_left = (linear_vel - (angular_vel * wheel_separation / 2.0)) / wheel_radius
        v_right = (linear_vel + (angular_vel * wheel_separation / 2.0)) / wheel_radius

        data.ctrl[left_actuator_id] = v_left
        data.ctrl[right_actuator_id] = v_right

        mujoco.mj_step(model, data)

        # 1. Body Frame Origin (Position X, Y, Z in world)
        pos = data.xpos[body_id]

        # 2. Body Frame Orientation (3x3 Rotation Matrix)
        R_current = np.round(data.xmat[body_id].reshape(3, 3), 3)

        # 3. Heading (Yaw Angle theta around World Z-axis)
        yaw_rad = np.arctan2(R_current[1, 0], R_current[0, 0])
        yaw_deg = np.degrees(yaw_rad)

        # Print transformation matrix when orientation updates
        if not np.array_equal(np.round(R_current, 2), last_printed_R):
            step_counter += 1
            print(f"--- [Step {step_counter}] Body Frame Transformation ---")
            print(f"Origin (World Pos) : X = {pos[0]:6.3f} m | Y = {pos[1]:6.3f} m | Z = {pos[2]:6.3f} m")
            print(f"Heading Angle (Yaw): {yaw_deg:6.1f}°")
            print("Rotation Matrix R_body^world [x_b | y_b | z_b]:")
            print(f"  |  {R_current[0,0]:6.3f}  {R_current[0,1]:6.3f}  {R_current[0,2]:6.3f}  |  (World X component)")
            print(f"  |  {R_current[1,0]:6.3f}  {R_current[1,1]:6.3f}  {R_current[1,2]:6.3f}  |  (World Y component)")
            print(f"  |  {R_current[2,0]:6.3f}  {R_current[2,1]:6.3f}  {R_current[2,2]:6.3f}  |  (World Z component)")
            print("-" * 55 + "\n")
            
            last_printed_R = np.round(R_current, 2)

        viewer.sync()
        time.sleep(0.005)
