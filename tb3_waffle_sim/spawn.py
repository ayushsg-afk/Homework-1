import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. Load the Waffle Pi model
model_path = os.path.expanduser('~/tb3_waffle_sim/waffle_pi.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

body_id = model.body('base_link').id

# 2. Launch 3D Simulation Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Display robot body frame axes (Red=X Front, Green=Y Left, Blue=Z Up)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    
    print("TurtleBot3 Waffle Pi spawned successfully!")
    print("Close the 3D window to stop.")

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
