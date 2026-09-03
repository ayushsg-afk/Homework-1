# TurtleBot 3 Waffle Pi MuJoCo Simulation

A lightweight differential drive robot simulation built using **MuJoCo** and **Python**.

## Project Features
* **Custom Physics Model (`waffle_pi.xml`)**: Dual-caster baseline balance for smooth acceleration and sharp turns.
* **Keyboard Teleop (`drive.py`)**: Real-time non-blocking velocity controls using `I`, `J`, `K`, `L`, and `M` keys.
* **Transformation Tracking**: Live updates of the robot's **3D Rotation Matrix ($R_b^w$)** and **Yaw Angle ($\theta$)**.
* **Visual Frame Axis**: Integrated body frame site visualization rendered directly at the center of mass.

## Keyboard Controls
| Key | Action |
| :---: | :--- |
| **I** | Drive Forward |
| **K** | Drive Backward |
| **J** | Turn Left (Counter-clockwise) |
| **L** | Turn Right (Clockwise) |
| **M / Space** | Emergency Stop |
| **R** | Reset Position to Origin |

## Setup & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/ayushsg-afk/turtlebot3-mujoco-sim.git](https://github.com/ayushsg-afk/turtlebot3-mujoco-sim.git)
   cd turtlebot3-mujoco-sim

