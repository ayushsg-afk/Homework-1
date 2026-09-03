"""
02_verify_skew_properties.py -- Task 2: Verify skew-symmetric identities
under time-varying angular velocity in MuJoCo.
"""

import os
import numpy as np
import mujoco

from utils import hat, get_body_orientation, is_close_to_identity

# Absolute path resolution ensures XML loads regardless of execution directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "asymmetric_body.xml")

N_CHECKS_PER_STEP = 5     # random (v, w, omega) triples per logged step
N_LOGGED_STEPS = 5        # simulated time points to check
STEPS_BETWEEN_LOGS = 200  # sim steps between logged checks


def omega_func(t):
    """Time-varying angular velocity vector omega(t) in rad/s."""
    return np.array([np.sin(t), np.cos(t), 0.5 * t])


def check_identities(R, rng):
    """
    Compute maximum residuals for:
        Identity 1: R @ np.cross(v, w)  vs.  np.cross(R @ v, R @ w)
        Identity 2: R @ hat(omega) @ R.T  vs.  hat(R @ omega)
    """
    max_residual_cross = 0.0
    max_residual_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        # Identity 1: R(v x w) == (Rv) x (Rw)
        lhs1 = R @ np.cross(v, w)
        rhs1 = np.cross(R @ v, R @ w)
        res1 = np.linalg.norm(lhs1 - rhs1)
        max_residual_cross = max(max_residual_cross, res1)

        # Identity 2: R hat(omega) R^T == hat(R omega)
        lhs2 = R @ hat(omega) @ R.T
        rhs2 = hat(R @ omega)
        res2 = np.linalg.norm(lhs2 - rhs2, ord='fro')
        max_residual_skew = max(max_residual_skew, res2)

    return max_residual_cross, max_residual_skew


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=0)

    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>28} {'max resid: R[w^]R^T=[Rw]^':>26}")
    print("-" * 72)

    for log_i in range(N_LOGGED_STEPS):
        for _ in range(STEPS_BETWEEN_LOGS):
            # Apply time-varying angular velocity to freejoint (qvel 3:6)
            data.qvel[3:6] = omega_func(data.time)
            mujoco.mj_step(model, data)

        R = get_body_orientation(data)
        assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"

        resid_cross, resid_skew = check_identities(R, rng)
        print(f"{log_i:5d} {data.time:8.3f} {resid_cross:28.3e} {resid_skew:26.3e}")


if __name__ == "__main__":
    main()
