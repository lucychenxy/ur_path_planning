import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import swift
from spatialmath import SE3

ur5 = rtb.models.UR5()

# q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# dq = np.array([0.2, 0.1, 0.3, 0.4, 0.2, 0.1])
q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
dq = np.array([2.0, 1.5, 3.0, 2.5, 2.0, 1.5])

robot = rtb.models.DH.UR5()

for payload_mass in range(1, 5):
    print(f"\n--- Payload: {payload_mass} kg ---")

    robot.payload(payload_mass, [0, 0, 0])

    M = robot.inertia(q)
    print("Inertia Matrix M(q):\n", M)

    C = robot.coriolis(q, dq)
    print("Coriolis and Centrifugal Matrix C(q, dq):\n", C)

    G = robot.gravload(q)
    print("Gravity Vector G(q):\n", G)
