# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
import os
basepath = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(basepath, "cpp", "build"))
import icp


def refine(pose, cloud_observed, cloud_model, p_distance=0.1, iterations=1):
    obj_T = pose.copy()

    cloud_estimated = np.dot(cloud_model, obj_T[:3, :3].T) + obj_T[:3, 3].T

    if cloud_estimated.shape[0] == 0 or cloud_observed.shape[0] == 0:
        return obj_T

    T = icp.icp(cloud_observed, cloud_estimated, iterations, p_distance)
    obj_T = T @ obj_T

    return obj_T
