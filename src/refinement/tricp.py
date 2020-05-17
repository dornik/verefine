# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
from scipy.spatial import cKDTree as KDTree
import os
basepath = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(basepath, "cpp", "build"))
import icp


def refine(pose, cloud_observed, cloud_model, cloud_explained, trim=0.99):
    obj_T = pose.copy()

    if cloud_explained is not None:
        # as in PHYSIM-MCTS (Mitash et al.)
        exp_tree = KDTree(cloud_explained)
        indices = exp_tree.query_ball_point(cloud_observed, r=0.008)
        unexplained = [len(ind) == 0 for ind in indices]
        cloud_observed_unexplained = cloud_observed[unexplained]
    else:
        cloud_observed_unexplained = cloud_observed.copy()  #

    cloud_estimated = np.dot(cloud_model, obj_T[:3, :3].T) + obj_T[:3, 3].T

    if cloud_estimated.shape[0] == 0 or cloud_observed_unexplained.shape[0] == 0:
        return obj_T

    T = icp.tricp(cloud_observed_unexplained, cloud_estimated, trim)
    obj_T = T @ obj_T

    return obj_T
