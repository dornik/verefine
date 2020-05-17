# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append("./3rdparty/DenseFusion")
sys.path.append("./3rdparty/DenseFusion/lib")
from lib.network import PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix


class DenseFusionRefine:

    def __init__(self, dataset):
        self.num_points = dataset.df_num_points
        self.obj_list = dataset.obj_ids

        self.refiner = PoseRefineNet(num_points=self.num_points, num_obj=len(self.obj_list))
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load(dataset.df_model_path))
        self.refiner.eval()

    def refine(self, obj_id, pose, emb, cloud, iterations=1):
        class_id = self.obj_list.index(obj_id)
        index = torch.LongTensor([class_id])
        index = Variable(index).cuda()

        pose_ref = pose.copy()
        for _ in range(iterations):
            # transform cloud according to pose estimate
            R = Variable(torch.from_numpy(pose_ref[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            ts = Variable(torch.from_numpy(pose_ref[:3, 3].astype(np.float32))).cuda().view(1, 3)\
                .repeat(self.num_points, 1).contiguous().view(1, self.num_points, 3)
            new_cloud = torch.bmm((cloud - ts), R).contiguous()

            # predict delta pose
            pred_q, pred_t = self.refiner(new_cloud, emb, index)
            pred_q = pred_q.view(1, 1, -1)
            pred_q = pred_q / (torch.norm(pred_q, dim=2).view(1, 1, 1))
            pred_q = pred_q.view(-1).cpu().data.numpy()
            pred_t = pred_t.view(-1).cpu().data.numpy()

            # apply delta to get new pose estimate
            pose_delta = quaternion_matrix(pred_q)
            pose_delta[:3, 3] = pred_t
            pose_ref = pose_ref @ pose_delta

        return pose_ref
