# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
import torch
import json
import trimesh
import skimage.io as scio
import pickle
import os
from torch.utils.data import Dataset

import src.verefine.config as config


class VerefineDataset(Dataset):

    def __init__(self, args):
        super(VerefineDataset, self).__init__()
        self.dataset = args.dataset
        self.base_path = f"{config.PATH_BOP19}{self.dataset}/"
        self.base_path_vf = f"{config.PATH_VEREFINE}{self.dataset}/"

        with open(f"{self.base_path_vf}verefine.json", 'r') as file:
            meta_vf = json.load(file)

        # set verefine config for dataset
        config.MODE = args.mode

        if self.dataset == 'lm':
            self.baseline = args.lm_baseline
            config.HYPOTHESES_PER_OBJECT = 5 if self.baseline == 'df' else 3
            config.REFINEMENT_ITERATIONS = 2 if self.baseline == 'df' else 5
            # with ppf+icp: 5 iterations per hypothesis, 10 iterations per ICP call = 50/150 iterations per object
            config.SIMULATION_STEPS = 3
            config.C = 0.1
        elif self.dataset == 'xapc':
            self.baseline = 'pcs'
            config.HYPOTHESES_PER_OBJECT = 25
            config.REFINEMENT_ITERATIONS = 2
            # 2 iterations per hypothesis, 25 hypotheses per object, 3 objects per scene = 150 iterations in total
            config.SIMULATION_STEPS = 60
            config.C = 0.1 if config.MODE < 5 else 0.2
            config.GAMMA = 0.99
        else:  # ycbv
            self.baseline = 'df'
            config.HYPOTHESES_PER_OBJECT = 5
            config.REFINEMENT_ITERATIONS = 2
            # 2 iterations per hypothesis, 1/5 hypotheses per object = 2/10 iterations per object
            config.SIMULATION_STEPS = 3
            config.C = 0.1
        if config.MODE < 3:
            config.HYPOTHESES_PER_OBJECT = 1  # use only first (i.e., highest confidence) hypothesis
        if args.refinement_iterations > 0:
            config.REFINEMENT_ITERATIONS = args.refinement_iterations  # override defaults

        # camera data
        self.width, self.height = meta_vf['im_size']
        self.camera_intrinsics = np.asarray(meta_vf['intrinsics']).reshape(3, 3)
        self.depth_scale = meta_vf['depth_scale']

        # object meta
        obj_meta = meta_vf['objects']
        self.num_objects = len(obj_meta)
        self.obj_ids = list(obj_meta.keys())
        self.obj_names = dict(zip(self.obj_ids, [obj_meta[obj_id]['name'] for obj_id in self.obj_ids]))
        self.obj_coms = [obj_meta[obj_id]['offset_center_mass'] for obj_id in self.obj_ids]
        self.obj_model_offset = [obj_meta[obj_id]['offset_bop'] for obj_id in self.obj_ids]\
            if self.dataset == 'ycbv' else [None] * self.num_objects
        self.obj_scale = meta_vf['object_scale']

        # meshes for simulation and rendering
        self.collider_paths = [f"{self.base_path_vf}colliders/obj_{int(obj_id):06}.obj" for obj_id in self.obj_ids]
        scale = np.diag([self.obj_scale]*3 + [1.0])
        self.meshes = [trimesh.load(f"{self.base_path}models_eval/obj_{int(obj_id):06}.ply").apply_transform(scale)
                       for obj_id in self.obj_ids]

        # point clouds for (Tr)ICP
        if self.dataset in ['xapc', 'lm']:
            self.pcds = [np.asarray(trimesh.load(f"{self.base_path_vf}clouds/obj_{int(obj_id):06}.ply").vertices)
                         for obj_id in self.obj_ids]

        # test targets
        self.test_targets = meta_vf['targets_path']
        with open(self.base_path + self.test_targets, 'r') as file:
            targets = json.load(file)
        self.targets = np.asarray([[target['scene_id'], target['im_id']] for target in targets])
        # still has frames multiple times per instance in frame -> only unique (scene, frame) tuples
        unique_scenes = np.unique(self.targets[:, 0])
        unique_frames_per_scene = [np.unique(self.targets[:, 1][self.targets[:, 0] == scene])
                                   for scene in unique_scenes]
        n_frames_per_scene = [frames.shape[0] for frames in unique_frames_per_scene]
        self.targets = np.hstack((np.repeat(unique_scenes, n_frames_per_scene).reshape(-1, 1),
                                  np.hstack(unique_frames_per_scene).reshape(-1, 1)))

        # DenseFusion specific
        if 'densefusion' in meta_vf:
            self.df_num_points = meta_vf['densefusion']['num_points']
            self.df_model_path = f"{config.PATH_VEREFINE}{meta_vf['densefusion']['path']}"

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        scene, frame = self.targets[item]

        # 1) load pre-computed infos (hypotheses pool, extrinsics, embedding for DF-R)
        if not os.path.exists(f"{self.base_path_vf}test/{scene:04}/{frame:06}_{self.baseline}.pkl"):
            return None, None
        with open(f"{self.base_path_vf}test/{scene:04}/{frame:06}_{self.baseline}.pkl", 'rb') as file:
            data = pickle.load(file)
        assert data['scene'] == scene and data['frame'] == frame

        # 2) load observation
        if self.dataset == 'xapc':
            # with table removed as in PHYSIM-MCTS (Mitash et al.)
            depth = scio.imread(f"{self.base_path_vf}test/{scene:04}/scene_depth/{frame:06}.png").astype(np.float32)
        else:
            if not os.path.exists(f"{self.base_path_vf}test/{scene:04}/{frame:06}_label.png"):
                return None, None  # -> no object detected
            labels = scio.imread(f"{self.base_path_vf}test/{scene:04}/{frame:06}_label.png")
            mask = labels > 0
            if mask.sum() == 0:
                return None, None

            depth = scio.imread(f"{self.base_path}test/{scene:06}/depth/{frame:06}.png").astype(np.float32)
            depth[mask == 0] = 0  # only depth of observed objects
        depth /= self.depth_scale  # to meters

        normals = scio.imread(f"{self.base_path_vf}test/{scene:04}/{frame:06}_normal.tiff").astype(np.float32)

        camera_extrinsics = np.asarray(data['extrinsics'])
        if self.dataset == 'xapc':
            camera_extrinsics = np.linalg.inv(camera_extrinsics)

        # optional: dependencies
        if self.dataset == 'lm':
            dependencies = [[0]]  # single object, no dependencies
        elif self.dataset == 'xapc':
            dependencies = data['dependencies']  # as in PHYSIM-MCTS (Mitash et al.)
        else:
            dependencies = None  # let verefine compute dependencies

        observation = {
            'scene': scene,
            'frame': frame,
            'depth': depth,
            'normal': normals,
            'dependencies': dependencies,
            'extrinsics': camera_extrinsics,
            'intrinsics': self.camera_intrinsics
        }

        # 3) prepare hypotheses pool
        hypotheses = []
        for oi, object_hypotheses in enumerate(data['frame_hypotheses']):

            if self.baseline == 'df':
                emb = torch.from_numpy(object_hypotheses['df_embedding']).cuda()
                cloud = torch.from_numpy(object_hypotheses['df_samples']).cuda()

            hypotheses.append([])
            for pose, c in zip(object_hypotheses['hypotheses'][:config.HYPOTHESES_PER_OBJECT],
                               object_hypotheses['confidences'][:config.HYPOTHESES_PER_OBJECT]):
                hypothesis = {
                    'obj_id': str(object_hypotheses['obj_id']),
                    'pose': pose,
                    'confidence': c
                }

                if self.dataset == 'ycbv':
                    hypothesis['obj_mask'] = labels == int(hypothesis['obj_id'])

                if self.baseline == 'df':
                    hypothesis['emb'] = emb
                    hypothesis['cloud_obs'] = cloud
                else:  # icp and tricp
                    hypothesis['cloud_obs'] = object_hypotheses['samples']

                hypotheses[-1].append(hypothesis)

        return observation, hypotheses
