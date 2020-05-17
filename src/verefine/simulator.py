# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import pybullet
import pybullet_data
import numpy as np
from scipy.spatial.transform.rotation import Rotation

import src.verefine.config as config


class Simulator:

    def __init__(self, dataset):
        self.dataset = dataset
        self.objects_to_use = dataset.obj_ids

        self.plane = None  # base plane
        self.models = dict()

        self.T_w2c = None
        self.T_c2w = None

        self._initialize_bullet()
        self._initialize_dataset()

    def initialize_frame(self, extrinsics):
        self.T_w2c = extrinsics

        self.T_c2w = np.eye(4)
        R, t = extrinsics[:3, :3], extrinsics[:3, 3]
        self.T_c2w[0:3, 0:3] = R.T
        self.T_c2w[0:3, 3] = -R.T @ t

    def reset_objects(self):
        # activate collisions - or move out of sim area and deactivate
        for obj_str in self.models.keys():
            if obj_str in self.objects_to_use:
                pybullet.changeDynamics(self.models[obj_str], -1, mass=1.0)
                pybullet.setCollisionFilterGroupMask(self.models[obj_str], -1, 1, 1)
            else:
                pybullet.resetBasePositionAndOrientation(self.models[obj_str], [0, 0, -5],
                                                         [0, 0, 0, 1],
                                                         self.world)
                pybullet.changeDynamics(self.models[obj_str], -1, mass=0)  # -> static
                pybullet.setCollisionFilterGroupMask(self.models[obj_str], -1, 0, 0)

    def initialize_scene(self, hypotheses):
        for hypothesis in hypotheses:
            if hypothesis['obj_id'] not in self.objects_to_use:
                continue
            T_obj = hypothesis['pose'].copy()

            position, orientation = self._cam_to_bullet(T_obj, hypothesis['obj_id'])
            pybullet.resetBasePositionAndOrientation(self.models[hypothesis['obj_id']], position, orientation, self.world)

    def simulate(self, obj_id):
        for i in range(config.SIMULATION_STEPS):
            pybullet.stepSimulation()

        # read-back transformation after simulation
        position, orientation = pybullet.getBasePositionAndOrientation(self.models[obj_id], self.world)
        T_phys = self._bullet_to_cam(list(position), list(orientation), obj_id)

        return T_phys

    def _initialize_bullet(self):
        """
        Initialize physics world in headless mode or with GUI (debug).
        """

        GUI = False
        if GUI:
            self.world = pybullet.connect(pybullet.GUI)
            pybullet.setRealTimeSimulation(0)
        else:
            self.world = pybullet.connect(pybullet.DIRECT)  # non-graphical client

        # set-up simulation
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(config.TIME_STEP)
        pybullet.setPhysicsEngineParameter(fixedTimeStep=config.TIME_STEP, numSolverIterations=config.SOLVER_ITERATIONS,
                                           numSubSteps=config.SOLVER_SUB_STEPS)

        # add ground plane
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pybullet.loadURDF("plane.urdf", globalScaling=1/10.0)
        pybullet.setCollisionFilterGroupMask(self.plane, -1, 1, 1)
        pybullet.resetBasePositionAndOrientation(self.plane, [0, 0, 0], [0, 0, 0, 1], self.world)

    def deinitialize(self):
        pybullet.disconnect()

    def _initialize_dataset(self):
        for i, (obj_id, model_path, obj_com) in enumerate(zip(self.dataset.obj_ids,
                                                              self.dataset.collider_paths,
                                                              self.dataset.obj_coms)):
            try:
                collision_shape = pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH,
                                                                fileName=model_path,
                                                                meshScale=[self.dataset.obj_scale]*3)
                model = pybullet.createMultiBody(baseMass=1.0,
                                                 baseVisualShapeIndex=-1,
                                                 baseCollisionShapeIndex=collision_shape,
                                                 basePosition=[i * 0.5, 0, 0],
                                                 baseOrientation=[1, 0, 0, 0],
                                                 baseInertialFramePosition=obj_com)
                self.models[obj_id] = model

            except Exception as _:
                self.models.clear()
                raise ValueError("Could not load models for simulation.")

    def _cam_to_world(self, in_cam):
        return self.T_c2w @ in_cam

    def _world_to_cam(self, in_world):
        return self.T_w2c @ in_world

    def _world_to_bullet(self, in_world, id):
        R = in_world[0:3, 0:3]
        orientation = Rotation.from_dcm(R).as_quat()

        # position is offset by center of mass (origin of collider in physics world)
        offset = R @ np.array(self.dataset.obj_coms[self.dataset.obj_ids.index(id)])
        position = in_world[0:3, 3] + offset

        return position, orientation

    def _bullet_to_world(self, position, orientation, id):
        in_world = np.eye(4)
        in_world[0:3, 0:3] = Rotation.from_quat(orientation).as_dcm()
        # position was offset by center of mass (origin of collider in physics world)
        offset = in_world[0:3, 0:3] @ np.array(self.dataset.obj_coms[self.dataset.obj_ids.index(id)])
        in_world[0:3, 3] = position - offset
        return in_world

    def _cam_to_bullet(self, in_cam, id):
        return self._world_to_bullet(self._cam_to_world(in_cam), id)

    def _bullet_to_cam(self, position, orientation, id):
        return self._world_to_cam(self._bullet_to_world(position, orientation, id))
