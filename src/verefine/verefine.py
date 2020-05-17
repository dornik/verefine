# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.sparse import csgraph
from functools import cmp_to_key

import src.verefine.config as config


class Verefine:

    def __init__(self, dataset, refiner, simulator, renderer):
        self.dataset = dataset
        self.refiner = refiner
        self.simulator = simulator
        self.renderer = renderer
        self.mode = config.MODE
        self.observation = None

    def refine(self, observation, hypotheses):
        self.observation = observation
        mode = self.mode
        simulator = self.simulator
        renderer = self.renderer

        # optional: initialize frame in simulator and renderer
        if simulator is not None:
            simulator.initialize_frame(observation['extrinsics'])
        if renderer is not None:
            renderer.set_observation(observation['depth'], observation['normal'])

        # optional: compute clustering and scene-level dependencies
        if mode > 3:
            if observation['dependencies'] is None:
                dependencies = self._compute_dependencies(observation, hypotheses)
            else:
                dependencies = observation['dependencies']
        else:
            # one cluster for whole scene
            dependencies = [list(range(len(hypotheses)))]

        # run methods per cluster
        final_hypotheses = []
        for cluster in dependencies:
            if mode > 3:
                cluster_hypotheses = [hypotheses[ci] for ci in cluster]
            else:
                cluster_hypotheses = hypotheses

            if mode < 5:
                final_hypotheses += self._refine_cluster(cluster_hypotheses)
            else:
                # single object in cluster? run RIR
                if len(cluster) == 1:
                    self.mode = 3
                    final_hypotheses += self._refine_cluster(cluster_hypotheses)
                    self.mode = 5
                # multiple objects? run VFd
                else:
                    final_hypotheses += self._refine_vfd(cluster_hypotheses)
        return final_hypotheses

    def _refine_object(self, hypotheses, iterations=1, fixed=[], statistics=None):
        mode = self.mode
        renderer = self.renderer
        refiner = self.refiner
        dataset = self.dataset
        observation = self.observation
        if mode > 1:
            model_id = renderer.dataset.obj_ids.index(hypotheses[0]['obj_id'])

        # 1) optional: get previous statistics
        if mode == 5:
            SIR_hypotheses, SIR_fits, SIR_plays, DUCB_rewards, DUCB_plays = statistics
        elif mode > 2:
            SIR_hypotheses, SIR_fits, SIR_plays, RIR_rewards = statistics
        elif mode > 1:
            SIR_hypotheses, SIR_fits, SIR_plays = statistics

        # 2) run refinement
        for _ in range(iterations):
            # a) get current hypothesis
            if mode > 2:
                # based on ucb(RIR stats)
                if mode == 5:
                    exploitation = DUCB_rewards / DUCB_plays
                    exploration = np.sqrt(np.log(DUCB_plays.sum()) / DUCB_plays)
                else:
                    exploitation = RIR_rewards
                    exploration = np.sqrt(np.log(SIR_plays.sum()) / SIR_plays)
                ucb_scores = exploitation + config.C * exploration

                hi = np.argmax(ucb_scores)
                hypothesis = hypotheses[hi]
                hypothesis['pose'] = SIR_hypotheses[
                    hi, int(SIR_plays[hi] - 1)]  # -> most recent refinement of this hypothesis
            else:
                # there is only one for modes 0-2
                hi = 0
                hypothesis = hypotheses[hi]

                if mode == 2:
                    # get most recent refinement
                    hypothesis['pose'] = SIR_hypotheses[hi, int(SIR_plays[hi] - 1)]
            T_cur = hypothesis['pose'].copy()

            # b) optional: simulate
            if mode > 0:
                T_sim = self._simulate_hypothesis(hypothesis, fixed)
                T_cur[:3, :3] = T_sim[:3, :3]  # update only R -> [R_sim; t_cur]

            # c) refine using baseline refinement method
            if dataset.baseline == 'df':
                T_ref = refiner.refine(hypothesis['obj_id'], T_cur, hypothesis['emb'], hypothesis['cloud_obs'])
            elif dataset.baseline == 'ppf':
                oi = dataset.obj_ids.index(str(hypothesis['obj_id']))
                T_ref = refiner.refine(T_cur, hypothesis['cloud_obs'], dataset.pcds[oi],
                                       p_distance=config.ICP_P_DISTANCE, iterations=config.ICP_ITERATIONS)
            elif dataset.baseline == 'pcs':
                cloud_explained = None
                for fixed_h in fixed:
                    fixed_oi = dataset.obj_ids.index(fixed_h['obj_id'])
                    fixed_pose = fixed_h['pose']
                    fixed_pts = np.dot(dataset.pcds[fixed_oi], fixed_pose[:3, :3].T) + fixed_pose[:3, 3].T
                    if cloud_explained is None:
                        cloud_explained = fixed_pts
                    else:
                        cloud_explained = np.concatenate((cloud_explained, fixed_pts), axis=0)
                oi = dataset.obj_ids.index(hypothesis['obj_id'])
                T_ref = refiner.refine(T_cur, hypothesis['cloud_obs'], dataset.pcds[oi], cloud_explained,
                                       trim=config.TRICP_TRIM)

            # d) update estimate
            if mode > 1:
                # continue with better one of T_sim, T_ref
                score_sim = renderer.compute_score([model_id], [T_sim],
                                                   observation['extrinsics'], observation['intrinsics'])
                score_ref = renderer.compute_score([model_id], [T_ref],
                                                   observation['extrinsics'], observation['intrinsics'])

                hypothesis['pose'] = T_ref if score_ref >= score_sim else T_sim
                score = max(score_sim, score_ref)

                # update SIR stats
                ri = int(SIR_plays[hi])
                SIR_hypotheses[hi, ri] = hypothesis['pose'].copy()
                SIR_fits[hi, ri] = score
                if 2 < mode < 5:
                    # optional: update RIR stats
                    RIR_rewards[hi] = (RIR_rewards[hi] * float(SIR_plays[hi]) + score) / (SIR_plays[hi] + 1.0)
                SIR_plays[hi] += 1
            else:
                # continue with refinement result T_ref
                hypothesis['pose'] = T_ref
        return hi, hypothesis['pose'].copy()

    def _refine_cluster(self, cluster_hypotheses):
        mode = self.mode
        simulator = self.simulator
        renderer = self.renderer
        observation = self.observation

        fixed = []
        cluster_results = []
        for oi, hypotheses in enumerate(cluster_hypotheses):

            iterations = config.REFINEMENT_ITERATIONS * config.HYPOTHESES_PER_OBJECT  # per object in cluster

            # 1) optional: prepare sim
            if mode > 0:
                simulator.objects_to_use = [hypotheses[0]['obj_id']] + [fix['obj_id'] for fix in fixed]
                simulator.reset_objects()

            # 2) optional: prepare SIR stats
            if mode > 1:
                model_id = renderer.dataset.obj_ids.index(hypotheses[0]['obj_id'])

                SIR_hypotheses = np.zeros((config.HYPOTHESES_PER_OBJECT, iterations + config.HYPOTHESES_PER_OBJECT,
                                           4, 4))
                SIR_fits = np.zeros((config.HYPOTHESES_PER_OBJECT, iterations + config.HYPOTHESES_PER_OBJECT))
                SIR_plays = np.zeros((config.HYPOTHESES_PER_OBJECT))

                for hi, hypothesis in enumerate(hypotheses):
                    T_init = hypothesis['pose'].copy()
                    f_init = renderer.compute_score([model_id], [T_init],
                                                    observation['extrinsics'], observation['intrinsics'])

                    SIR_hypotheses[hi, 0] = T_init
                    SIR_fits[hi, 0] = f_init if not np.isnan(f_init) else 0
                    SIR_plays[hi] = 1

            # 3) optional: prepare RIR stats
            if mode > 2:
                RIR_rewards = SIR_fits[:, 0].copy()

            # 4) refine
            if mode > 2:
                statistics = (SIR_hypotheses, SIR_fits, SIR_plays, RIR_rewards)
            elif mode > 1:
                statistics = (SIR_hypotheses, SIR_fits, SIR_plays)
            else:
                statistics = None
            self._refine_object(hypotheses, iterations, fixed, statistics)

            # 5) optional: select best overall
            if mode > 1:
                best_hi, best_ri = np.unravel_index(SIR_fits.argmax(), SIR_fits.shape)
                hypothesis = hypotheses[best_hi]
                hypothesis['pose'] = SIR_hypotheses[best_hi, best_ri].copy()
                hypothesis['confidence'] = SIR_fits.max()
            else:
                hypothesis = hypotheses[0]
            cluster_results.append(hypothesis)

            # for VFb, add refined object to simulation environment
            if mode > 3 and len(cluster_hypotheses) > 1:
                fixed.append(hypothesis)
        return cluster_results

    def _refine_vfd(self, cluster_hypotheses):
        simulator = self.simulator
        renderer = self.renderer
        observation = self.observation

        num_objects = len(cluster_hypotheses)
        iterations = config.REFINEMENT_ITERATIONS * config.HYPOTHESES_PER_OBJECT  # per object in cluster

        # prepare statistics
        SIR_hypotheses = np.zeros((num_objects, config.HYPOTHESES_PER_OBJECT,
                                   iterations + config.HYPOTHESES_PER_OBJECT, 4, 4))
        SIR_fits = np.zeros((num_objects, config.HYPOTHESES_PER_OBJECT,
                             iterations + config.HYPOTHESES_PER_OBJECT))
        SIR_plays = np.zeros((num_objects, config.HYPOTHESES_PER_OBJECT))
        DUCB_rewards = np.zeros((num_objects, config.HYPOTHESES_PER_OBJECT))
        DUCB_plays = np.zeros((num_objects, config.HYPOTHESES_PER_OBJECT))

        cluster_results = []
        for iteration in range(iterations):

            # iteratively build a full scene
            fixed = []
            selected_hi = []
            selected_ids = []
            selected_poses = []
            for oi, hypotheses in enumerate(cluster_hypotheses):

                model_id = renderer.dataset.obj_ids.index(hypotheses[0]['obj_id'])

                # 1) prepare sim
                simulator.objects_to_use = [hypotheses[0]['obj_id']] + [fix['obj_id'] for fix in fixed]
                simulator.reset_objects()

                if iteration == 0:
                    # 2) init SIR stats
                    for hi, hypothesis in enumerate(hypotheses):
                        T_init = hypothesis['pose'].copy()
                        f_init = renderer.compute_score([model_id], [T_init],
                                                        observation['extrinsics'], observation['intrinsics'])

                        SIR_hypotheses[oi, hi, 0] = T_init
                        SIR_fits[oi, hi, 0] = f_init if not np.isnan(f_init) else 0
                        SIR_plays[oi, hi] = 1

                    # 3) init D-UCB stats
                    DUCB_rewards[oi] = SIR_fits[oi, :, 0].copy()
                    DUCB_plays[oi] = SIR_plays[oi, :].copy()

                # 4) refine
                statistics = (SIR_hypotheses[oi], SIR_fits[oi], SIR_plays[oi], DUCB_rewards[oi], DUCB_plays[oi])
                hi, pose = self._refine_object(hypotheses, 1, fixed, statistics)

                # 5) select best based on UCB (in refine)
                hypothesis = hypotheses[hi]
                hypothesis['pose'] = pose
                fixed.append(hypothesis)  # add object to simulation environment
                selected_hi.append(hi)
                selected_ids.append(model_id)
                selected_poses.append(pose.copy())

            # 6) compute score for full scene
            score = renderer.compute_score(selected_ids, selected_poses,
                                           observation['extrinsics'], observation['intrinsics'])
            # update statistics
            for oi, hi in enumerate(selected_hi):
                DUCB_rewards[oi] *= config.GAMMA
                DUCB_plays[oi] *= config.GAMMA
                DUCB_rewards[oi, hi] += score
                DUCB_plays[oi, hi] += 1

        # select best per object as final hypothesis
        for oi in range(num_objects):
            best_hi, best_ri = np.unravel_index(SIR_fits[oi].argmax(), SIR_fits[oi].shape)
            hypothesis = cluster_hypotheses[oi][best_hi]
            hypothesis['pose'] = SIR_hypotheses[oi, best_hi, best_ri].copy()
            hypothesis['confidence'] = SIR_fits[oi].max()

            cluster_results.append(hypothesis)
        return cluster_results

    def _simulate_hypothesis(self, obj_hypothesis, fixed):
        # initialize simulation with the object hypothesis (+ optionally: already fixed objects)
        scene = [obj_hypothesis] + fixed
        self.simulator.initialize_scene(scene)

        # simulation is progressed for [config.SIMULATION_STEPS] steps and resulting pose of object is read back
        T_sim = self.simulator.simulate(obj_hypothesis['obj_id'])
        return T_sim

    def _compute_dependencies(self, observation, frame_hypotheses):
        # 1) camera space clustering
        #    - if masks touch, the objects overlap in camera space -> occlusion
        #    - this also means they are in contact -> support
        depth = observation['depth']
        masks = [np.logical_and(hs[0]['obj_mask'], depth > 0) for hs in frame_hypotheses]
        fat_masks = [binary_dilation(mask) for mask in masks]
        overlap = np.dstack(fat_masks)
        intersect = overlap.sum(axis=2) > 1
        adjacency = overlap[intersect]
        adjacency = np.dot(adjacency.T, adjacency)

        def cluster_matrix(adjacency):
            # bandwidth reduction -> permutation s.t. distance on nonzero entries from the center diagonal is minimized
            # - via http://raphael.candelier.fr/?blog=Adj2cluster
            #   and http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
            # - results in blocks in the adjacency matrix that correspond with the clusters
            # - iteratively extend the block while the candidate region contains nonzero elements (i.e. is connected)
            r = csgraph.reverse_cuthill_mckee(csgraph.csgraph_from_dense(adjacency), True)
            clusters = [[r[0]]]
            for i in range(1, len(r)):
                if np.any(
                        adjacency[clusters[-1], r[i]]):  # ri connected to current cluster? -> add ri to cluster
                    clusters[-1].append(r[i])
                else:  # otherwise: start a new cluster with ri
                    clusters.append([r[i]])
            return clusters

        clusters = cluster_matrix(adjacency)

        # 2) heuristic check for direction of relation
        #    - x farther from camera than y -> x occluded by y
        #    - x higher than y -> x supported by y
        #    - note: we only use extreme points, so the dependencies will only be approximate
        def mask_to_world_coords(px_mask):
            umap = np.array([[j for _ in range(640)] for j in range(480)])
            vmap = np.array([[i for i in range(640)] for _ in range(480)])

            camera_intrinsics = observation['intrinsics']
            fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], \
                             camera_intrinsics[1, 2]
            valid_mask = np.logical_and(px_mask > 0, depth > 0)
            D_masked = depth[valid_mask > 0]
            X_masked = (vmap[valid_mask > 0] - cx) * D_masked / fx
            Y_masked = (umap[valid_mask > 0] - cy) * D_masked / fy
            return np.dot(np.dstack([X_masked, Y_masked, D_masked]),
                          observation['extrinsics'][:3, :3]) + np.linalg.inv(observation['extrinsics'])[:3, 3].T

        clouds = [mask_to_world_coords(mask).reshape(-1, 3) for mask in masks]
        max_z = [cloud[:, 2].max() for cloud in clouds]
        min_d = [observation['depth'][np.logical_and(mask, observation['depth'] > 0)].min() for mask in masks]

        supports = np.zeros_like(adjacency, dtype=np.int8)
        occluded = np.zeros_like(adjacency, dtype=np.int8)
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[0]):
                supports[i, j] = max_z[i] > max_z[j]
                occluded[i, j] = min_d[i] > min_d[j]

        # add information in opposite direction
        supports = supports - supports.T
        occluded = occluded - occluded.T

        # 2.1) sort objects per cluster by support
        supports[supports == 0] = occluded[supports == 0]  # use occlusion where support is inconclusive
        clusters = [sorted(cluster, key=cmp_to_key(lambda i1, i2: int(supports[i1, i2]))) for cluster in clusters]

        # 2.2) sort support clusters by occlusion between base objects
        clusters = sorted(clusters, key=cmp_to_key(lambda l1, l2: int(occluded[l1[0], l2[0]])))

        return clusters

