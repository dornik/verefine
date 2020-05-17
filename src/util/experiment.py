# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import torch
import numpy as np
from tqdm import tqdm
import argparse
from time import time
import os

import sys
sys.path.append('.')
import src.verefine.config as config
from src.verefine.simulator import Simulator
from src.verefine.renderer import Renderer
from src.verefine.verefine import Verefine
from src.util.dataset import VerefineDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VeREFINE')
    parser.add_argument('--dataset', type=str, default='lm', choices=['lm', 'ycbv', 'xapc'],
                        help='The dataset to use.')
    parser.add_argument('--mode', type=int, default=0, choices=list(range(6)),
                        help='0: baseline refiner. 1: PIR. 2: SIR. 3: RIR. 4: VFb. 5 VFd.')
    parser.add_argument('--refinement_iterations', type=int, default=-1,
                        help='Budget of refinement iterations per hypothesis. -1: use defaults')
    parser.add_argument('--lm_baseline', type=str, default='df', choices=['df', 'ppf'],
                        help='Which baseline pose estimator to use with LINEMOD.')
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    print("preparing...")
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    # init dataset
    print("   init dataset...")
    dataset = VerefineDataset(args)

    # optional: init simulator and renderer
    if config.MODE >= 1:
        print("   init simulator...")
        simulator = Simulator(dataset)
    else:
        simulator = None
    if config.MODE >= 2:
        print("   init renderer...")
        renderer = Renderer(dataset, dataset.width, dataset.height)
    else:
        renderer = None
    print("running experiment...")

    experiment_name = (f"VeREFINE-{dataset.baseline}-mode{config.MODE}-{config.REFINEMENT_ITERATIONS}refs"
                       f"_{args.dataset}-test")
    print(f"   results file: {experiment_name}")

    # init baseline refiner
    if dataset.baseline == 'df':
        from src.refinement.dfr import DenseFusionRefine

        refiner = DenseFusionRefine(dataset)
    elif dataset.baseline == 'ppf':
        from src.refinement import icp

        refiner = icp
    elif dataset.baseline == 'pcs':
        from src.refinement import tricp

        refiner = tricp

    # init verefine
    verefine = Verefine(dataset, refiner, simulator, renderer)

    # run experiment
    for observation, hypotheses in tqdm(dataset):

        if observation is None:
            continue

        # refine hypotheses
        st = time()
        final_hypotheses = verefine.refine(observation, hypotheses)
        duration = time() - st

        # write results in bop format
        scene, frame = observation['scene'], observation['frame']
        for hypothesis in final_hypotheses:

            if dataset.dataset == 'ycbv':
                # fix model alignment for BOP (DF is trained on YCBV differently centered models)
                offset = np.eye(4)
                offset[:3, 3] = dataset.obj_model_offset[int(hypothesis['obj_id']) - 1]
                hypothesis['pose'] = hypothesis['pose'] @ offset

            with open(f"./logs/{experiment_name}.csv", 'a') as file:
                file.write(f"{scene},{frame},{hypothesis['obj_id']},{float(hypothesis['confidence']):.3f},"
                           f"{' '.join([f'{v:.6f}' for v in hypothesis['pose'][:3, :3].reshape(9)])},"
                           f"{' '.join([f'{v * 1000:.6f}' for v in hypothesis['pose'][:3, 3]])},{duration:.3f}\n")

    if config.MODE >= 1:
        simulator.deinitialize()
