# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna


PATH_BOP19 = '/path/to/BOP19/'
PATH_VEREFINE = '/path/to/verefine/'

# VEREFINE
HYPOTHESES_PER_OBJECT = 5
REFINEMENT_ITERATIONS = 2
SIMULATION_STEPS = 3
MODE = 0
# MODES = {
#     0: "BASELINE",
#     1: "PIR",
#     2: "SIR",
#     3: "RIR",
#     4: "VFb",
#     5: "VFd"
# }

# PHYSICS SIMULATION
TIME_STEP = 1/60
SOLVER_ITERATIONS = 10
SOLVER_SUB_STEPS = 4

# REFINEMENT
ICP_ITERATIONS = 10  # per VeREFINE iteration
ICP_P_DISTANCE = 0.1
DFR_ITERATIONS = 1  # per VeREFINE iteration
TRICP_TRIM = 1.0  # as in PHYSIM-MCTS (Mitash et al.)

# RENDERING
CLIP_NEAR, CLIP_FAR = 0.01, 5.0  # clipping distances in renderer

# BANDIT
C = 0.1  # exploration rate in UCB
GAMMA = 0.99  # discount factor in D-UCB
