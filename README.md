# VeREFINE

This repository implements the methods described in *VeREFINE: Integrating Object Pose Verification with Physics-guided 
Iterative Refinement*. The code and data required to reproduce the results in Tables I, II and III are provided.

## Dependencies
The code has been tested on Ubuntu 16.04 and 20.04 with Python 3.6. To set-up the Python environment, use Anaconda and 
the provided YAML file:

`conda env create -f environment.yml --name verefine`

`conda activate verefine`

GLFW is required for rendering:

`sudo apt install libglfw3`

We provide bindings for ICP and Trimmed ICP in PCL via pybind11. This requires the following dependencies to be installed:

`sudo apt install build-essential libboost-dev libeigen3-dev libpcl-dev`

pybind11 is installed through the conda environment. The bindings themselves can be built using:

`bash src/refinement/cpp/build.sh`


## Datasets
We use the BOP toolkit for evaluation. Please refer to [this repository](https://github.com/thodan/bop_toolkit) for
installation instructions. We recommend to also install the CPP version of the renderer. 

The LINEMOD and YCB VIDEO datasets can be downloaded on the [BOP Challenge 2019 page](https://bop.felk.cvut.cz/datasets/). 
We require the base archive, object models and BOP test images. Extract them into some directory `$BOP_DIR`. In addition, 
we have converted the Rutgers Extended RGBD (Extended APC) dataset to the BOP format. 

The converted dataset, pre-computed hypotheses pools for all evaluated baselines and datasets, 
collider meshes and the target definitions needed for evaluation are provided 
[here](https://drive.google.com/file/d/1FmDx2YqBV6f3ELEK5PqlUYrJkSjpg8I-/view?usp=sharing). The following steps are 
required to prepare the datasets for the experiments and the evaluation:

* Extract the provided zip file into some directory `$VEREFINE_DIR`. There should be three directories - `bop`, 
`bop_toolkit` and `verefine`.
* The `bop/lm/test_targets_verefine.json` file needs to be moved to the respective BOP dataset folder in `$BOP_DIR/lm`.
* The `bop/xapc` directory needs to be moved to the BOP dataset directory `$BOP_DIR`.
* Copy the two files in `bop_toolkit` over the corresponding files in your BOP toolkit installation. This adds the 
definitions for the Extended APC dataset.
* Adapt the config file in `src/verefine/config.py` to reflect your dataset paths. `PATH_BOP19` should point to 
the BOP dataset directory `$BOP_DIR`. `PATH_VEREFINE` should point to the additional files in `$VEREFINE_DIR`.

## Evaluation
See `evaluate.sh` for the settings used to run the experiments. Note that the experiments for Tables I (ICP results) and
 II (TrICP) require the PCL bindings. Running `src/util/experiment.py` will create log files in the BOP format in the 
 `logs` directory. The logs can be evaluated using the BOP toolkit.

For the LINEMOD and Extended APC datasets, use:

`python $BOP_TOOLKIT_PATH/scripts/eval_bop19.py --result_filenames=$PATH_TO_LOG.csv --targets_filename=$BOP_DIR/$DATASET/test_targets_verefine.json`

For the YCB VIDEO dataset, use: 

`python $BOP_TOOLKIT_PATH/scripts/eval_bop19.py --result_filenames=$PATH_TO_LOG.csv --targets_filename=$BOP_DIR/ycbv/test_targets_bop19.json`
