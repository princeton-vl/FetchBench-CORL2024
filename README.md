# FetchBench Benchmark Environments


## About this repository

This repository contains Isaac-Gym environments for the FetchBench benchmark (https://arxiv.org/abs/2406.11793) .

## 1. Installation

Please follow the steps below to perform the installation：

### Create Virtual Env

We suggest using python=3.8 and numpy=1.23.5.

```
conda create -n FetchBench python=3.8 numpy=1.23.5
conda activate FetchBench
```

### Install Pytorch

We suggest using pytorch=1.13.0. Please ref: https://pytorch.org/get-started/previous-versions/ .

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Install Python Dependencies

```
pip install -r requirement.txt
```

### Download Assets

Please download environment asset (asset_release.zip) from https://drive.google.com/drive/folders/0AFFm3c3bWvtLUk9PVA?dmr=1&ec=wgc-drive-hero-goto.

The assets include procedural assets generated from Infinigen (https://github.com/princeton-vl/infinigen) and third-party asset from Acronym dataset (https://github.com/NVlabs/acronym, The dataset is released under CC BY-NC 4.0.).

### Install Thirdy-party Packages

Download third_party files (3rd_parties.zip) from https://drive.google.com/drive/folders/0AFFm3c3bWvtLUk9PVA?dmr=1&ec=wgc-drive-hero-goto and put it under FetchBench/ . The packages under thirdy_party contain third-party codes from the following sources.

#### Install Isaac-Gym

We provide a copy of IsaacGym 4.0.0 (https://developer.nvidia.com/isaac-gym/download).

```
cd third_party/isaac-gym/python
pip install -e .
```

#### Install Curobo

We provide an adapted version of CuRobo 0.6.2 (https://curobo.org/).

```
cd third_party/curobo
pip install -e . --no-build-isolation
```

For cuda version mismatch issue, one can install the cudatoolkit-dev as follows

```
conda install conda-forge::cudatoolkit-dev
```

#### (Optional) Install Contact-GraspNet-Pytorch

If you want to run methods using contact-graspnet, we provide a copy of contact-graspnet-pytorch (https://github.com/elchun/contact_graspnet_pytorch).

```
cd third_party/contact_graspnet_pytorch
pip install -e .
```

#### (Optional) Install OMPL Packages

If you want to run methods using ompl motion planning packages, please follow the step in (https://github.com/lyfkyle/pybullet_ompl?tab=readme-ov-file), and add the ompl python-bindings to the conda environment.

#### (Optional) Install Cabinet and SceneCollisionNet

If you want to run methods using cabinet, we provide a copy of scenecollisionnet (https://github.com/NVlabs/SceneCollisionNet) and cabinet (https://github.com/NVlabs/cabi_net).

```
conda install pytorch-scatter -c pyg
cd third_party/SceneCollisionNet
pip install -e .
cd third_party/cabinet
pip install -e .
cd third_party/cabinet/pointnet2
pip install -e .
```

#### (Optional) Download Imitation Learning Models

If you want to run imitation learning models, please download the checkpoints (imit_ckpts.zip) from https://drive.google.com/drive/folders/0AFFm3c3bWvtLUk9PVA?dmr=1&ec=wgc-drive-hero-goto.

#### FAQ:

1. What if I want to test some baselines but do not want to install other additional packages, e.g., OMPL?

One can modify the code in InfiniGym/isaacgymenvs/tasks/__init__.py to comment out the corresponding methods' import to prevent explicitly loading these uninstalled packages. For example, if one does not install OMPL python-bindings, one should comment out all methods with Pyompl keywords. In this way, one can still test other methods with cabinet or contact-graspnet-pytorch.

## 2. Run


### Add Env variables.

Please add the ASSET_PATH environment variable to specify the path to the asset directory.

```
export ASSET_PATH=/path/to/the/assets
```

### Minimal installation Test

For minimal installation of isaacgym and curobo, one can run:

```
cd InfiniGym

python isaacgymenvs/eval.py task=FetchMeshCurobo scene=benchmark_eval/RigidObjDesk_0
```
### Benchmark Test

The overall command to test each method is

``` 
python isaacgymenvs/eval.py task=${METHOD} scene=bechmark_eval/${TASK} task.solution.XXX=YYY (Overwrite configs)...
```

The list of \${METHOD} is shown in isaacgymenvs/config/task and the list of benchmark \${TASK} are shown in isaacgymenvs/config/scene/benchmark_eval .

To be specific, to run the imitation learning models with a specific checkpoint, run:

```
python isaacgymenvs/eval.py task=FetchPtdImit${TYPE} scene=${TASK} task.solution.ckpt_path=/path/to/checkpoint/folder
```
where ${TYPE} in \{E2E, TwoStage, CuroboCGN\}.

### Reference Code

1. We provide reference code to generate infinite training tasks in InfiniGym/isaacgymenvs/tasks/fetch/infini_scene/infini_scenes.py.

2. We provide reference code to generate infinite expert fetching trajectories in InfiniGym/isaacgymenvs/data_gen.py and InfiniGym/isaacgymenvs/tasks/fetch/fetch_mesh_curobo_datagen.py .

3. We provide reference code to train the imitation learning models in InfiniGym/isaacgymenvs/train_imit.py. The code submodule (https://github.com/princeton-vl/FetchBench-Imit.git) is adapted from Optimus (https://github.com/NVlabs/Optimus?tab=readme-ov-file) under Nvidia License.

4. We provide reference code to summarize the results of all benchmark tasks in InfiniGym/isaacgymenvs/result.py .

5. We will release the baseline dataset and the data generation pipeline soon. Please contact us if you would like to have these asap.

## Citing

If you find our code useful, please cite:

```
@article{han2024fetchbench,
  title={FetchBench: A Simulation Benchmark for Robot Fetching},
  author={Han, Beining and Parakh, Meenal and Geng, Derek and Defay, Jack A and Gan, Luyang and Deng, Jia},
  journal={arXiv preprint arXiv:2406.11793},
  year={2024}
}
```

