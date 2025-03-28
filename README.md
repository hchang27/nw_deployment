# Parkour

## Setup IsaacGym (v2)

**Update 2024-05-05**: It turned out that IsaacGym has some weird dependencies. I
could not get it to work using the setup guide below, so I resorted to using the 
frozen dependency list from `luma02` to install the dependencies. The key is to
pass `--no-deps` when installing from the requirements.txt file.

```shell
yes | pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
yes | pip install -r dependencies/luma02_requirements.txt --no-deps
yes | pip install ml-logger params-proto functional_notations waterbear --no-deps
yes | pip install matplotlib
```

## Setup IsaacGym

**Important Notes**: IsaacGym `Preview4` has strict dependency
on `cuda` toolkit `11.8` and `numpy==1.23.5`. Higher CUDA and
`numpy` versions will not work.

The easiest way to install CUDA toolkits (and the `pytorch`
distribution) is to use conda:

```shell
yes | conda create -n parkour python=3.8
conda activate parkour

conda install -q cuda -c nvidia/label/cuda-11.8.0
conda install -q pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
yes | pip install "numpy==1.23.5" 
```

Since IsaacGym is 💩'tty, you have to pamper its dependencies carefully. Run the following to install the dependencies without their child dependencies. Note that this `--no-deps` flag is important to avoid wasting your time.

```shell
 pip -q install torchtyping --no-deps
 pip -q install ml-logger params-proto functional_notations waterbear --no-deps
 pip -q install vuer
 pip -q install scipy --no-deps
 pip -q install more_itertools
 yes | pip install boto3 cloudpickle==1.3.0 dill google-api-python-client google-cloud-storage imageio imageio-ffmpeg 'jaynes>=0.9.0' matplotlib pycurl requests-futures requests-toolbelt ruamel.yaml sanic sanic-cors scikit-image scikit-video wcmatch --no-deps
```

Afterward

```shell
pip -q install typeguard
pip -q install packaging
pip -q install pyparsing
pip -q install cycler
pip -q install open3d
pip -q install open3d --no-deps
pip -q install dash
pip -q install pandas --no-deps
pip -q install pyfqmr --no-deps
pip -q install pydelatin --no-deps
pip -q install wandb --no-deps
pip -q install appdirs --no-deps
pip -q install opencv-python --no-deps
```

```shell
yes | pip install python-opencv-headless tqdm
```

```shell
pip install pydelatin tqdm opencv-python ipdb pyfqmr
pip install ml-logger functional_notations jaynes
```

Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym (Again, use `pytorch==11.8`!

```shell
cd isaacgym/python && pip install -e .
```

## LucidSim Pipeline

We include the following domains: `ball chasing`, `gap`, `hurdle`, `stairs`.

### Ball

1. **Generate teacher trajectories.** You can see an example of this in the main block of `lucidsim.traj_generation.diverse_traj_wrapper.py` .
   Please don't overwrite existing datasets though.
   - Set the dataset name inside `lucidsim.ball_pit.ball_traj_gen.py` in the `dataset_prefix` field before you run
2. **Render Ego Views.** Render with `lucidsim.ball_pit.ego_view_farm_mask.py`
   - Set the dataset prefix to the one from above in the `dataset_prefix` field.
   - You will need to render each type of input (depth, background_mask, object_mask). To do this, just change the `render_type` argument
   - Please don't overwrite existing datasets
3. **Generate Realistic Images.** For the ball, you run `imagen.workflows.sdxl_turbo.object_background_masking.py`, with the dataset prefix and set of prompts. `Imagen` repo needs heavy cleanup.
4. **Behavior Cloning.** run BC in `agility_analysis.matia.go1.main_tracking.py`
   - You can set the datasets you want to use for train/val (see the example)
   - There are also a few training parameters you can mess with
5. **Evaluate in Synthetic Environments.** Using your checkpoint, run evaluation in `dog-park/lucidsim/play_tracking.py`

### Usage

`cd main_street/scripts`

1. Train base policy:

```bash
python train.py --exptid xxx-xx-WHATEVER --device cuda:0
```

Train 10-15k iterations (8-10 hours on 3090) (at least 15k recommended).

2. Train distillation policy:

```bash
python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera
```

Train 5-10k iterations (5-10 hours on 3090) (at least 5k recommended).

> You can run either base or distillation policy at arbitary gpu # as long as you set `--device cuda:#`, no need to
> set `CUDA_VISIBLE_DEVICES`.

3. Play base policy:

```bash
python play.py --exptid xxx-xx
```

No need to write the full exptid. The parser will auto match runs with first 6 strings (xxx-xx). So better make sure you don't reuse xxx-xx.
Delay is added after 8k iters. If you want to play after 8k, add `--delay`

4. Play distillation policy:

```bash
python play.py --exptid yyy-yy --delay --use_camera
```

5. Save models for deployment:

```bash
python save_jit.py --exptid xxx-xx
```

This will save the models in `main_street/logs/parkour_new/xxx-xx/traced/`.

### Viewer Usage

Can be used in both IsaacGym and web viewer.

- `ALT + Mouse Left + Drag Mouse`: move view.
- `[ ]`: switch to next/prev robot.
- `Space`: pause/unpause.
- `F`: switch between free camera and following camera.

### Arguments

- --exptid: string, can be `xxx-xx-WHATEVER`, `xxx-xx` is typically numbers only. `WHATEVER` is the description of the run.
- --device: can be `cuda:0`, `cpu`, etc.
- --delay: whether add delay or not.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --use_camera: use camera or scandots.
- --web: used for playing on headless machines. It will forward a port with vscode and you can visualize seemlessly in vscode with your idle
  gpu or cpu. [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) vscode extension required, otherwise
  you can view it in any browser.
