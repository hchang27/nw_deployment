from asyncio import sleep
from pathlib import Path

from ml_logger import logger

import lcm

from tassa.events import ClientEvent
import torch
from vuer.serdes import jpg

from matplotlib import pyplot as plt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

default_joint_angles = {  # = target angles [rad] when action = 0.0
    'FL_hip_joint': 0.1,  # [rad]
    'RL_hip_joint': 0.1,  # [rad]
    'FR_hip_joint': -0.1,  # [rad]
    'RR_hip_joint': -0.1,  # [rad]

    'FL_thigh_joint': 0.8,  # [rad]
    'RL_thigh_joint': 1.,  # [rad]
    'FR_thigh_joint': 0.8,  # [rad]
    'RR_thigh_joint': 1.,  # [rad]

    'FL_calf_joint': -1.5,  # [rad]
    'RL_calf_joint': -1.5,  # [rad]
    'FR_calf_joint': -1.5,  # [rad]
    'RR_calf_joint': -1.5,  # [rad]
}

unitree_joint_names = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                       'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                       'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                       'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']

isaacgym_joint_names = [
    'FL_hip_joint',
    'FL_thigh_joint',
    'FL_calf_joint',
    'FR_hip_joint',
    'FR_thigh_joint',
    'FR_calf_joint',
    'RL_hip_joint',
    'RL_thigh_joint',
    'RL_calf_joint',
    'RR_hip_joint',
    'RR_thigh_joint',
    'RR_calf_joint',
]

JOINT_IDX_MAPPING = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

delay = 0

# prefix = "/lucid-sim/lucid-sim/2023-12-28/20.44.12/scripts/train/distillation/2023-12-28/20.44.12/go1/200"

# prefix = "/alanyu/scratch/2023/12-31/130804"
# prefix = "/alanyu/scratch/2023/12-31/131801"
# prefix = "/alanyu/scratch/2023/12-31/132326"

# neural
# prefix = "/alanyu/scratch/2023/12-31/133753"

# ultra
# prefix = "/alanyu/scratch/2023/12-31/134357"

# prefix = "/alanyu/scratch/2023/12-31/141525"

# prefix = "/alanyu/scratch/2024/01-03/161711"


# realsense
# prefix = "/alanyu/scratch/2024/01-03/192604"
# prefix = "/alanyu/scratch/2024/01-03/194052"
# prefix = "/alanyu/scratch/2024/01-03/203930"
# prefix = "/alanyu/scratch/2024/01-04/223649"

# prefix = "/alanyu/scratch/2024/01-04/224404"
# prefix="/alanyu/scratch/2024/01-07/202157"

prefix = "/alanyu/scratch/2024/01-08/200534"

with logger.Prefix(prefix):
    data, = logger.load_pkl("log_dict.pkl")

depths = data['depths'][delay:]
imu = torch.cat(data['obs'][delay:])[:, :2]

roll, pitch = imu[:, 0], imu[:, 1]
data = data['actions_unitree'][delay:]
data = torch.cat(data, dim=0).detach()[:, JOINT_IDX_MAPPING]

depth_updates = [depth.numpy() for depth in depths if depth is not None]
logger.save_video(depth_updates, "depth.mp4", fps=10)
print(logger.get_dash_url())
### Actions playback ###

# data = data['actions_isaac'][delay:]
# data = torch.cat(data, dim=0).detach()
# 
data = 0.25 * torch.clip(data, -4.8, 4.8) + \
       torch.tensor([default_joint_angles[i] for i in isaacgym_joint_names], device=data.device)[None, ...]
# 
data = data.cpu().numpy()
# 
end = len(data) - delay - 1

##########################

### Observations Playback ###

# from go1_gym_deploy.task_registry import task_registry
# 
# env_cfg, train_cfg, state_estimator, lcm_agent, actor, zed_node = task_registry.make_agents(
#     "go1",
#     lc,
#     "cuda:0",
#     prefix,
# )
# 
# obses = data['obs'][delay:]
# depths = data['depth'][delay:]
# depth_latent = None
# 
# actions = []
# 
# for i in range(len(obses)):
#     action, depth_latent = actor(obses[i], {depths[i]}, depth_default=depth_latent)
#     actions.append(action.detach())
# 
# actions = torch.cat(actions, dim=0)
# actions = 0.25 * torch.clip(actions, -4.8, 4.8) + \
#           torch.tensor([default_joint_angles[i] for i in isaacgym_joint_names], device=actions.device)[None, ...]
# 
# data = actions.cpu().numpy()
# 
# end = len(data) - delay

##################################

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Urdf, Movable, ImageBackground

pi = 3.1415

app = Vuer(static_root=Path(__file__).parent / "/Users/alanyu/urop/parkour/main_street/assets/robots/gabe_go1")

# app = Vuer(static_root=Path(__file__).parent / "/home/exx/mit/parkour/main_street/assets/robots/gabe_go1")
# host = "http://luma02.csail.mit.edu:8300"
host = "http://localhost:8012"

cmap = plt.get_cmap('viridis')


@app.spawn
async def main(session):
    session.set @ DefaultScene(
        grid=True,
        endStep=end
    )

    while True:
        await sleep(0.005)


async def frame_handler(e: ClientEvent, session: VuerSession):
    print(f"event: {e}")

    step = e.value["step"]

    session.upsert @ [
        Urdf(
            src=f"{host}/static/urdf/go1.urdf",
            jointValues={isaacgym_joint_names[j]: angle for j, angle in enumerate(data[step].tolist())},
            # rotation=[roll[step].item(), pitch[step].item(), 0],
            key="go1"
        ),
        # ImageBackground(cmap(frame), fixed=True, depthOffset=0.5, key='ego-view')
        # ImageBackground(cmap(frame), fixed=True, depthOffset=0.5, position=[0, 0, 0],
        #                 rotation=[roll[step].item(), pitch[step].item(), 0], key='ego-view')
    ]


app.add_handler("TIMELINE_STEP", frame_handler)
app.run()
