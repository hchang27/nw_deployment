from asyncio import sleep
from pathlib import Path

from ml_logger import logger
from tassa.events import ClientEvent
import torch

import pandas as pd

# actuator net
# prefix = "/alanyu/scratch/2023/12-29/000800"
# prefix = "/alanyu/scratch/2023/12-29/173531"

with logger.Prefix(prefix):
    data, = logger.load_pkl("log_dict.pkl")
    data = data['actions_unitree']
    data = torch.cat(data, dim=0).numpy()

end = len(data)

from vuer.server import Vuer, VuerProxy
from vuer.schemas import DefaultScene, Urdf, Movable

pi = 3.1415

default_angles_unitree = torch.tensor([[-0.1000, 0.8000, -1.5000, 0.1000, 0.8000, -1.5000, -0.1000, 1.0000,
                                        -1.5000, 0.1000, 1.0000, -1.5000]])

# app = Vuer(
#     static_root=Path(__file__).parent / "/Users/alanyu/urop/parkour/main_street/assets/robots/go1")

app = Vuer(static_root=Path(__file__).parent / "/Users/alanyu/urop/parkour/main_street/assets/robots/gabe_go1")

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

# host = "http://luma02.csail.mit.edu:8300"
host = "http://localhost:8012"


@app.spawn
async def main(session):
    session.set @ DefaultScene(
        Movable(
            Urdf(
                # src="http://localhost:8012/static/urdf/go1_new.urdf",
                src=f"{host}/static/urdf/go1.urdf",
                jointValues={
                    "FL_hip_joint": 0.1,
                    "RL_hip_joint": 0.1,
                    "FR_hip_joint": -0.1,
                    "RR_hip_joint": -0.1,
                    "FL_thigh_joint": 0.8,
                    "RL_thigh_joint": 1.0,
                    "FR_thigh_joint": 0.8,
                    "RR_thigh_joint": 1.0,
                    "FL_calf_joint": -1.5,
                    "RL_calf_joint": -1.5,
                    "FR_calf_joint": -1.5,
                    "RR_calf_joint": -1.5,
                },
                key="go1",
            ),
            position=[0, 0, 0.3],
            scale=0.4,
        ),
        grid=True,
        endStep=end
    )

    while True:
        await sleep(0.005)


async def frame_handler(e: ClientEvent, session: VuerProxy):
    print(f"event: {e}")

    step = e.value["step"]

    session.update @ Urdf(
        src=f"{host}/static/urdf/go1.urdf",
        jointValues={unitree_joint_names[j]: 0.25 * angle + default_angles_unitree[0, j].item() for j, angle in enumerate(data[step].tolist())},
        key="go1"
    )


app.add_handler("TIMELINE_STEP", frame_handler)
app.run()
