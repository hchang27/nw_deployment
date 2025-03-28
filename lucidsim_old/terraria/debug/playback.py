import os
from asyncio import sleep
from typing import Literal

import torch
import torchvision.transforms.v2 as T
from ml_logger import logger
from params_proto import PrefixProto, Proto
from vuer import Vuer
from vuer.schemas import DefaultScene

from agility_analysis.matia.go1.dataset import LucidDreams
from agility_analysis.matia.go1.scandots_dataset import Scandots
from agility_analysis.matia.go1.stacked_dataset import StackedLucidDreams
from agility_analysis.matia.go1.stacked_scandots_dataset import StackedScandots
from lucidsim_old import ROBOT_DIR
from lucidsim_old.utils import DEFAULT_DOF_POS_UNITREE, ISAAC_DOF_NAMES, UNITREE_DOF_NAMES, Go1


class PlayBack(PrefixProto):
    local_dataset_root: str = os.path.join(os.environ["DATASETS"], "lucidsim",
                                           "scenes")  # , "experiments", "real2real")

    # scenes: List[str] = ["gap/scene_00003", "flat/scene_00002"]
    scene = "hurdle/scene_00001"
    trajectory = 4

    image_type: Literal["rgb", "depth", "augmented"] = "depth"

    image_size = (45, 80)

    normalize_depth = True

    # for frame stacking 
    use_stacked_dreams = False

    # the older frames will be fed in as differences with the most recent frame 
    compute_deltas = False
    drop_last = False
    stack_size = 1

    # system
    seed = 100
    device = "cuda:0"

    freeze_teacher_modules = False

    # gpu_load = False # whether to load all images into GPU memory

    teacher_checkpoint = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"
    # # rgb_checkpoint = "/alanyu/scratch/2024/02-17/143140/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-17/145856/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-17/171630/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-17/172559/checkpoints/net_500.pt"

    # recurrent, latent
    # rgb_checkpoint = "/alanyu/scratch/2024/02-19/163734/checkpoints/net_0.pt"
    # rgb_checkpoint = "/instant-feature/scratch/2024/02-16/185428/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-19/170127/checkpoints/net_500.pt"
    rgb_checkpoint = "/alanyu/scratch/2024/02-20/005314/checkpoints/net_500.pt"

    imagenet_pipe = True

    gpu_load = True

    no_val = False
    recurrent = False
    serve_root = Proto(env="$HOME")

    def __post_init__(self, _deps=None):
        if self.imagenet_pipe:
            pipeline = [
                T.Resize((45, 80), interpolation=T.InterpolationMode.BILINEAR),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            pipeline = [
                T.Resize((45, 80), interpolation=T.InterpolationMode.BILINEAR),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=False),
                T.Normalize(mean=[127.5, 127.5, 127.5], std=[255, 255, 255]),
            ]

        self.transform = T.Compose(pipeline)
        self.rgb_policy = logger.torch_load(PlayBack.rgb_checkpoint)
        if hasattr(self.rgb_policy.vision_head, "hidden_states"):
            self.rgb_policy.vision_head.hidden_states = None
        self.rgb_policy.eval()

        self.app = Vuer(static_root=self.serve_root, queries=dict(grid=False, background="b78585"), port=8013,
                        uri="ws://localhost:8013")

        loader_class = StackedLucidDreams if self.use_stacked_dreams else LucidDreams
        # loader_class = StackedScandots

        self.data_loader = loader_class(stack_size=self.stack_size,
                                        compute_deltas=self.compute_deltas,
                                        local_dataset_root=self.local_dataset_root,
                                        scenes=[self.scene],
                                        trajectories_per_scene=[[self.trajectory]],
                                        image_type=self.image_type,
                                        device=self.device,
                                        transform=self.transform,
                                        tensor_size=self.image_size,
                                        gpu_load=self.gpu_load,
                                        normalize_depth=self.normalize_depth,
                                        )

    async def main(self, sess):
        print("hey")
        self.urdf_path = f"http://localhost:8013/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf"
        
        print('urdf path', self.urdf_path)
        sess.set @ DefaultScene(
            Go1(self.urdf_path, joints={name: 0.0 for name in ISAAC_DOF_NAMES}, key="teacher"),
            Go1(self.urdf_path, joints={name: 0.0 for name in ISAAC_DOF_NAMES}, key="student", position=[0, 0.5, 0]),
            up=[0, 0, 1],
            endStep=len(self.data_loader.obs_data) - 1,
        )

        # first, run and cache student actions 
        self.student_actions = []
        self.teacher_actions = []

        self.losses = []

        rgb_buffer = torch.zeros((1, 3, *self.image_size), device=self.device)
        for i, (camera, obs, teacher_action) in enumerate(self.data_loader.sample_batch(batch_size=1, shuffle=False)):
            self.teacher_actions.append(teacher_action.cpu()[0])

            # rgb_buffer = torch.cat([rgb_buffer[1:], camera], dim=0)

            with torch.no_grad():
                # camera = obs[..., 53:53 + 132]
                student_action, _, _ = self.rgb_policy(camera, obs)
                # student_action, _, _ = self.rgb_policy(rgb_buffer.reshape(-1, *self.image_size)[None, ...], obs)

            self.student_actions.append(student_action.cpu()[0])
            self.losses.append((teacher_action - student_action).norm(2, dim=-1).item())

        self.losses
        
        # for i, (camera, obs, teacher_action) in enumerate(self.data_loader.sample_batch(batch_size=1, shuffle=False)):
        #     if self.use_stacked_dreams:
        #         batch, stack, channels, height, width = camera.shape
        # 
        #         if self.drop_last:
        #             # drop the most recent frame, and only keep the deltas
        #             camera = camera[:, :-1, :, :, :]
        # 
        #         camera = camera.reshape(batch, (stack - int(self.drop_last)) * channels, height, width)
        # 
        #     pred_actions, vision_latent, teacher_scandots_latent = self.rgb_policy(camera, obs)
        # 
        #     # used by cxx; around 0.6 - 0.7 is good 
        #     action_loss = (teacher_action - pred_actions).norm(p=2, dim=1).mean()
        #     latent_loss = (vision_latent - teacher_scandots_latent).norm(p=2, dim=1).mean()
        #     
        #     self.losses.append(action_loss.item())
        # self.losses
        # 
        import numpy as np
        print(np.mean(self.losses))        
        from matplotlib import pyplot as plt
        plt.plot(self.losses)
        plt.ylim(0, 2)
        plt.show()

        while True:
            await sleep(0.02)

    async def step_handler(self, event, sess):
        step = int(event.value["step"])

        teacher_action = self.teacher_actions[step]
        student_action = self.student_actions[step]

        action_loss = (teacher_action - student_action).norm(2, dim=-1)
        print("action loss", action_loss.item())

        teacher_joints_unitree = 0.25 * teacher_action + DEFAULT_DOF_POS_UNITREE
        student_joints_unitree = 0.25 * student_action + DEFAULT_DOF_POS_UNITREE

        sess.upsert @ [
            Go1(self.urdf_path,
                joints={name: teacher_joints_unitree[i].float().item() for i, name in enumerate(UNITREE_DOF_NAMES)},
                key="teacher"),
            Go1(self.urdf_path,
                joints={name: student_joints_unitree[i].float().item() for i, name in enumerate(UNITREE_DOF_NAMES)},
                key="student", position=[0, 0, 0]),
        ]
        # sess.update @ [
        #     Go1(self.urdf_path, joints={name: teacher_action[i] for i, name in enumerate(ISAAC_DOF_NAMES)},
        #         key="teacher"),
        #     Go1(self.urdf_path, joints={name: student_action[i] for i, name in enumerate(ISAAC_DOF_NAMES)},
        #         key="student", position=[0, 0, 1]),
        # ]

    def __call__(self):
        print('starting')
        self.app.add_handler("TIMELINE_STEP", self.step_handler)
        self.app.spawn(self.main)
        self.app.run()


if __name__ == '__main__':
    playback = PlayBack()
    playback()
