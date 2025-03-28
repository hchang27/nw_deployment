import os
from asyncio import sleep
from typing import Literal

import torch
import torchvision.transforms.v2 as T
from ml_logger import logger
from params_proto import PrefixProto, Proto
from PIL import Image
from tqdm import tqdm
from vuer import Vuer
from vuer.schemas import DefaultScene

from lucidsim_old import ROBOT_DIR
from lucidsim_old.utils import DEFAULT_DOF_POS_UNITREE, ISAAC_DOF_NAMES, UNITREE_DOF_NAMES, Go1


class PlayBack(PrefixProto):
    local_dataset_root: str = os.path.join(os.environ["DATASETS"], "lucidsim",
                                           "scenes")  # , "experiments", "real2real")

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

    # teacher_checkpoint = "/lucid-sim/lucid-sim/baselines/launch_grav_teacher_stairs/2024-01-11/11.04.12/go1/700"
    # # rgb_checkpoint = "/alanyu/scratch/2024/02-17/143140/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-17/145856/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-17/171630/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-17/172559/checkpoints/net_500.pt"

    # recurrent, latent
    # rgb_checkpoint = "/alanyu/scratch/2024/02-19/163734/checkpoints/net_0.pt"
    # rgb_checkpoint = "/instant-feature/scratch/2024/02-16/185428/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-19/170127/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-20/005314/checkpoints/net_500.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-21/134631/checkpoints/net_250.pt"
    # rgb_checkpoint = "/alanyu/scratch/2024/02-21/140216/checkpoints/net_500.pt"
    # teacher_checkpoint = "/alanyu/scratch/2024/02-20/005314/checkpoints/net_0.pt"
    rgb_checkpoint = "/alanyu/scratch/2024/02-22/233553/checkpoints/net_50.pt"
    teacher_checkpoint = "/alanyu/scratch/2024/02-22/233553/checkpoints/net_0.pt"

    # teacher trajectory, fixed the buffer issue.
    # log_path = "/instant-feature/scratch/2024/02-21/132822"

    # student trajectory, fixed the buffer issue.
    log_path = "/instant-feature/scratch/2024/02-21/132019"

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

        self.teacher_policy = logger.torch_load(PlayBack.teacher_checkpoint)
        if hasattr(self.teacher_policy.vision_head, "hidden_states"):
            self.teacher_policy.vision_head.hidden_states = None
        self.teacher_policy.eval()

        self.app = Vuer(static_root=self.serve_root, queries=dict(grid=False, background="b78585"), port=8013,
                        uri="ws://localhost:8013")

    async def main(self, sess):
        import numpy as np
        print("hey")
        self.urdf_path = f"http://localhost:8013/static/{os.path.relpath(ROBOT_DIR, self.serve_root)}/gabe_go1/urdf/go1.urdf"

        print('urdf path', self.urdf_path)

        with logger.Prefix(self.log_path):
            image_list = sorted(logger.glob("depth/*.png"))
            logger.make_video(image_list, "depth.mp4", fps=50)
            data, = logger.load_pkl("log_data.pkl")

            images = []

            for image_path in tqdm(image_list, desc="loading images"):
                buff = logger.load_file(image_path)
                img = np.array(Image.open(buff))
                # img = torch.from_numpy(img).permute(2, 0, 1)
                images.append(img)

        sess.set @ DefaultScene(
            Go1(self.urdf_path, joints={name: 0.0 for name in ISAAC_DOF_NAMES}, key="teacher"),
            Go1(self.urdf_path, joints={name: 0.0 for name in ISAAC_DOF_NAMES}, key="student", position=[0, 0.0, 0.0]),
            up=[0, 0, 1],
            endStep=len(data["obs"]) - 1,
        )

        # first, run and cache student actions 
        self.losses = []

        teacher = []
        student = []

        for i, image in enumerate(images):
            obs = torch.from_numpy(data["obs"][i]).float().to(self.device)
            log_teacher_action = torch.from_numpy(data["teacher_actions"][i]).float().to(self.device)
            log_student_action = torch.from_numpy(data["student_actions"][i]).float().to(self.device)

            # student.append(log_student_action)

            if self.normalize_depth:
                image = (image - image.min()) / (image.max() - image.min() + 1e-3) * 255
                image = image.astype(np.uint8)

            image = torch.from_numpy(image).to(self.device)[None, ...]
            image = image.permute(0, 3, 1, 2)

            # from matplotlib import pyplot as plt
            # plt.imshow(image.cpu().permute(1, 2, 0))
            # plt.show()
            image = self.transform(image)

            # student_action, _, _ = self.rgb_policy(image, obs)
            with torch.no_grad():
                student_action = self.rgb_policy(image, obs)
                student.append(student_action[0])

                teacher_action = self.teacher_policy(None, obs)
                teacher.append(teacher_action[0])

        self.student_actions = torch.stack(student)[:, 0, :].cpu()
        self.teacher_actions = torch.stack(teacher)[:, 0, :].cpu()

        self.losses = (self.student_actions - self.teacher_actions).norm(2, dim=-1).mean().item()
        print(f"loss: {self.losses}")

        import numpy as np
        # print(np.mean(self.losses))
        # from matplotlib import pyplot as plt
        # plt.plot(self.losses)
        # plt.ylim(0, 2)
        # plt.show()

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
