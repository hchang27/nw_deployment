import isaacgym

import os
from asyncio import sleep

import numpy as np
import torch
from params_proto import Proto, PrefixProto
from vuer import VuerSession, Vuer
from vuer.schemas import DefaultScene, Sphere, TriMesh

from lucidsim_old.ball_pit.traj_gen import BallTrajGenerator
from lucidsim_old import ROBOT_DIR
from lucidsim_old.utils import ISAAC_DOF_NAMES, euler_from_quaternion, get_three_mat, Go1, quat_rotate

serve_root = os.environ["HOME"]

traj_gen = BallTrajGenerator(log=False, render_video=False)

queries = dict(grid=False, background="000000")
app = Vuer(static_root=serve_root, queries=queries)

ROBOT_PATH = f"http://localhost:8012/static/{ROBOT_DIR}/gabe_go1/urdf/go1.urdf"

def set_poses(sess: VuerSession, root_states, dof_angles):
    joint_values = {name: angle.item() for name, angle in zip(ISAAC_DOF_NAMES, dof_angles[0].cpu())}

    quat_t = root_states[0, 3:7][None, ...]
    global_rot = euler_from_quaternion(quat_t.float())

    r, p, y = [angle.item() for angle in global_rot]

    position = root_states[0, :3]

    # cam_position = position.astype(np.float32) + \
    #                quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base]))[
    #                    0].numpy()

    position = position.tolist()
    # cam_position = cam_position.tolist()
    # mat = get_three_mat(cam_position, [r, p, y])

    ball_location = root_states[1, :3].tolist()

    sess.upsert @ (
        Go1(
            ROBOT_PATH,
            joint_values,
            global_rotation=(r, p, y),
            position=position,
        ),
        Sphere(
            key="ball",
            position=ball_location,
            args=[0.2, 20, 20],
            materialType="standard",
            material=dict(color="red"),
        ),
        # CameraView
    )


@app.spawn
async def main(sess: VuerSession):
    vertices = traj_gen.env.terrain.vertices
    faces = traj_gen.env.terrain.triangles

    sess.set @ DefaultScene(
        TriMesh(
            key="terrain",
            vertices=np.array(vertices),
            faces=np.array(faces),
        )
    )

    obs, _ = traj_gen.env.reset()
    for _ in range(500):
        obs = traj_gen.handle_step(obs)

        root_states = traj_gen.env.root_states
        dof_angles = traj_gen.env.dof_pos

        set_poses(sess, root_states, dof_angles)

        await sleep(0.02)


app.run()
