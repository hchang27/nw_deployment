import isaacgym

import os
from asyncio import sleep

import numpy as np
from params_proto import Proto
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, TriMesh

from lucidsim_old import ROBOT_DIR
from lucidsim_old.mesh_world.traj_gen import TrajGenerator
from lucidsim_old.utils import ISAAC_DOF_NAMES, Go1, euler_from_quaternion

serve_root = os.environ["HOME"]
traj_gen = TrajGenerator(log=False,
                         dataset_prefix="gap/scene_00005",
                         prefix="scenes/experiments/real2real",
                         local_prefix=Proto(env="$DATASETS/lucidsim/scenes/experiments/real2real"),
                         add_noise_prob=0)

queries = dict(grid=False, background="000000")
app = Vuer(static_root=serve_root, queries=queries)

ROBOT_PATH = f"http://localhost:8012/static/{ROBOT_DIR}/gabe_go1/urdf/go1.urdf"


def set_poses(sess: VuerSession, root_states, dof_angles):
    joint_values = {name: angle.item() for name, angle in zip(ISAAC_DOF_NAMES, dof_angles[0].cpu())}

    quat_t = root_states[0, 3:7][None, ...]
    global_rot = euler_from_quaternion(quat_t.float())

    r, p, y = (angle.item() for angle in global_rot)

    position = root_states[0, :3]

    # cam_position = position.astype(np.float32) + \
    #                quat_rotate(quat_t.float(), torch.tensor([CameraArgs.cam_to_base]))[
    #                    0].numpy()

    position = position.tolist()
    # cam_position = cam_position.tolist()
    # mat = get_three_mat(cam_position, [r, p, y])

    sess.update @ (
        Go1(
            ROBOT_PATH,
            joint_values,
            global_rotation=(r, p, y),
            position=position,
        ),
    )


@app.spawn
async def main(sess: VuerSession):
    vertices = traj_gen.env.terrain.vertices
    faces = traj_gen.env.terrain.triangles
    print(f"Loaded mesh with {vertices.shape} vertices and {faces.shape} faces")
    xyz = np.array(vertices)
    uv = xyz[:, :2]  # take the x, y coordinates
    uv -= uv.min(axis=0)  # normalize the uv coordinates to 0 - 1
    uv /= uv.max(axis=0)

    sess.set @ DefaultScene(
        TriMesh(
            key="terrain",
            vertices=np.array(vertices),
            faces=np.array(faces),
            position=[-traj_gen.env.terrain.cfg.border_size, -traj_gen.env.terrain.cfg.border_size, 0],
            materialType="standard",
            # uv=uv,
            # material=dict(map="http://localhost:8012/static/grass.png")
        ),
        Go1(
            ROBOT_PATH,
            joints={name: 0 for name in ISAAC_DOF_NAMES},
            global_rotation=(0, 0, 0),
            position=(0, 0, 0),
        )
    )

    infos = {}

    num_labels = len(traj_gen.labels)
    labels_list = [(marker["start"], marker["goal"]) for marker in traj_gen.labels]

    traj_gen.env.set_terrain_goal(labels_list[10], 0, 0, shift_corners=True)
    obs, _ = traj_gen.env.reset()

    for _ in range(500):
        obs, infos = traj_gen.handle_step(obs, infos)

        root_states = traj_gen.env.root_states
        dof_angles = traj_gen.env.dof_pos

        set_poses(sess, root_states, dof_angles)

        await sleep(0.02)


app.run()
