import jaynes
from ml_logger.job import instr
from params_proto.hyper import Sweep

from lucidsim_old.mesh_world.eval.mesh_depth_headless import EvalMesh


def main(**deps):
    # print('done!')
    # exit()

    eval = EvalMesh(**deps)
    eval()


with Sweep(EvalMesh) as sweep:
    EvalMesh.checkpoint_meta = "teacher"
    EvalMesh.task = "go1"
    EvalMesh.use_camera = False
    EvalMesh.checkpoint = "/lucid-sim/lucid-sim/baselines/launch_distill_grav_realsense/2024-01-06/02.17.41/go1/200/"

    with sweep.product:
        EvalMesh.dataset_prefix = [
            "stairs/scene_00002",
            "stairs/scene_00003",
            "stairs/scene_00004",
            "stairs/scene_00005",
            "stairs/scene_00006"
        ]

len(list(sweep))
print(list(sweep))

host = 'vision47'
visible_devices = [0, 1, 2, 3, 6, 7]

for i, deps in enumerate(sweep):
    thunk = instr(main, **deps)
    jaynes.config(mode="local")
    # jaynes.config(
    #     launch=dict(ip=host),
    #     runner=dict(
    #         envs=f"CUDA_VISIBLE_DEVICES={visible_devices[i]}"
    #     ),
    #     verbose=True,
    # )
    jaynes.run(thunk)

jaynes.listen()
