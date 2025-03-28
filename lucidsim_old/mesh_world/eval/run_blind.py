import jaynes
from ml_logger.job import instr
from params_proto.hyper import Sweep

from lucidsim_old.mesh_world.eval.mesh_depth_headless import EvalMesh


def main(**deps):
    eval = EvalMesh(**deps)
    eval()


with Sweep(EvalMesh) as sweep:
    EvalMesh.checkpoint_meta = "blind"
    EvalMesh.task = "go1_flat"
    EvalMesh.use_camera = False
    EvalMesh.checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-01-19/23.48.06/go1_flat/200/"

    with sweep.product:
        EvalMesh.dataset_prefix = [
            # "stairs/scene_00002",
            "stairs/scene_00003",
            "stairs/scene_00004",
            "stairs/scene_00005",
            "stairs/scene_00006"
        ]

print(" there are", len(list(sweep)))
print(list(sweep))

for deps in sweep:
    thunk = instr(main, **deps)
    jaynes.config(mode="local")
    jaynes.run(thunk)
