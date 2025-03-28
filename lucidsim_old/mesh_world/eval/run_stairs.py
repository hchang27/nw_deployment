from params_proto.hyper import Sweep

from lucidsim_old.mesh_world.eval.mesh import EvalMesh

from ml_logger.job import instr
import jaynes

def main(**deps):
    eval = EvalMesh(**deps)
    eval()


with Sweep(EvalMesh) as sweep:
    EvalMesh.port = 8012
    EvalMesh.checkpoint_meta = "rgb"
    EvalMesh.rgb_checkpoint = "/alanyu/scratch/2024/02-02/022538/checkpoints/net_1000.pt"

    with sweep.product:
        EvalMesh.dataset_prefix = [
            # "stairs/scene_00002",
            # "stairs/scene_00003",
            "stairs/scene_00004",
            "stairs/scene_00005",
            "stairs/scene_00006"
        ]

len(list(sweep))
print(list(sweep))

for deps in sweep:
    try:
        thunk = instr(main, **deps)
        jaynes.config(mode="local")
        jaynes.run(thunk)
    except ValueError:
        continue
