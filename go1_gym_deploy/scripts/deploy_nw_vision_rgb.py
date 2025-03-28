from params_proto import Flag, ParamsProto, Proto

import os
import sys
import torch
cxx_path = "/home/unitree/nw_deploy/parkour" 
if cxx_path not in sys.path:
    sys.path.append(cxx_path)
os.environ["PYTHONPATH"] = cxx_path + ":" + os.environ.get("PYTHONPATH", "")
from cxx.modules.parkour_actor import ParkourActor, PolicyArgs
from go1_gym_deploy.modules.base.rs_node import RealSenseCamera


class Args(ParamsProto):
    checkpoint: str = Proto(
        default="/lucid-sim/lucid-sim/scripts/train/2024-03-04/00.25.56/00.25.56/1/", help="Path to the model checkpoint."
    )
    device: str = Proto(default="cuda", help="Device to run the model on.")
    offline_mode: bool = Flag(default=False, help="Run the model in offline mode, with downloaded checkpoint.")
    direction_distillation: bool = Flag(default=False, help="Use direction distillation.")
    mode = "rgb"


if __name__ == "__main__":
    import time

    import lcm

    from go1_gym_deploy.modules.parkour.parkour_lcm_agent import ParkourLCMAgent
    from go1_gym_deploy.modules.parkour.parkour_state_estimator import ParkourStateEstimator
    from go1_gym_deploy.utils.deployment_runner import DeploymentRunner

    device = "cuda"
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    go1_test_policy = "/home/unitree/go2_hardware/scripts/stair_blind_kp30.jit"

    # 20 / 0.5
    # Args.checkpoint = "/lucid-sim/lucid-sim/scripts/train/2024-03-22/15.45.45/15.45.45/1"  # direciton distilation
    Args.checkpoint = "/lucid-sim/lucid-sim/baselines/launch_gains/2024-03-20/04.03.35/go1/300/20/0.5/checkpoints/model_last.pt"
    # Args.checkpoint = "/home/escher/nodistill.pt"
    # Args.offline_mode = True

    # 0.27 cam, direction distillation, lag 4
    # Args.checkpoint = "/instant-feature/scratch/2024/04-02/041420/checkpoints/model_last.pt"
    # Args.offline_mode = False
    # Args.direction_distillation = True

    # no direction, lag 6
    # Args.checkpoint = "/instant-feature/scratch/2024/04-02/041520/checkpoints/model_last.pt"
    # Args.offline_mode = False

    # Args.checkpoint = "/instant-feature/scratch/2024/05-23/003246/checkpoints/model_last.pt"

    # Warning: tilt a lot?
    # different seed + 6 delay + more envs, no direction distill
    # Args.checkpoint = "/instant-feature/scratch/2024/04-04/053203/checkpoints/model_last.pt"
    Args.offline_mode = True

    PolicyArgs.direction_distillation = Args.direction_distillation

    cam_node = RealSenseCamera(
        fps=30,
        res=(480, 640),
        # fps=30,
        # res=(360, 640),
    )
    state_estimator = ParkourStateEstimator(lc, "cpu")
    state_estimator.spin_process()

    lcm_agent = ParkourLCMAgent(
        state_estimator=state_estimator,
        cam_node=cam_node,
        device=Args.device,
    )

    # actor = ParkourActor()
    # actor.load(Args.checkpoint)
    # actor.to(Args.device)
    actor =  torch.jit.load(go1_test_policy, map_location=device)
    time.sleep(1)

    # acquire image here
    cam_node.spin_process(Args.mode)
    print("starting camera")
    while len(cam_node.frame) < 1:
        time.sleep(0.01)

    print("camera is ready")

    time.sleep(5)

    deployment_runner = DeploymentRunner(actor, lcm_agent, mode=Args.mode)

    max_steps = 10000000

    deployment_runner.run(max_steps=max_steps, vision_key=Args.mode, wait=True, debug=False)
    # deployment_runner.run(max_steps=max_steps, wait=True, debug=True)
