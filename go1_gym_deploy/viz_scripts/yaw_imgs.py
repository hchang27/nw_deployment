import torch
from params_proto import Flag, ParamsProto, Proto
from PIL import Image


class Args(ParamsProto):
    checkpoint: str = Proto(default="/lucid-sim/lucid-sim/scripts/train/2024-03-04/00.25.56/00.25.56/1/",
                            help="Path to the model checkpoint.")
    device: str = Proto(default="cuda", help="Device to run the model on.")
    offline_mode: bool = Flag(default=False, help="Run the model in offline mode, with downloaded checkpoint.")


if __name__ == '__main__':
    Args.device = "cpu"

    import lcm

    from go1_gym_deploy import task_registry

    Args.checkpoint = "/alanyu/scratch/2024/03-15/155934/checkpoints/net_last.pt"

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

    env_cfg, train_cfg, state_estimator, lcm_agent, actor, cam_node = task_registry.make_agents(
        "go1_ball",
        lc,
        Args.device,
        Args.checkpoint,
        offline=Args.offline_mode,
    )
    import numpy as np
    from matplotlib import pyplot as plt

    img = Image.open("/Users/alanyu/Desktop/test_1.png")
    img = np.array(img)[:, :, :3]
    
    plt.imshow(img)
    plt.show()

    img = lcm_agent.process_frame(img)

    plt.imshow(img[0].permute(1, 2, 0))
    plt.show()

    # reflect img
    img = img.flip(-1)

    action, latent, teacher_latent, yaw = actor(img, torch.zeros(1, 583, dtype=torch.float32), None)
    print(yaw)
