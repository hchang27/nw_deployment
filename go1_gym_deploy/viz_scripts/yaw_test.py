import asyncio

import torch
from params_proto import Flag, ParamsProto, Proto
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, ImageBackground
from vuer.serdes import jpg


class Args(ParamsProto):
    checkpoint: str = Proto(default="/lucid-sim/lucid-sim/scripts/train/2024-03-04/00.25.56/00.25.56/1/",
                            help="Path to the model checkpoint.")
    device: str = Proto(default="cuda", help="Device to run the model on.")
    offline_mode: bool = Flag(default=False, help="Run the model in offline mode, with downloaded checkpoint.")


if __name__ == '__main__':
    import time

    import lcm

    from go1_gym_deploy import task_registry

    Args.checkpoint = "/alanyu/scratch/2024/03-15/155934/checkpoints/net_last.pt"

    device = "cuda"
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

    env_cfg, train_cfg, state_estimator, lcm_agent, actor, cam_node = task_registry.make_agents(
        "go1_ball",
        lc,
        Args.device,
        Args.checkpoint,
        offline=Args.offline_mode,
    )

    cam_node.spin_process()
    print("starting camera")
    while len(cam_node.frame) < 1:
        time.sleep(0.01)

    print('camera is ready')

    app = Vuer(
        uri="ws://localhost:8112",
        queries=dict(
            reconnect=True,
            grid=False,
            backgroundColor="black",
        ),
        port=8112,
    )


    @app.spawn(start=True)
    async def show_heatmap(session: VuerSession):
        import numpy as np
        
        session.set @ DefaultScene()

        while True:
            obs = torch.zeros((1, 583), device="cuda")
            frame = lcm_agent.retrieve_depth(force=True)
            data = frame.cpu()[0].permute(1,2,0).numpy()
            
            # normalize
            data = (((data - data.min()) / (data.max() - data.min())) * 255).astype(np.uint8) 
            
            
            print(data.shape, data.dtype)

            session.upsert(
                ImageBackground(
                    src=jpg(data, 50),
                    key="image",
                ),
                to="bgChildren",
            )

            action, vision_latent, teacher_scandots_latent, yaw = actor(frame,
                                                                        obs,
                                                                        vision_latent=None,
                                                                        )
            print("Yaw:", yaw)
            await asyncio.sleep(0.01)
