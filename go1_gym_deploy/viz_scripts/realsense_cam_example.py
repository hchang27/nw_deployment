import time
from asyncio import sleep

import numpy as np
from matplotlib.cm import get_cmap
from vuer import Vuer
from vuer.schemas import DefaultScene, ImageBackground
from vuer.serdes import jpg

# from go1_gym_deploy.cfgs.go1_config import Go1ParkourCfg
from go1_gym_deploy.modules.base.rs_node import RealSenseCamera
from go1_gym_deploy.modules.lucidsim.transformer_lcm_agent import TransformerLCMAgent

# rs = RealSenseCamera(fps=30, res=(144, 256))
# rs = RealSenseCamera(fps=90, res=(144, 256))
rs = RealSenseCamera(fps=60, res=(270, 480))
# rs = RealSenseCamera()
rs.spin_process("depth")

print("starting camera")
while len(rs.frame) < 1:
    time.sleep(0.01)

print("camera is ready")
# while True:
#     print(rs.frame["depth"].mean())
lcm_agent = TransformerLCMAgent(
        imagenet_pipe=False,
        stack_size=7,
        state_estimator=None,
        render_type="depth",
        device="cuda",
        cam_node=rs,
    )
# lcm_agent = ParkourLCMAgent(
#     **vars(Go1ParkourCfg.policy),
#     **vars(Go1ParkourCfg.control),
#     **vars(Go1ParkourCfg.vision),
#     **vars(Go1ParkourCfg.terrain),
#     **vars(Go1ParkourCfg.init_state),
#     obs_scales=Go1ParkourCfg.obs_scales,
#     state_estimator=None,
#     device="cuda",
#     cam_node=rs,
# )

NEAR_CLIP = 0.28

cmap = get_cmap("gray")

if __name__ == "__main__":
    app = Vuer()

    @app.spawn
    async def main(sess):
        sess.set @ DefaultScene()

        while True:
            depth = rs.frame["depth"]
            depth = np.clip(depth, NEAR_CLIP, 2.0)
            depth = (depth - 0.28) / (2.0 - 0.28)
            depth = (depth) * 255
            depth = depth.astype("uint8")
            
            # depth = lcm_agent.retrieve_vision(force=True).cpu()[0][0][0]
            # depth = depth.permute(0, 1)
            # depth = depth.numpy()
            # 
            # print("shape", depth.shape)
            # depth = (cmap(depth)[0, ..., :3]) * 255

            # print(depth.dtype)
            # depth = (((depth - depth.min()) / depth.max()) * 255).astype(np.uint8)[0]

            sess.upsert(
                ImageBackground(
                    src=jpg(depth, 50),
                    key="image",
                ),
                to="bgChildren",
            )

            await sleep(0.1)

    app.run()
