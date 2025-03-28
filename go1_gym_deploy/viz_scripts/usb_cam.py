import time
from asyncio import sleep

from vuer import Vuer
from vuer.schemas import DefaultScene, ImageBackground

# from go1_gym_deploy.cfgs.lucidsim_depth_cfg import LucidSimRGBCfg
from go1_gym_deploy.modules.base.usb_cam import USBCamera
from go1_gym_deploy.modules.lucidsim.transformer_lcm_agent import TransformerLCMAgent

# device_list = uvc.device_list()
# uid = None
# for device in device_list:
#     if device["name"] == "Global Shutter Camera":
#         uid = device["uid"]
#
# if uid is None:
#     exit("Couldn't Find the camera")
#
# cap = uvc.Capture(uid)

# print("Done!")

if __name__ == "__main__":
    app = Vuer(
        queries=dict(
            show_grid=False,
        )
    )
    # mode_id = -11
    # cap.frame_mode = cap.available_modes[mode_id]

    cam_node = USBCamera()
    cam_node.spin_process()
    print("starting camera")
    while (cam_node.frame["rgb"]).sum() < 1:
        time.sleep(0.01)

    lcm_agent = TransformerLCMAgent(
        imagenet_pipe=False,
        stack_size=7,
        state_estimator=None,
        device="cuda",
        cam_node=cam_node,
    )

    @app.spawn
    async def main(sess):
        sess.set @ DefaultScene()

        while True:
            # rgb = cap.get_frame_robust().bgr
            frame = lcm_agent.retrieve_vision()[0][0].cpu().permute(1, 2, 0) + 0.5
            frame = frame.numpy()
            # frame = cam_node.frame["rgb"]

            sess.upsert(
                ImageBackground(
                    frame,
                    key="image",
                ),
                to="bgChildren",
            )

            await sleep(0.1)

        cap.close()

    app.run()
