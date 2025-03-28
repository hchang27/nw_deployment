import cv2

from go1_gym_deploy.cfgs.lucidsim_depth_cfg import LucidSimRGBCfg
from go1_gym_deploy.modules.base.usb_cam import USBCamera
from go1_gym_deploy.modules.lucidsim.lucid_lcm_agent import LucidLCMAgent

if __name__ == "__main__":
    cam_node = USBCamera()
    cam_node.spin_process()

    lcm_agent = LucidLCMAgent(
        **vars(LucidSimRGBCfg.policy),
        **vars(LucidSimRGBCfg.control),
        **vars(LucidSimRGBCfg.vision),
        **vars(LucidSimRGBCfg.terrain),
        **vars(LucidSimRGBCfg.init_state),
        obs_scales=LucidSimRGBCfg.obs_scales,
        state_estimator=None,
        device="cuda:0",
        cam_node=cam_node,
    )

    for _ in range(30 * 10):
        frame = lcm_agent.retrieve_depth(force=True)
        frame = frame[0].permute(1, 2, 0).cpu().numpy()
        frame = ((frame - (-3)) / (3 - (-3)) * 255).astype("uint8")
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
