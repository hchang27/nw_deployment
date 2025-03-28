import time
from asyncio import sleep

from vuer import Vuer
from vuer.schemas import DefaultScene, ImageBackground
import math
# from go1_gym_deploy.cfgs.lucidsim_depth_cfg import LucidSimRGBCfg
from go1_gym_deploy.modules.base.rs_node import RealSenseCamera
from go1_gym_deploy.modules.lucidsim.transformer_lcm_agent import TransformerLCMAgent
import pyrealsense2 as rs

def list_rgb_modes():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()

    # Start the pipeline to determine connected device and available modes
    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()

    # Fetch all sensor objects from the device (color, depth, infrared)
    sensors = device.query_sensors()

    for sensor in sensors:
        # Check if the sensor is the RGB camera
        if sensor.get_info(rs.camera_info.name) == 'Depth Camera':
            print("RGB Camera Modes:")
            # Get all profiles supported by this sensor
            profiles = sensor.get_stream_profiles()
            for profile in profiles:
                # Check if the profile is a video stream profile
                if profile.stream_type() == rs.stream.color and profile.stream_index() == 0:  # color stream index for RGB is typically 0
                    video_profile = profile.as_video_stream_profile()
                    print(f"  Mode: {video_profile.format()} {video_profile.width()}x{video_profile.height()}@{video_profile.fps()}FPS")

    # Stop the pipeline
    pipeline.stop()
    
def print_fov_of_selected_mode():
    # Desired resolution and frame rate
    desired_width = 424
    desired_height = 240
    desired_fps = 60

    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()

    # Start the pipeline to determine connected device and available modes
    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()

    # Fetch all sensor objects from the device (color, depth, infrared)
    sensors = device.query_sensors()

    for sensor in sensors:
        # Check if the sensor is the RGB camera
        if sensor.get_info(rs.camera_info.name) == 'Depth Camera':
            # Get all profiles supported by this sensor
            profiles = sensor.get_stream_profiles()
            for profile in profiles:
                if profile.stream_type() == rs.stream.color and profile.stream_index() == 0:
                    video_profile = profile.as_video_stream_profile()
                    if (video_profile.width() == desired_width and 
                        video_profile.height() == desired_height and 
                        video_profile.fps() == desired_fps):
                        # Retrieve the intrinsics to get the FOV
                        intrinsics = video_profile.get_intrinsics()
                        h_fov = 2 * math.atan((desired_width / (2 * intrinsics.fx))) * (180 / math.pi)
                        v_fov = 2 * math.atan((desired_height / (2 * intrinsics.fy))) * (180 / math.pi)
                        print(f"Selected Mode FOV: Horizontal {h_fov:.2f}°, Vertical {v_fov:.2f}°")
                        return

    # Stop the pipeline
    pipeline.stop()

# list_rgb_modes()
print_fov_of_selected_mode()
exit()


if __name__ == "__main__":
    app = Vuer(
        queries=dict(
            show_grid=False,
        )
    )
    # mode_id = -11
    # cap.frame_mode = cap.available_modes[mode_id]

    cam_node = RealSenseCamera(fps=60, res=(240, 424))
    key = "rgb"
    cam_node.spin_process(key)
    print("starting camera")
    while (cam_node.frame[key]).sum() < 1:
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
            # frame = lcm_agent.retrieve_vision()[0][0].cpu().permute(1, 2, 0) + 0.5
            # frame = frame.numpy()
            frame = cam_node.frame[key]

            sess.upsert(
                ImageBackground(
                    frame,
                    key="image",
                ),
                to="bgChildren",
            )

            await sleep(0.016)

        cap.close()

    app.run()
