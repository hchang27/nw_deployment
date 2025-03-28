import pyrealsense2 as rs
import math

def calculate_fov(intrinsics):
    """
    Calculate the horizontal and vertical field of view (FOV) in degrees.
    """
    hfov = math.degrees(2 * math.atan2(intrinsics.width / 2, intrinsics.fx))
    vfov = math.degrees(2 * math.atan2(intrinsics.height / 2, intrinsics.fy))
    return hfov, vfov

def list_depth_modes():
    # Create a context object. This object owns the handles to all connected realsense devices
    context = rs.context()

    # Get a list of all connected devices
    connected_devices = context.query_devices()
    if len(connected_devices) == 0:
        print("No RealSense devices were found.")
        return

    for dev_idx, device in enumerate(connected_devices):
        device_name = device.get_info(rs.camera_info.name)
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"\nDevice {dev_idx + 1}: {device_name} (Serial Number: {serial_number})")

        # Iterate over all sensors of the device
        for sensor in device.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            # We're interested in the depth sensor
            if "Stereo Module" in sensor_name or "Depth" in sensor_name:
                print(f"  Sensor: {sensor_name}")
                # Get all stream profiles of the depth sensor
                profiles = sensor.get_stream_profiles()
                depth_profiles = [p.as_video_stream_profile() for p in profiles if p.stream_type() == rs.stream.depth]

                unique_modes = set()
                for profile in depth_profiles:
                    width = profile.width()
                    height = profile.height()
                    fps = profile.fps()
                    format = str(profile.format()).split('.')[-1]  # Get format enum name

                    # Get intrinsics to calculate FOV
                    try:
                        intrinsics = profile.get_intrinsics()
                        hfov, vfov = calculate_fov(intrinsics)
                        aspect_ratio = round(width / height, 2)
                    except Exception as e:
                        hfov = vfov = aspect_ratio = 'N/A'

                    mode_info = (
                        f"    Resolution: {width}x{height}, "
                        f"FPS: {fps}, "
                        f"Format: {format}, "
                        f"HFOV: {hfov if hfov == 'N/A' else round(hfov, 2)}°, "
                        f"VFOV: {vfov if vfov == 'N/A' else round(vfov, 2)}°, "
                        f"Aspect Ratio: {aspect_ratio}"
                    )

                    if mode_info not in unique_modes:
                        unique_modes.add(mode_info)

                if unique_modes:
                    print("  Supported Depth Modes:")
                    for mode in sorted(unique_modes):
                        print(mode)
                else:
                    print("    No depth modes found for this sensor.")

if __name__ == "__main__":
    list_depth_modes()
