import pyrealsense2 as rs
import numpy as np
import cv2
import time

def test_all_supported_modes():
    """
    Test all available modes directly supported by the RealSense camera.
    """
    # Create context to discover devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices connected!")
        return
    
    print(f"Found {len(devices)} RealSense device(s)")
    
    # For each device, test all supported modes
    for device_idx, device in enumerate(devices):
        device_name = device.get_info(rs.camera_info.name)
        serial_number = device.get_info(rs.camera_info.serial_number)
        
        print(f"\n===== Testing Device {device_idx}: {device_name} (S/N: {serial_number}) =====")
        
        # Get all sensors for this device
        sensors = device.query_sensors()
        print(f"Device has {len(sensors)} sensors")
        
        # Iterate through all sensors
        for sensor_idx, sensor in enumerate(sensors):
            sensor_name = sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else f"Sensor {sensor_idx}"
            
            print(f"\n--- Testing {sensor_name} ---")
            
            # Get all stream profiles
            profiles = sensor.get_stream_profiles()
            print(f"Found {len(profiles)} stream profiles")
            
            # Group profiles by stream type
            stream_profiles = {}
            for profile in profiles:
                stream_type = profile.stream_type()
                if stream_type not in stream_profiles:
                    stream_profiles[stream_type] = []
                stream_profiles[stream_type].append(profile)
            
            # Test each type of stream
            for stream_type, profiles in stream_profiles.items():
                stream_name = rs.stream(stream_type).name
                print(f"\n  Stream: {stream_name}")
                
                # Create a set of unique resolutions and frame rates
                unique_modes = set()
                for profile in profiles:
                    video_profile = profile.as_video_stream_profile()
                    w, h = video_profile.width(), video_profile.height()
                    fps = video_profile.fps()
                    format_name = profile.format()
                    unique_modes.add((w, h, fps, format_name))
                
                # Sort modes by resolution and fps
                sorted_modes = sorted(list(unique_modes), key=lambda x: (x[0] * x[1], x[2]))
                
                # Display all modes
                print(f"  Found {len(sorted_modes)} unique modes:")
                for i, (w, h, fps, format_name) in enumerate(sorted_modes):
                    print(f"    {i+1}. {w}x{h} @ {fps}fps - Format: {rs.format(format_name).name}")
                
                # Test a subset of modes (to avoid testing everything which could take too long)
                test_modes = []
                if len(sorted_modes) <= 5:
                    test_modes = sorted_modes
                else:
                    # Select a representative sample
                    test_modes = [sorted_modes[0]]  # Smallest resolution
                    test_modes.append(sorted_modes[-1])  # Largest resolution
                    
                    # Add a few in between
                    step = len(sorted_modes) // 3
                    if step > 0:
                        test_modes.append(sorted_modes[step])
                        test_modes.append(sorted_modes[2*step])
                
                # Test the selected modes
                print(f"\n  Testing {len(test_modes)} sample modes:")
                for i, (w, h, fps, format_name) in enumerate(test_modes):
                    print(f"\n    Testing mode {i+1}: {w}x{h} @ {fps}fps - Format: {rs.format(format_name).name}")
                    test_stream_mode(stream_type, w, h, format_name, fps)
                
def test_stream_mode(stream_type, width, height, format_name, fps):
    """
    Test a specific stream mode.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(stream_type, width, height, format_name, fps)
    
    try:
        # Start the pipeline with the configuration
        start_time = time.time()
        profile = pipeline.start(config)
        startup_time = time.time() - start_time
        print(f"      Pipeline started successfully in {startup_time:.2f} seconds")
        
        # Get a few frames
        frames_received = 0
        start_time = time.time()
        timeout = 5000  # 5 second timeout
        
        for i in range(3):
            try:
                frames = pipeline.wait_for_frames(timeout)
                frame = frames.first_or_default(stream_type)
                if frame:
                    frames_received += 1
                    print(f"      Frame {i+1}: Valid frame received")
                else:
                    print(f"      Frame {i+1}: No valid frame received")
            except Exception as e:
                print(f"      Frame {i+1}: Error waiting for frame: {str(e)}")
            
        # Calculate frame rate
        elapsed = time.time() - start_time
        if elapsed > 0 and frames_received > 0:
            actual_fps = frames_received / elapsed
            print(f"      Actual FPS: {actual_fps:.2f}")
        
        # Success
        print(f"      Mode test completed successfully")
        
    except Exception as e:
        print(f"      Failed to test mode: {str(e)}")
    
    finally:
        # Stop the pipeline
        try:
            pipeline.stop()
            print("      Pipeline stopped")
        except Exception as e:
            print(f"      Error stopping pipeline: {str(e)}")
        
        # Wait a moment to allow resources to be released
        time.sleep(1)

if __name__ == "__main__":
    test_all_supported_modes()