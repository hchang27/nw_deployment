import pyrealsense2 as rs
import numpy as np
import cv2
import time

def test_specific_configuration():
    # Create pipeline and config objects
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # First, let's try to get a list of connected devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices detected!")
            return
            
        print(f"Found {len(devices)} RealSense device(s)")
        print(f"Device: {devices[0].get_info(rs.camera_info.name)}")
        print(f"Serial Number: {devices[0].get_info(rs.camera_info.serial_number)}")
        
        # Let's check if any other application might be using the device
        print("\nChecking for active device streams...")
        active_sensors = 0
        for sensor in devices[0].query_sensors():
            try:
                # Try to open the sensor with a zero timeout
                # If it's already in use, this will fail
                sensor.open(sensor.get_stream_profiles()[0])
                sensor.close()
                print(f"Sensor {sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else 'Unknown'} is available")
            except Exception as e:
                active_sensors += 1
                print(f"Sensor appears to be in use: {str(e)}")
        
        if active_sensors > 0:
            print(f"WARNING: {active_sensors} sensor(s) appear to be already in use by another application")
        
        # Try configuring with just color stream (similar to your changed config)
        print("\nTrying RGB-only configuration...")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Try to start the pipeline with explicit error handling
        try:
            profile = pipeline.start(config)
            print("RGB pipeline started successfully")
            
            # Get the active profile details
            active_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            print(f"Active profile: {active_profile.width()}x{active_profile.height()} at {active_profile.fps()}fps (format: {active_profile.format()})")
            
            # Get a few frames
            for i in range(3):
                frames = pipeline.wait_for_frames(5000)  # 5 second timeout
                color_frame = frames.get_color_frame()
                if color_frame:
                    print(f"Frame {i+1}: Valid color frame received")
                else:
                    print(f"Frame {i+1}: No valid color frame received")
                time.sleep(0.1)
                
            pipeline.stop()
            print("Test completed successfully")
            
        except Exception as e:
            print(f"Failed to start pipeline: {str(e)}")
            print("\nPossible reasons for failure:")
            print("1. The specific format/resolution/framerate combination is not supported")
            print("2. Another process is currently using the camera")
            print("3. There might be a USB bandwidth issue")
            print("4. The device might have been disconnected")
    
    except Exception as e:
        print(f"Error during configuration test: {str(e)}")
    
    finally:
        # Make sure the pipeline is stopped even if an exception occurred
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    test_specific_configuration()