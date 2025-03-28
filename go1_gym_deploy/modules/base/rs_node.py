import os
import subprocess
import threading
import time
from functools import partial
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from typing import Union, Tuple

import numpy as np

from go1_gym_deploy import ON_ORIN

if ON_ORIN:
    import pyrealsense2 as rs
import pyrealsense2 as rs

IMAGE_DATA_TYPE_MAP = {
    "rgb": np.uint8,
    "gbr": np.uint8,
    "depth": np.float32,
    "pointcloud": np.float32,
    "grab_ts": np.float32,
}

print(f"running from process {os.getpid()}")


class RealSenseCamera:
    verbose = False

    @staticmethod
    def RS415(fps: Union[int, float]):
        """Setting the frame rate to be lower than the fps target.

        :param init:
        :type init:
        :param fps:
        :type fps:
        :return:
        :rtype:
        """
        if fps > 60:
            return 90
        elif fps > 30:
            return 60
        elif fps > 15:
            return 30
        else:
            return 15

    def __init__(self, fps: int = 30, res=(480, 640), exposure=None, **kwargs):
        """
        :param fps: Need to be able to accommodate non-standard fps, for robots.
        :param mode: Use ULTRA by default, but set to lower if Orin struggles.
        :param res: Resolution
        :return:
        :rtype:
        """
        self.fps = fps
        self.frame = dict()
        self.buffers = []
        self.exposure = exposure
        # used for pointcloud calculation
        # self.pc = rs.pointcloud()

        self.SHAPE = res

        # save the target frame rate to avoid aliasing
        self.dt = 1 / fps

        print(
            f"""
fps: {fps}
dt: {self.dt}
res: {self.SHAPE}"""
        )

    def __enter__(self):
        # align = rs.align(rs.stream.color)
        self.pipeline = pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.SHAPE[1], self.SHAPE[0], rs.format.rgba8, self.fps)
        # config.enable_stream(rs.stream.depth, self.SHAPE[1], self.SHAPE[0], rs.format.z16, self.fps)
        profile = pipeline.start(config)
        # Options: 0: left pixel, 1: max of left, 2: min of left.
        self.hole_filling_filter = rs.hole_filling_filter(2)
        color_sensor = profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        if self.exposure is not None:
            color_sensor.set_option(rs.option.exposure, self.exposure)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)

        # AE control is important. If it is too dark, the range map will look bad.
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        return self

    def __exit__(self, *_):
        self.pipeline.stop()
        self.release()

    open = __enter__
    close = __exit__

    def grab(self, *keys):
        """Grab the current frame from the camera. Takes in a list of keys.

        Usage:

            with EgoCamera() as zed:
                rgb, depth = zed.get_capture('rgb', 'depth')

        :param keys: One of ['rgb', 'depth']
        :return:
        """
        # use the default depth.
        frames = self.pipeline.wait_for_frames()

        for k in keys:
            if k == "rgb":
                buff = self.rgb_frame = frames.get_color_frame()
                results = buff.get_data()
                results = np.asanyarray(results)
                # haoran: rgba to rgb
                results = results[:, :, :3]
            elif k == "depth":
                buff = self.depth_frame = frames.get_depth_frame()
                buff = self.hole_filling_filter.process(buff)
                results = buff.get_data()
                results = np.asanyarray(results) / 1_000
            elif k == "grab_ts":
                results = time.perf_counter()
                results = np.asanyarray(results)
            elif k == "pointcloud":
                points = self.pc.calculate(self.depth_frame)
                self.pc.map_to(self.rgb_frame)
                results = points.get_vertices()
                results = np.asanyarray(results).view(np.float32)
                results = results.reshape(self.SHAPE + (3,))
            else:
                raise ValueError(f"Unknown key {k}")

            if k not in self.frame:
                self.frame[k] = results
            else:
                np.copyto(self.frame[k], results)

        if len(keys) == 1:
            return self.frame[k]

        return [self.frame[k] for k in keys]

    time = None
    timing_eps = 0.01
    elapsed_time_mean = 0

    def poll(self, *keys):
        """
        Run a never-ending, frame-grabbing loop.

        :param keys:
        :type keys:
        :return:
        :rtype:
        """
        try:
            with self:
                self.time = time.perf_counter()
                while True:
                    self.grab(*keys)
                    # print('grabbed')
                    # elapsed_time = time.perf_counter() - self.time
                    # self.elapsed_time_mean = 0.9 * self.elapsed_time_mean + 0.1 * elapsed_time
                    # pause = max(self.dt - elapsed_time - self.timing_eps, 0.0001)
                    # # if self.dt < elapsed_time:
                    # #     print(f"elapsed time {elapsed_time} is greater than dt {self.dt}")
                    # time.sleep(pause)
                    # self.time = time.perf_counter()
        except KeyboardInterrupt:
            exit()

    @staticmethod
    def create_buffer(shape: Tuple[int], dtype=np.uint8, name="realsense-", create=None):
        size = np.prod(shape) * np.dtype(dtype).itemsize

        if create:
            try:
                shm = SharedMemory(create=True, size=size, name=name)
                print(
                    f"created a shared memory {name} with size {size} from item size "
                    f"{np.dtype(dtype).itemsize} dtype {dtype} shape {shape}"
                )
            except FileExistsError as e:
                shm = SharedMemory(name=name, size=size)
                shm.unlink()
                print(f"remove the shared memory file {name}. Please try again!")
                raise e

        else:
            shm = SharedMemory(name=name, size=size)
            print(
                f"connected to a shared memory {name} with size {size} from item size "
                f"{np.dtype(dtype).itemsize} dtype {dtype} shape {shape}"
            )

        np_buffer = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        return np_buffer, shm

    def spin(self, *keys):
        """Starting the poling thread.

        :param keys:
        :type keys:
        :return:
        :rtype:
        """

        self.process = threading.Thread(target=partial(self.poll, *keys), daemon=True)
        self.process.start()
        return self

    def release(self):
        for b in self.buffers:
            b.unlink()

        self.frame.clear()

    def share_buffers(self, *keys, create=None):
        buffer_creation_error = []

        if "pointcloud" in keys:
            # we add "rgb" and "depth" ahead of the keys passed in.
            keys = {"rgb", "depth", *keys}

        for k in keys:
            dtype = IMAGE_DATA_TYPE_MAP[k]
            # The pointcloud computation requires the depth map
            if k == "depth":
                image_shape = self.SHAPE
            # The pointcloud computation requires the color map
            elif k == 'grab_ts':
                image_shape = (1,)
            elif k == "rgb":
                image_shape = self.SHAPE + (3,)
            elif k == "pointcloud":
                image_shape = self.SHAPE + (3,)
            else:
                raise ValueError(f"Unknown key {k}")

            name = "realsense-" + k
            try:
                np_buffer, shm = self.create_buffer(image_shape, dtype, name, create=create)
                self.buffers.append(shm)
                self.frame[k] = np_buffer
            except FileExistsError:
                buffer_creation_error.append(name)

        else:
            if buffer_creation_error:
                 raise FileExistsError(
                    f"These buffers already exists and have been cleaned up {buffer_creation_error}. Please restart the process!"
                )

        return self

    def spin_process(self, *keys):
        """
        Ensure that the child process is killed when the parent is killed
        reference:
            https://stackoverflow.com/a/322317/1560241

        :param keys:
        :type keys:
        :return:
        :rtype:
        """
        args = f"ps ax | grep {__file__} | grep -v grep | awk '{{print $1}}' | xargs kill"
        subprocess.run(args, shell=True)
        time.sleep(0.0)

        try:
            self.share_buffers(*keys, create=True)
        except FileExistsError:
            # try again, if it errors again, then rerun this script.
            self.share_buffers(*keys, create=True)

        self.process = Process(target=partial(self.poll, *keys), daemon=False)
        self.process.start()
        return self.process


def entry_point():
    from params_proto import ParamsProto, Proto

    class Args(ParamsProto):
        """The arguments for the camera."""

        fps: Union[int, float] = 30
        """Frames per second, can be fractional."""
        keys: Tuple[str] = Proto(["rgb", "depth", "pointcloud"], dtype=lambda s: [k for k in s.split(",") if k])
        """Image Keys, """

    rs_cam = RealSenseCamera(fps=Args.fps)
    rs_cam.share_buffers(*Args.keys, create=False)
    rs_cam.poll(*Args.keys)


if __name__ == "__main__":
    entry_point()
