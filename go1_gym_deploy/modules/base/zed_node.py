import os
import shlex
import subprocess
import sys
import threading
import time
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Union, Literal, Tuple

import cv2 as cv
import numpy as np

# image_types = Enum("image_types", "rgb depth gbr")

IMAGE_DATA_TYPE_MAP = {
    "rgb": np.uint8,
    "gbr": np.uint8,
    "depth": np.float32,
    "pointcloud": np.float32,
}

print(f"running from process {os.getpid()}")


class ZedCamera:
    verbose = False

    @staticmethod
    def ZED_X(init, fps: Union[int, float]):
        """Setting the frame rate to be lower than the fps target.

        :param init: ZED Init Parameters
        :type init: sl.InitParameters()
        :param fps:
        :type fps:
        :return:
        :rtype:
        """
        from pyzed import sl

        if fps > 60:
            init.camera_resolution = sl.RESOLUTION.SVGA
            return 120
        elif fps > 30:
            return 60
        elif fps > 15:
            return 30
        else:
            return 15

    def __init__(self, fps: int = 120, mode: str = "ULTRA", res=None, **kwargs):
        """
        :param fps: Need to be able to accommodate non-standard fps, for robots.
        :param mode: Use ULTRA by default, but set to lower if Orin struggles.
        :param res: Resolution
        :return:
        :rtype:
        """
        self.fps = fps
        self.mode = mode
        self.frame = dict()
        self.buffers = []

        from pyzed import sl

        self.RESOLUTION_MAP = {
            # height, width for numpy convention
            sl.RESOLUTION.AUTO: (1200, 1920),
            sl.RESOLUTION.HD1080: (1080, 1920),
            sl.RESOLUTION.HD1200: (1200, 1920),
            sl.RESOLUTION.SVGA: (600, 960),
            # sl.RESOLUTION.HD2K: (0, 0),
            # sl.RESOLUTION.HD720: (0, 0),
            # sl.RESOLUTION.LAST: (0, 0),
            # sl.RESOLUTION.VGA: (0, 0),
        }
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
            depth_minimum_distance=0.1,  # Set the minimum depth perception distance to 15cm
        )
        self.runtime_parameters = sl.RuntimeParameters(
            confidence_threshold=100,
        )

        self.rgb_buff = sl.Mat()
        self.depth_buff = sl.Mat()
        self.pointcloud_buff = sl.Mat()

        # save the target frame rate to avoid aliasing
        self.dt = 1 / fps
        self.init_params.camera_fps = self.ZED_X(self.init_params, fps)
        if res:
            self.init_params.camera_resolution = getattr(sl.RESOLUTION, res)

        self.res = self.init_params.camera_resolution.name

        # we don't need this.
        # resolution = RESOLUTION_MAP[init_params.camera_resolution]

        self.init_params.depth_mode = getattr(sl.DEPTH_MODE, mode)
        self.init_params.depth_stabilization = 0
        self.init_params.sdk_verbose = int(self.verbose)

        # only used for pointclouds
        # init_parameters.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

        self.runtime_parameters.enable_fill_mode = True

        print(
            f"""
fps: {fps}
dt: {self.dt}
mode: {mode}
res: {self.res}"""
        )

    def __enter__(self):
        from pyzed import sl

        print(f"Opening ZED camera in process {os.getpid()}, {self.init_params}")
        err = self.zed.open(self.init_params)
        print(f"camera is now open in {os.getpid()} with error code {err}")
        if err != sl.ERROR_CODE.SUCCESS:
            print("Error {}, exit program".format(err))
            sys.exit()

        return self

    def __exit__(self, *_):
        self.zed.close()
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
        from pyzed import sl

        if self.zed.grab(self.runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            return None

        for k in keys:
            if k == "bgr" or k == "rgb":
                buff = self.rgb_buff
                self.zed.retrieve_image(buff, sl.VIEW.LEFT)
            elif k == "depth":
                buff = self.depth_buff
                self.zed.retrieve_measure(buff, sl.MEASURE.DEPTH)
            elif k == "pointcloud":
                buff = self.pointcloud_buff
                self.zed.retrieve_measure(buff, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            else:
                raise ValueError(f"Unknown key {k}")

            results = buff.get_data()
            # results = np.asanyarray(results)

            if k == "rgb":
                # copy to the existing value
                results = cv.cvtColor(results, cv.COLOR_BGR2RGB)

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
                    elapsed_time = time.perf_counter() - self.time
                    self.elapsed_time_mean = 0.9 * self.elapsed_time_mean + 0.1 * elapsed_time
                    pause = max(self.dt - elapsed_time - self.timing_eps, 0.0001)
                    # if self.dt < elapsed_time:
                    #     print(f"elapsed time {elapsed_time} is greater than dt {self.dt}")
                    time.sleep(pause)
                    self.time = time.perf_counter()
        except KeyboardInterrupt:
            exit()

    @staticmethod
    def create_buffer(shape: Tuple[int], dtype=np.uint8, name="zed", create=None):
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
        self.thread = threading.Thread(target=partial(self.poll, *keys), daemon=True)
        self.thread.start()
        return self

    def release(self):
        for b in self.buffers:
            b.unlink()

        self.frame.clear()

    def __delete__(self, instance):
        instance.release()

    def share_buffers(self, *keys, create=None):
        shape = self.RESOLUTION_MAP[self.init_params.camera_resolution]
        buffer_creation_error = []

        for k in keys:
            dtype = IMAGE_DATA_TYPE_MAP[k]
            if k == "depth":
                image_shape = shape
            elif k == "pointcloud":
                image_shape = shape + (4,)
            else:
                # color images have four dimensions
                image_shape = shape + (3,)

            name = "zed-" + k
            try:
                np_buffer, shm = self.create_buffer(image_shape, dtype, name, create=create)
                self.buffers.append(shm)
                self.frame[k] = np_buffer
            except FileExistsError as e:
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
        import atexit
        import signal
        import sys

        args = f"ps ax | grep {__file__} | grep -v grep | awk '{{print $1}}' | xargs kill"
        p = subprocess.run(args, shell=True)
        time.sleep(0.0)

        try:
            self.share_buffers(*keys, create=True)
        except FileExistsError as e:
            # try again, if it errors again, then rerun this script.
            self.share_buffers(*keys, create=True)

        args = [sys.executable, __file__, "--keys", ",".join(keys), "--fps", str(self.fps), "--mode", self.mode, "--res", self.res]
        print(args)
        p = subprocess.Popen(args)

        def cleanup():
            p.kill()
            print("cleaning up")
            os.killpg(0, signal.SIGKILL)

        atexit.register(cleanup)

        # fuck StereoLabs. does not work in child processes
        # self.thread = Process(target=partial(self.poll, *keys), daemon=False)
        # self.thread.start()
        return p


def entry_point():
    from params_proto import ParamsProto, Proto

    class Args(ParamsProto):
        """The arguments for the camera."""

        fps: Union[int, float] = 60
        """Frames per second, can be fractional."""
        mode: Literal["ULTRA", "NEURAL", "PERFORMANCE", "QUALITY"] = "ULTRA"
        """Depth Inference Mode. Performance is the fastest"""
        res: Literal["SVGA", "HD1200", "HD1080"] = "SVGA"
        """Resolution"""
        keys: Tuple[str] = Proto(["rgb", "depth"], dtype=lambda s: [k for k in s.split(",") if k])
        """Image Keys, """

    zed = ZedCamera(fps=Args.fps, mode=Args.mode, res=Args.res)
    zed.share_buffers(*Args.keys, create=False)
    zed.poll(*Args.keys)


if __name__ == "__main__":
    entry_point()
