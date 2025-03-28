import os
import subprocess
import threading
import time
from functools import partial
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple, Union

import cv2
import numpy as np

from go1_gym_deploy import ON_ORIN

if ON_ORIN:
    import uvc

IMAGE_DATA_TYPE_MAP = {
    "rgb": np.uint8,
}

print(f"running from process {os.getpid()}")


class USBCamera:
    verbose = False

    # res -> fps -> mode_id
    mode_mapping = {
        (1080, 1920): {30: -11},
        (720, 1280): {60: 46},
    }

    def __init__(self, res=(720, 1280), resize_shape=(45, 80), fps=60, exposure=None, **kwargs):
        """
        :param fps: Need to be able to accommodate non-standard fps, for robots.
        :param mode: Use ULTRA by default, but set to lower if Orin struggles.
        :param resize_shape: Resize the image to this shape for storage in the buffer
        :param res: Resolution
        :return:
        :rtype:
        """
        self.fps = fps
        self.frame = dict()
        self.buffers = []
        self.exposure = exposure

        self.MODE_SHAPE = res
        self.SHAPE = resize_shape or res

        # save the target frame rate to avoid aliasing
        self.dt = 1 / fps

        print(
            f"""
            fps: {fps}
            dt: {self.dt}
            mode_shape: {self.MODE_SHAPE}
            res: {self.SHAPE}"""
        )

    def __enter__(self):
        device_list = uvc.device_list()

        cam_uid = device_list[0]["uid"]
        self.cap = uvc.Capture(cam_uid)

        if self.cap.name != "Global Shutter Camera":
            print(f"Camera name is not Global Shutter Camera. It is {self.cap.name}")
            exit()

        mode_id = self.mode_mapping[self.MODE_SHAPE][self.fps]
        try:
            mode = self.cap.available_modes[mode_id]
            self.cap.frame_mode = mode
        except uvc.InitError:
            print(f"mode_id {mode_id} not available. Available modes: {self.mode_mapping[self]}")
            exit()

        return self

    def __exit__(self, *_):
        self.cap.close()
        self.release()

    open = __enter__
    close = __exit__

    def grab(self):
        frame = self.cap.get_frame_robust().bgr
        # frame = (np.abs(np.random.randn(*self.SHAPE,3)) * 128).astype(np.uint8)
        frame = cv2.resize(frame, self.SHAPE[::-1], interpolation=cv2.INTER_CUBIC)
        # map to RGB
        self.rgb_frame = frame[:, :, ::-1]
        # result = np.asanyarray(self.rgb_frame)
        np.copyto(self.frame["rgb"], self.rgb_frame)

        return self.frame["rgb"]

    time = None
    timing_eps = 0.01
    elapsed_time_mean = 0

    def poll(self):
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
                    self.grab()
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
    def create_buffer(shape: Tuple[int], dtype=np.uint8, name="usb-", create=None):
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

    def spin(self):
        """Starting the poling thread.

        :param keys:
        :type keys:
        :return:
        :rtype:
        """

        self.process = threading.Thread(target=self.poll, daemon=True)
        self.process.start()
        return self

    def release(self):
        for b in self.buffers:
            b.unlink()

        self.frame.clear()

    def share_buffers(self, create=None):
        buffer_creation_error = []

        dtype = IMAGE_DATA_TYPE_MAP["rgb"]
        image_shape = self.SHAPE + (3,)
        name = "usb-" + "rgb"
        try:
            np_buffer, shm = self.create_buffer(image_shape, dtype, name, create=create)
            self.buffers.append(shm)
            self.frame["rgb"] = np_buffer
        except FileExistsError:
            buffer_creation_error.append(name)

        if buffer_creation_error:
            raise FileExistsError(
                f"These buffers already exists and have been cleaned up {buffer_creation_error}. Please restart the process!"
            )

        return self

    def spin_process(self):
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
            self.share_buffers(create=True)
        except FileExistsError:
            # try again, if it errors again, then rerun this script.
            self.share_buffers(create=True)

        self.process = Process(target=partial(self.poll), daemon=False)
        self.process.start()
        return self.process


def entry_point():
    from params_proto import ParamsProto

    class Args(ParamsProto):
        """The arguments for the camera."""

        fps: Union[int, float] = 30
        """Frames per second, can be fractional."""

    rs_cam = USBCamera(fps=Args.fps)
    rs_cam.share_buffers(create=False)
    rs_cam.poll()


if __name__ == "__main__":
    entry_point()
