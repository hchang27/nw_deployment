"""Frustum Sampling

Sample points from a camera frustum, and then dump into a datafile.
"""
from typing import Tuple

import numpy as np


def frustum_sampling(
        fov: float,
        width: float,
        height: float,
        near: float,
        far: float,
        num_samples=1,
        seed=None,
        **_,
) -> Tuple[np.ndarray]:
    """Sample Camera Frustum

    For rectangular cameras, sample a point between near and far plane, relative to camera transform

    Output in view space: X right, Y up, Z backward

    (Assumed to have the same intrinsics for each sample)


    :param fov: vertical field-of-view, in degrees
    :param width: px
    :param height: py
    :param near: in meters (world units)
    :param far: in meters (world units)
    """
    if seed is not None:
        np.random.seed(seed)

    aspect = width / height
    fov_rad = fov * np.pi / 180
    foh_rad = np.arctan(np.tan(fov_rad / 2) * aspect) * 2

    d = np.random.uniform(near, far, size=num_samples)  # distance in camera coordinates

    y_range = d * np.tan(fov_rad / 2)
    y = np.random.uniform(-y_range, y_range, size=num_samples)

    x_range = d * np.tan(foh_rad / 2)
    x = np.random.uniform(-x_range, x_range, size=num_samples)

    return np.stack([x, y, -d]).T


if __name__ == "__main__":
    from datetime import datetime
    from ml_logger import ML_Logger
    from params_proto import ParamsProto


    class Args(ParamsProto):
        """Collect Random Samples from Camera Frustum
        """
        fov = 42.5
        width = 320
        height = 240
        near = 0.5
        far = 0.9
        num_samples = 825

        seed = 300


    data_logger = ML_Logger(
        root="http://luma01.csail.mit.edu:4000",
        # prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/{datetime.now():%Y%m%d-%H%M%S}",
        # prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/ball-{Args.seed}",
        # prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/ball-train-v8",
        prefix=f"lucidsim/experiments/matia/visuoservoing/ball_gen/ball-test-v9",
    )
    data_logger.job_started(Args=Args.__dict__)
    data_logger.remove("points.pkl")

    print(data_logger.get_dash_url())

    num_x = 40
    num_y = 30

    # # debug-1
    # for i in range(len(points)):
    #     points[i, 0] = 0.01 * i
    #     points[i, 1] = 0.03 * i
    # # debug-2
    # points = frustum_sampling(**Args.__dict__)
    # xs = np.linspace(-0.32, 0.32, 33)
    # ys = np.linspace(-0.24, 0.24, 25)
    # xs, ys = np.meshgrid(xs, ys)

    ## train-v1
    # Args.num_samples = num_x * num_y
    # points = frustum_sampling(**Args.__dict__)
    # xs = np.linspace(-0.32, 0.32, num_x)
    # ys = np.linspace(-0.24, 0.24, num_y)
    # xs, ys = np.meshgrid(xs, ys)
    # 
    # points[:, 0] = xs.flatten()
    # points[:, 1] = ys.flatten()

    # ## test-v1
    Args.num_samples = 15 * 11
    points = frustum_sampling(**Args.__dict__)
    xs = np.linspace(-0.30, 0.30, 15)
    ys = np.linspace(-0.22, 0.22, 11)
    xs, ys = np.meshgrid(xs, ys)
    points[:, 0] = xs.flatten()
    points[:, 1] = ys.flatten()
    points += np.random.uniform(-0.02, 0.02, size=points.shape)

    data_logger.save_pkl(points, "points.pkl", append=False)
    print("saving points")
    print(points[0])
    data_logger.job_completed()
