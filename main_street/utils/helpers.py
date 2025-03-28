import argparse
import copy
import os
import random
from typing import Tuple, List

import numpy as np
import torch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_rotate
from scipy import interpolate
from torchtyping import TensorType


def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None, mask=None):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)
        mask (np.ndarray): mask to apply to the terrain. If provided, will ignore everywhere the mask is 1

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range,
        (
            int(terrain.width * terrain.horizontal_scale / downsampled_scale),
            int(terrain.length * terrain.horizontal_scale / downsampled_scale),
        ),
    )

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind="linear")

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    if mask is not None:
        idxs = np.where(mask == 0)
        terrain.height_field_raw[idxs] += z_upsampled.astype(np.int16)[idxs]
    else:
        terrain.height_field_raw += z_upsampled.astype(np.int16)

    return terrain


def heightmap_from_terrain(vertices, faces, heightmap_width, heightmap_length):
    """
    Create a heightmap from the given trimesh data.

    :param vertices: List of vertices (x, y, z coordinates)
    :param faces: List of faces (indices into the vertex list)
    :param heightmap_width: Width of the heightmap in pixels
    :param heightmap_length: Height of the heightmap in pixels
    :return: 2D numpy array representing the heightmap
    """

    # Initialize the heightmap
    heightmap = np.zeros((heightmap_length, heightmap_width))

    # Initialize a count map to store the number of points per pixel
    count_map = np.zeros((heightmap_length, heightmap_width))

    # Calculate bounds of the mesh
    min_x = min(vertices[:, 0])
    max_x = max(vertices[:, 0])
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])

    # Scale factors for mapping vertices to heightmap pixels
    scale_x = heightmap_width / (max_x - min_x)
    scale_y = heightmap_length / (max_y - min_y)

    # Process each face
    for face in faces:
        # Get the vertices of the face
        v1, v2, v3 = [vertices[i] for i in face]

        # Aggregate vertices into pixels
        for v in [v1, v2, v3]:
            # Map the vertex position to a pixel in the heightmap
            pixel_x = int((v[0] - min_x) * scale_x)
            pixel_y = int((v[1] - min_y) * scale_y)

            # Aggregate the height and count
            if 0 <= pixel_x < heightmap_width and 0 <= pixel_y < heightmap_length:
                heightmap[pixel_y, pixel_x] += v[2]
                count_map[pixel_y, pixel_x] += 1

    # Average the heights
    with np.errstate(divide="ignore", invalid="ignore"):
        heightmap = np.true_divide(heightmap, count_map)
        heightmap[~np.isfinite(heightmap)] = 0  # Replace NaNs and inf with 0

    return heightmap


def create_interpolated_heightmap(vertices, heightmap_width, heightmap_height, horizontal_scale):
    # Extract x, y, and z coordinates from vertices
    points = vertices[:, :2]  # x, y coordinates
    values = vertices[:, 2]  # z coordinates (heights)

    mesh_width_px = int(np.ceil((points[:, 0].max() - points[:, 0].min()) / horizontal_scale))
    mesh_height_px = int(np.ceil((points[:, 1].max() - points[:, 1].min()) / horizontal_scale))

    # Create a grid on which to interpolate
    grid_y, grid_x = np.meshgrid(
        np.linspace(points[:, 1].min(), points[:, 1].max(), mesh_height_px),
        np.linspace(points[:, 0].min(), points[:, 0].max(), mesh_width_px),
    )

    # Interpolate using griddata
    heightmap = interpolate.griddata(points, values, (grid_x, grid_y), method="nearest")

    # pad with zeros to match desired heightmap size
    heightmap = np.pad(
        heightmap, ((0, heightmap_height - heightmap.shape[0]), (0, heightmap_width - heightmap.shape[1])), "constant", constant_values=0
    )

    # Replace NaNs with a default value (e.g., minimum height)
    heightmap = np.nan_to_num(heightmap, nan=np.min(values))

    return heightmap


def create_max_heightmap(vertices, heightmap_width, heightmap_height, horizontal_scale):
    # Extract x, y, and z coordinates from vertices
    points = vertices[:, :2]  # x, y coordinates
    values = vertices[:, 2]  # z coordinates (heights)

    # Discretize points to pixel grid
    pixel_x = np.floor((points[:, 0] - points[:, 0].min()) / horizontal_scale).astype(int)
    pixel_y = np.floor((points[:, 1] - points[:, 1].min()) / horizontal_scale).astype(int)

    # Initialize height map
    heightmap = np.full((heightmap_height, heightmap_width), np.nan)

    # Aggregate heights: For each pixel, choose the maximum height
    for x, y, z in zip(pixel_x, pixel_y, values):
        if x < heightmap_width and y < heightmap_height:
            heightmap[y, x] = max(heightmap[y, x], z) if not np.isnan(heightmap[y, x]) else z

    # Replace NaNs with a default value (e.g., minimum height)
    heightmap = np.nan_to_num(heightmap, nan=np.min(values))

    return heightmap.T


def get_mat(position, rotation=None) -> List:
    from main_street.utils import euler_angles_to_matrix

    # Column-major
    mat = np.array([[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, position[2]], [0, 0, 0, 1]])

    if rotation is not None:
        rotation = torch.tensor(rotation)[None, ...]
        # convert euler to rot matrix

        # gym_to_3js
        extra_rot = torch.tensor([[0.0, 0.0, -1], [-1, 0, 0.0], [0.0, 1.0, 0]])

        # extra_rot = torch.eye(3)

        rot_mat = euler_angles_to_matrix(rotation, convention="XYZ")[0] @ extra_rot
        mat[:3, :3] = rot_mat

        mat_column_major = mat.T

    return mat_column_major.reshape(-1).tolist()


def sample_camera_frustum(horizontal_fov, width, height, near, far) -> Tuple[float, float, float]:
    """
    For rectangular cameras, sample a point between near and far plane, relative to camera transform

    :param horizontal_fov: in degrees
    :param width: px
    :param height: px
    :param near: in meters (world units)
    :param far: in meters (world units)
    """

    vertical_fov = get_vertical_fov(horizontal_fov, width, height)

    horizontal_fov = horizontal_fov * np.pi / 180

    theta = np.random.uniform(-horizontal_fov / 2, horizontal_fov / 2)  # horiz angle
    phi = np.random.uniform(-vertical_fov / 2, vertical_fov / 2)  # vert angle
    r = np.random.uniform(near, far)  # distance

    x = r * np.cos(phi) * np.sin(theta)  # right
    y = r * np.sin(phi)  # up
    z = -r * np.cos(phi) * np.cos(theta)  # backward

    return (x, y, z)


def sample_camera_frustum_batch(
    horizontal_fov: float, width: float, height: float, near: float, far: float, num_samples=1, **kwargs
) -> Tuple[np.ndarray]:
    """
    For rectangular cameras, sample a point between near and far plane, relative to camera transform

    Output in view space: X forward, Y left, Z up

    (Assumed to have the same intrinsics for each sample)


    :param horizontal_fov: in degrees
    :param width: px
    :param height: px
    :param near: in meters (world units)
    :param far: in meters (world units)
    """

    vertical_fov = get_vertical_fov(horizontal_fov, width, height)
    vertical_fov = vertical_fov * np.pi / 180

    horizontal_fov = horizontal_fov * np.pi / 180

    dist = np.random.uniform(near, far, size=(num_samples, 1))  # distance

    y_range = dist * np.tan(vertical_fov / 2)
    y = np.random.uniform(-y_range, y_range, size=(num_samples, 1))

    x_range = dist * np.tan(horizontal_fov / 2)
    x = np.random.uniform(-x_range, x_range, size=(num_samples, 1))

    z = -dist

    x, y, z = -z, -x, -y

    # x = -z # forward
    # y = -x # left
    # z = -y # up

    return x, y, z


def get_vertical_fov(horizontal_fov, width, height):
    """
    For rectangular cameras

    :param horizontal_fov: expected to be in degrees
    :param width: px
    :param height: px
    :return: vertical FoV ( degrees )
    """

    horizontal_fov *= np.pi / 180

    aspect_ratio = width / height

    vertical_fov = 2 * np.arctan(np.tan(horizontal_fov / 2) / aspect_ratio)

    vertical_fov *= 180 / np.pi

    return vertical_fov


def get_horizontal_fov(vertical_fov, width, height):
    """
    For rectangular cameras

    :param vertical_fov: in degrees
    :param width: px
    :param height: px
    :return: horizontal FoV (degrees)
    """
    vertical_fov *= np.pi / 180

    aspect_ratio = width / height

    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * aspect_ratio)

    horizontal_fov *= 180 / np.pi

    return horizontal_fov


def smart_delta_yaw(yaw1, yaw2):
    # Calculate the difference
    delta = yaw2 - yaw1

    # Adjust differences to find the shortest path
    delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi

    return delta


def spherical_to_cartesian(spherical_coordinates):
    """
    spherical_coordinates: (num_envs, 3) tensor
    Output: (num_envs, 3) tensor in Cartesian coordinates.
    """

    r = spherical_coordinates[:, 0]
    theta = spherical_coordinates[:, 1]
    phi = spherical_coordinates[:, 2]

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    return torch.stack((x, y, z), dim=-1)


def spherical_to_cartesian_velocity(v_spherical, coords_spherical):
    """
    Convert spherical velocities to cartesian velocities

    Parameters:
    v_spherical: Spherical velocity components tensor of shape (n, 3) ([vr, vtheta, vphi])
    coords_spherical: Spherical coordinates tensor of shape (n, 3) ([r, theta, phi])

    Returns:
    v_cartesian: Cartesian velocity components tensor of shape (n, 3) ([vx, vy, vz])
    """

    # Extract the individual components
    vr, vtheta, vphi = v_spherical[:, 0], v_spherical[:, 1], v_spherical[:, 2]
    r, theta, phi = (
        coords_spherical[:, 0],
        coords_spherical[:, 1],
        coords_spherical[:, 2],
    )

    # Calculate the trigonometric terms
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    # Calculate the Cartesian velocity components
    vx = vr * sin_theta * cos_phi - r * cos_theta * cos_phi * vtheta - r * sin_theta * sin_phi * vphi
    vy = vr * sin_theta * sin_phi - r * cos_theta * sin_phi * vtheta + r * sin_theta * cos_phi * vphi
    vz = vr * cos_theta + r * sin_theta * vtheta

    # Concatenate the components into an (n, 3) tensor
    v_cartesian = torch.stack([vx, vy, vz], dim=1)

    return v_cartesian


def phi_to_cartesian_velocity(v_phi, coords_spherical):
    """
    Convert spherical velocities to cartesian velocities

    Parameters:
    v_theta: Spherical velocity components tensor of shape (n, 1) ([vtheta])
    coords_spherical: Spherical coordinates tensor of shape (n, 3) ([r, theta, phi])

    Returns:
    v_cartesian: Cartesian velocity components tensor of shape (n, 3) ([vx, vy, vz])
    """
    v_spherical = torch.stack([torch.zeros_like(v_phi), torch.zeros_like(v_phi), v_phi], dim=1)
    return spherical_to_cartesian_velocity(v_spherical, coords_spherical)


def project_point(p, cam_pos, cam_target, cam_properties):
    # Calculate fx and fy
    fx = cam_properties.width / (2 * np.tan((cam_properties.horizontal_fov * np.pi / 180) / 2))
    fy = fx
    cx, cy = cam_properties.width / 2, cam_properties.height / 2

    intrinsics = torch.tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        device=p.device,
    ).float()

    # Recompute up direction
    forward = cam_target - cam_pos
    cam_z = forward
    cam_z /= torch.norm(cam_z)

    up = torch.tensor([0.0, 0.0, 1.0], device=p.device)
    cam_x = torch.cross(forward, up)  # because x goes to the right in the image
    cam_x /= torch.norm(cam_x)

    cam_y = torch.cross(forward, cam_x)
    cam_y /= torch.norm(cam_y)

    # Compute camera rotation matrix
    cam_rotation_matrix = torch.stack([cam_x, cam_y, cam_z], dim=-1)  # cam to world

    point_cam_frame = torch.matmul(cam_rotation_matrix.T, p - cam_pos)  # + cam_pos # + cam_pos

    # 3D point in camera frame
    x, y, z = point_cam_frame

    # Early exit if z is outside the range of the camera
    if z < cam_properties.near_plane or z > cam_properties.far_plane:
        return None

    # Perspective division
    x /= z
    y /= z

    # Project onto image
    u, v, _ = intrinsics @ torch.tensor([[x, y, 1.0]], device=p.device).T

    # Check if u, v are within the image bounds before returning
    if 0 <= u < cam_properties.width and 0 <= v < cam_properties.height:
        return int(u), int(v)
    else:
        return None


def euler_from_quaternion(quat_angle: TensorType["batch", 4]) -> Tuple[TensorType["batch", 1]]:  # noqa: F821
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def extract_local_roll_pitch(base_quat: TensorType["batch", 4]) -> Tuple[TensorType["batch", 1]]:  # noqa: F821
    """
    Extract roll, pitch, yaw from the base quaternion
    """

    forward_to_robot = torch.tensor([1.0, 0.0, 0.0], device=base_quat.device).repeat(base_quat.shape[0], 1)
    forward_to_world = quat_rotate(base_quat, forward_to_robot)

    pitch = -torch.asin(forward_to_world[:, 2])

    up_to_robot = torch.tensor([0.0, 0.0, 1.0], device=base_quat.device).repeat(base_quat.shape[0], 1)
    up_to_world = quat_rotate(base_quat, up_to_robot)

    roll = -(np.pi / 2 - torch.acos(up_to_world[:, 1]))

    return roll, pitch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    print(">>> updating the environment configuration from ", env_cfg)

    # seed
    if env_cfg is not None:
        if args.use_camera:
            env_cfg.depth.use_camera = args.use_camera

        if env_cfg.depth.use_camera and args.headless:  # set camera specific parameters
            env_cfg.env.num_envs = env_cfg.depth.camera_num_envs
            env_cfg.terrain.num_rows = env_cfg.depth.camera_terrain_num_rows
            env_cfg.terrain.num_cols = env_cfg.depth.camera_terrain_num_cols
            env_cfg.terrain.max_error = env_cfg.terrain.max_error_camera
            env_cfg.terrain.horizontal_scale = env_cfg.terrain.horizontal_scale_camera
            env_cfg.terrain.simplify_grid = True
            env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.2
            env_cfg.terrain.terrain_dict["parkour_flat"] = 0.05
            env_cfg.terrain.terrain_dict["parkour_gap"] = 0.2
            env_cfg.terrain.terrain_dict["parkour_step"] = 0.2
            env_cfg.terrain.terrain_dict["demo"] = 0.15
            env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())

        if env_cfg.depth.use_camera:
            env_cfg.terrain.y_range = [-0.1, 0.1]

        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.task_both:
            env_cfg.env.task_both = args.task_both
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols
        if args.delay:
            env_cfg.domain_rand.action_delay = args.delay
        # if not args.delay and not args.resume and not args.use_camera and args.headless:  # if train from scratch
        if not args.delay and not args.use_camera and args.headless:  # if train from scratch
            env_cfg.domain_rand.action_delay = True
            env_cfg.domain_rand.action_curr_step = env_cfg.domain_rand.action_curr_step_scratch
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.use_camera:
            cfg_train.depth_encoder.if_depth = args.use_camera
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
            # if args.resume:
            #     cfg_train.runner.resume = args.resume
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        # if args.checkpoint is not None:
        #     cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "go1",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--device", "type": str, "default": "cuda:0", "help": "Device for sim, rl, and graphics"},
        {"name": "--rows", "type": int, "help": "num_rows."},
        {"name": "--cols", "type": int, "help": "num_cols"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str, "default": "parkour_new", "help": "run folder name."},
        {"name": "--teacher", "type": str, "help": "Name of the teacher policy to use when distilling"},
        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--resumeid", "type": str, "help": "exptid"},
        {"name": "--daggerid", "type": str, "help": "name of dagger run"},
        {"name": "--use_camera", "action": "store_true", "default": False, "help": "render camera for distillation"},
        {"name": "--mask_obs", "action": "store_true", "default": False, "help": "Mask observation when playing"},
        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Load jit script when playing"},
        {"name": "--use_latent", "action": "store_true", "default": False, "help": "Load depth latent when playing"},
        {"name": "--draw", "action": "store_true", "default": False, "help": "draw debug plot when playing"},
        {"name": "--save", "action": "store_true", "default": False, "help": "save data for evaluation"},
        {"name": "--task_both", "action": "store_true", "default": False, "help": "Both climbing and hitting policies"},
        {"name": "--nodelay", "action": "store_true", "default": False, "help": "Add action delay"},
        {"name": "--delay", "action": "store_true", "default": False, "help": "Add action delay"},
        {"name": "--hitid", "type": str, "default": None, "help": "exptid fot hitting policy"},
        {"name": "--web", "action": "store_true", "default": False, "help": "if use web viewer"},
        {"name": "--no_wandb", "action": "store_true", "default": False, "help": "no wandb"},
    ]
    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path, name):
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name + ".pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


# overide gymutil
def parse_device_str(device_str):
    # defaults
    device = "cpu"
    device_id = 0

    if device_str == "cpu" or device_str == "cuda":
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(":")
        assert len(device_args) == 2 and device_args[0] == "cuda", f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id


def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument("--headless", action="store_true", help="Run headless without creating a viewer window")
    if no_graphics:
        parser.add_argument(
            "--nographics",
            action="store_true",
            help="Disable graphics context creation, no viewer window is created, and no headless rendering is available",
        )
    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Physics Device in PyTorch-like syntax")
    parser.add_argument("--pipeline", type=str, default="gpu", help="Tensor API pipeline (cpu/gpu)")
    parser.add_argument("--graphics_device_id", type=int, default=0, help="Graphics Device ID")

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument("--flex", action="store_true", help="Use FleX for physics")
    physics_group.add_argument("--physx", action="store_true", help="Use PhysX for physics")

    parser.add_argument("--num_threads", type=int, default=0, help="Number of cores used by PhysX")
    parser.add_argument("--subscenes", type=int, default=0, help="Number of PhysX subscenes to simulate in parallel")
    parser.add_argument("--slices", type=int, help="Number of client threads that process env slices")

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    print(">>>>> args.device", args.device)

    if args.device is not None:
        args.sim_device = args.device
        args.rl_device = args.device

        print(">>>>> args.device", args.device)
        print(">>>>> args.sim_device", args.sim_device)

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert pipeline == "cpu" or pipeline in ("gpu", "cuda"), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = pipeline in ("gpu", "cuda")

    if args.sim_device_type != "cuda" and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = "cuda:0"
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if args.sim_device_type != "cuda" and pipeline == "gpu":
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = "CPU"
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = args.sim_device_type == "cuda"

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args
