import random

import numpy as np
import open3d as o3d
import pyfqmr
from isaacgym import terrain_utils
from pydelatin import Delatin
from scipy.ndimage import binary_dilation

from main_street.envs.base.legged_robot_config import LeggedRobotCfg
from main_street.utils.helpers import (
    create_interpolated_heightmap,
    random_uniform_terrain,
)


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[: i + 1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.smooth_height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.mesh_src is not None:
            self.from_src()
            # set heightsamples
            if not self.cfg.mesh_from_heightmap:
                return
        elif cfg.heightmap_src is not None:
            self.height_field_raw = cfg.heightmap_src
            self.terrain_type[:, :] = 22  # note: even for flat ground we'll have it as that
        elif cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()

        self.heightsamples = self.height_field_raw

        if self.type == "trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(
                    self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, self.cfg.slope_treshold
                )
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width * 2 + 1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(
                        target_count=int(0.05 * self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10
                    )

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(
                    self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error
                )
            print(f"Created {self.vertices.shape[0]} vertices")
            print(f"Created {self.triangles.shape[0]} triangles")

    def from_src(self, vertices=None, triangles=None):
        mesh = o3d.io.read_triangle_mesh(self.cfg.mesh_src)

        print("mesh source:", self.cfg.mesh_src)

        mesh_vertices = np.array(mesh.vertices)
        mesh_triangles = np.array(mesh.triangles)

        # point_cloud = mesh.sample_points_poisson_disk(self.cfg.n_sample_pcd)
        print(f"Sampling {self.cfg.n_sample_pcd} points from mesh")
        point_cloud = mesh.sample_points_uniformly(self.cfg.n_sample_pcd)
        pcd_vertices = np.array(point_cloud.points)

        if self.cfg.mesh_tf_rot is not None:
            rotation = mesh.get_rotation_matrix_from_xyz(np.deg2rad(self.cfg.mesh_tf_rot)).astype(np.float32)
            mesh_vertices = mesh_vertices @ rotation.T
            pcd_vertices = pcd_vertices @ rotation.T

            # rotate 90 degrees around the GLOBAL z axis
            # extra_rotation = mesh.get_rotation_matrix_from_xyz(np.deg2rad([0, 0, -90])).astype(np.float32)
            # mesh_vertices = mesh_vertices @ extra_rotation.T
            # pcd_vertices = pcd_vertices @ extra_rotation.T

        if self.cfg.mesh_tf_pos is not None:
            mesh_vertices = mesh_vertices + np.array(self.cfg.mesh_tf_pos)
            pcd_vertices = pcd_vertices + np.array(self.cfg.mesh_tf_pos)

        mesh_vertices = mesh_vertices.astype(np.float32)
        mesh_triangles = mesh_triangles.astype(np.uint32)
        pcd_vertices = pcd_vertices.astype(np.float32)

        # FIXME: make actual mask
        self.x_edge_mask = np.zeros((self.tot_rows, self.tot_cols), dtype=bool)

        self.bottom = np.min(mesh_vertices[:, 2])

        mesh_center = np.mean(mesh_vertices, axis=0)
        mesh_length = np.max(mesh_vertices[:, 0]) - np.min(mesh_vertices[:, 0])
        mesh_width = np.max(mesh_vertices[:, 1]) - np.min(mesh_vertices[:, 1])
        mesh_height = np.max(mesh_vertices[:, 2]) - np.min(mesh_vertices[:, 2])

        assert mesh_length <= self.env_length, "Mesh length {} is larger than environment length {}".format(mesh_length, self.env_length)
        assert mesh_width <= self.env_width, "Mesh width {} is larger than environment width {}".format(mesh_width, self.env_width)
        if self.cfg.simplify_grid:
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(mesh_vertices, mesh_triangles)
            mesh_simplifier.simplify_mesh(
                target_count=int(0.05 * mesh_triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10
            )

            mesh_vertices, mesh_triangles, normals = mesh_simplifier.getMesh()
            mesh_vertices = mesh_vertices.astype(np.float32)
            mesh_triangles = mesh_triangles.astype(np.uint32)

        # make mesh corner start at 0
        self.corner_shift = np.min(mesh_vertices[:, :-1], axis=0)

        mesh_vertices[:, :-1] -= self.corner_shift
        pcd_vertices[:, :-1] -= self.corner_shift

        all_vertices = []
        all_triangles = []

        mesh_heightmap = create_interpolated_heightmap(
            pcd_vertices,
            int(self.env_length / self.cfg.horizontal_scale),
            int(self.env_width / self.cfg.horizontal_scale),
            self.cfg.horizontal_scale,
        )
        # mesh_heightmap = create_max_heightmap(
        #     mesh_vertices,
        #     int(self.env_length / self.cfg.horizontal_scale),
        #     int(self.env_width / self.cfg.horizontal_scale),
        #     self.cfg.horizontal_scale,
        # )

        border_shift = np.array([self.cfg.border_size, self.cfg.border_size, 0])

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                env_origin_x = i * self.env_length + 1.0
                env_origin_y = (j + 0.5) * self.env_width

                # if self.cfg.origin_zero_z:
                env_origin_z = 0

                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

                # TODO: set a real id number
                self.terrain_type[i, j] = 17 if self.cfg.flat_mask else 22

                # TODO: set goals
                if self.num_goals > 0:
                    self.goals[i, j, :, :] = np.zeros((self.num_goals, 3))

                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y] = (mesh_heightmap / self.cfg.vertical_scale).astype(np.int16)

                # all_vertices.append(border_shift + mesh_vertices + np.array([env_origin_x, env_origin_y, env_origin_z]))
                # all_vertices.append(mesh_vertices + np.array([i * self.env_length, j * self.env_width, 0]))
                all_vertices.append(border_shift + mesh_vertices + np.array([i * self.env_length, j * self.env_width, 0]))
                all_triangles.append(mesh_triangles + (j * self.cfg.num_rows + i) * mesh_vertices.shape[0])

        self.heightsamples = self.height_field_raw

        # ground_z = self.bottom - 0.01
        ground_z = 0

        # Flip order to be CCW
        # ground_vertices = np.array(
        #     [
        #         [-self.cfg.border_size, -self.cfg.border_size, ground_z],
        #         [-self.cfg.border_size, self.cfg.border_size + self.env_width * self.cfg.num_cols, ground_z],
        #         [
        #             self.cfg.border_size + self.env_length * self.cfg.num_rows,
        #             self.cfg.border_size + self.env_width * self.cfg.num_cols,
        #             ground_z,
        #         ],
        #         [self.cfg.border_size + self.env_length * self.cfg.num_rows, -self.cfg.border_size, ground_z],
        #     ]
        # )[::-1]

        ground_vertices = np.array(
            [
                [0, 0, ground_z],
                [0, 2 * self.cfg.border_size + self.env_width * self.cfg.num_cols, ground_z],
                [
                    2 * self.cfg.border_size + self.env_length * self.cfg.num_rows,
                    2 * self.cfg.border_size + self.env_width * self.cfg.num_cols,
                    ground_z,
                ],
                [2 * self.cfg.border_size + self.env_length * self.cfg.num_rows, 0, ground_z],
            ]
        )[::-1]

        ground_triangles = len(mesh_vertices) * self.cfg.num_rows * self.cfg.num_cols + np.array([[0, 1, 2], [2, 3, 0]])

        all_vertices.append(ground_vertices)
        all_triangles.append(ground_triangles)

        self.vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
        self.triangles = np.concatenate(all_triangles, axis=0).astype(np.uint32)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    difficulty = i / (self.cfg.num_rows - 1)
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1, mask=None):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        random_uniform_terrain(
            terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale, mask=mask
        )

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        if hasattr(self.cfg, "difficulty"):
            difficulty = self.cfg.difficulty

        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0] / 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            idx = 2
            if choice < self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice < self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.0)
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.0
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2 * difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(
                terrain,
                stone_size=1.5 - 0.2 * difficulty,
                stone_distance=0.0 + 0.4 * difficulty,
                max_height=0.2 * difficulty,
                platform_size=1.2,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0.0, 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty)
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1.0, step_height=height, platform_size=3.0)
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1 + 0.3 * difficulty]  # offset to stone_len
            y_range = [0.2, 0.3 + 0.1 * difficulty]
            stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
            incline_height = 0.25 * difficulty
            last_incline_height = incline_height + 0.1 - 0.1 * difficulty
            parkour_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                x_range=x_range,
                y_range=y_range,
                incline_height=incline_height,
                stone_len=stone_len,
                stone_width=1.0,
                last_incline_height=last_incline_height,
                pad_height=0,
                pit_depth=[0.2, 1],
            )
            idx = 15
            # terrain.height_field_raw[:] = 0
            # use the s
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16

            if hasattr(self.cfg, "half_valid_width"):
                half_valid_width = self.cfg.half_valid_width
            else:
                half_valid_width = [0.4, 0.8]

            hurdle_mask = parkour_hurdle_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                stone_len=0.1 + 0.3 * difficulty,
                hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.25 * difficulty],
                pad_height=0,
                x_range=[1.2, 2.2],
                y_range=self.cfg.y_range,
                half_valid_width=half_valid_width,
            )
            terrain.smooth_height_field_raw = terrain.height_field_raw.copy()
            if hasattr(self.cfg, "no_roughness") and self.cfg.no_roughness:
                pass
            elif hasattr(self.cfg, "flat_roughness_only") and self.cfg.flat_roughness_only:
                self.add_roughness(terrain, mask=hurdle_mask)
            else:
                self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            if hasattr(self.cfg, "half_valid_width"):
                half_valid_width = self.cfg.half_valid_width
                idx = 16
            else:
                half_valid_width = [0.45, 1.0]
                idx = 17

            parkour_hurdle_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                stone_len=0.1 + 0.3 * difficulty,
                hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                pad_height=0,
                x_range=self.cfg.x_range,
                y_range=self.cfg.y_range,
                half_valid_width=half_valid_width,
                flat=True,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                step_height=0.1 + 0.35 * difficulty,
                x_range=[0.3, 1.5],
                y_range=self.cfg.y_range,
                half_valid_width=[0.5, 1],
                pad_height=0,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            if hasattr(self.cfg, "half_valid_width"):
                half_valid_width = self.cfg.half_valid_width
            else:
                half_valid_width = [0.45, 1.0]

            idx = 19
            parkour_gap_terrain(
                terrain,
                num_gaps=self.num_goals - 2,
                gap_size=0.1 + 0.7 * difficulty,
                gap_depth=[0.2, 1],
                pad_height=0,
                x_range=[0.8, 1.5],
                y_range=self.cfg.y_range,
                half_valid_width=half_valid_width,
            )
            terrain.smooth_height_field_raw = terrain.height_field_raw.copy()
            self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain)
        elif choice < self.proportions[20]:
            # stairs terrain
            idx = 21

            architecture_staircase_terrain(
                terrain,
                num_goals=self.num_goals - 2,
                difficulty=difficulty,
                half_valid_width=[self.env_width / 4, self.env_width / 2],
                # add_walls=walls,
            )

            # difficulty = 0
            #
            # walls = np.random.choice([True, False], p=[0.5, 0.5])
            # flat_ending = np.random.choice([True, False], p=[0.5, 0.5])
            #
            # staircase_terrain(
            #     terrain,
            #     num_goals=self.num_goals - 2,
            #     # x_range=[0.25, 0.3 + 0.5 * (1 - difficulty)],
            #     x_range=np.array([0.55, 0.70]) - 0.45 * difficulty,
            #     y_range=self.cfg.y_range,  # not used
            #     half_valid_width=[self.env_width / 4, self.env_width / 2],
            #     stair_height=0.2 * difficulty + 0.05,
            #     pad_height=0,
            #     add_walls=walls,  # walls,
            #     flat_ending=flat_ending,
            # )
            terrain.smooth_height_field_raw = terrain.height_field_raw.copy()
            self.add_roughness(terrain)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
        if hasattr(terrain, "smooth_height_field_raw"):
            self.smooth_height_field_raw[start_x:end_x, start_y:end_y] = terrain.smooth_height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 0.5) / terrain.horizontal_scale)  # within 1 meter square range
        x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx

        if self.num_goals > 0:
            self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]

        # self.env_slope_vec[i, j] = terrain.slope_vector

    def resample_hurdle_goals(self, i, j, old_goals, noise):
        """
        half valid width depends on flat or hurdle
        """

        raw_old_goals = old_goals - [i * self.env_length, j * self.env_width, 0]
        num_goals = raw_old_goals.shape[0]

        new_goals = raw_old_goals.copy()

        for k in range(1, num_goals - 1):
            new_goals[k, 1] += np.random.uniform(-noise, noise)

        return new_goals


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0


def gap_parkour_terrain(terrain, difficulty, platform_size=2.0):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -400
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty  # 10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)


def parkour_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    x_range=[1.8, 1.9],
    y_range=[0.0, 0.1],
    z_range=[-0.2, 0.2],
    stone_len=1.0,
    stone_width=0.6,
    pad_width=0.1,
    pad_height=0.5,
    incline_height=0.1,
    last_incline_height=0.6,
    last_stone_len=1.6,
    pit_depth=[0.5, 1.0],
):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones + 2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)

    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0

    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[
                dis_x - last_stone_len // 2 : dis_x + last_stone_len // 2, dis_y - stone_width // 2 : dis_y + stone_width // 2
            ] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2, dis_y - stone_width // 2 : dis_y + stone_width // 2
            ] = heights.astype(int) + dis_z

        goals[i + 1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_gap_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_gaps=8,
    gap_size=0.3,
    x_range=[1.6, 2.4],
    y_range=[-1.2, 1.2],
    half_valid_width=[0.6, 1.2],
    gap_depth=-200,
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_gaps + 2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)

    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth

    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2,
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x - gap_size // 2 : dis_x + gap_size // 2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = gap_depth

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_hurdle_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    stone_len=0.3,
    x_range=[1.5, 2.4],
    y_range=[-0.4, 0.4],
    half_valid_width=[0.4, 0.8],
    hurdle_height_range=[0.2, 0.3],
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x

    mask = np.zeros_like(terrain.height_field_raw)

    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2,] = np.random.randint(
                hurdle_height_min, hurdle_height_max
            )
            mask[dis_x - stone_len // 2 : dis_x + stone_len // 2,] = 1

            terrain.height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2, : mid_y + rand_y - half_valid_width] = 0
            mask[dis_x - stone_len // 2 : dis_x + stone_len // 2, : mid_y + rand_y - half_valid_width] = 0

            terrain.height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2, mid_y + rand_y + half_valid_width :] = 0
            mask[dis_x - stone_len // 2 : dis_x + stone_len // 2, mid_y + rand_y + half_valid_width :] = 0

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

    return mask


def architecture_staircase_terrain(
    terrain,
    num_goals,
    difficulty,
    half_valid_width,
    platform_len=2.5,
):
    height_range = [0.05, 0.21]
    valid_lengths = [0.3, 0.4, 0.5, 0.6]
    x_noise = 0.05
    num_steps = [7, 8, 9, 10, 11, 12, 13, 14]

    mid_y = terrain.length // 2

    goals = np.zeros((num_goals + 2, 2))

    platform_len_scaled = round(platform_len / terrain.horizontal_scale)
    terrain.height_field_raw[0:platform_len_scaled, :] = 0

    dis_x = platform_len_scaled
    current_height = 0
    goals[0] = [platform_len_scaled - round(1 / terrain.horizontal_scale), mid_y]

    height = height_range[0] + (height_range[1] - height_range[0]) * difficulty

    # height_min = height_range[0] + (height_range[1] - height_range[0]) * difficulty
    # height_max = min(height_range[1], height_min + 0.1)
    #
    # height = np.random.uniform(height_min, height_max)

    num_steps = np.random.choice(num_steps)

    length = np.random.choice(valid_lengths)

    length_scaled = round(length / terrain.horizontal_scale)
    current_height = 0

    stair_height_scaled = round(height / terrain.vertical_scale)
    dis_x_min = dis_x = round(2 / terrain.horizontal_scale)

    stair_xs = []

    for step in range(num_steps):
        # rand_x = np.random.uniform(-x_noise, x_noise)

        current_height += stair_height_scaled
        terrain.height_field_raw[dis_x : dis_x + length_scaled, :] = current_height
        stair_xs.append(dis_x + length_scaled // 2)

        dis_x += length_scaled

    # platform
    terrain.height_field_raw[dis_x:, :] = current_height
    half_valid_width_scaled = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    # evenly from dis_x_min to dis_x + 2
    for i in range(1, num_goals + 1):
        x = dis_x_min + (dis_x - dis_x_min) * i // (num_goals + 1)

        y_noise = np.random.uniform(-half_valid_width_scaled // 3, half_valid_width_scaled // 3)

        y = terrain.length // 2 + y_noise
        goals[i] = [x, y]

    final_goal_x = int(dis_x + (1.5 / terrain.horizontal_scale))
    goals[-1] = [final_goal_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale
    return terrain


def staircase_terrain(
    terrain,
    total_height=4.0,
    num_goals=8,
    platform_len=2.5,
    platform_height=0.0,
    x_range=[0.2, 0.4],
    y_range=[-0.15, 0.15],
    half_valid_width=[0.45, 0.5],
    stair_height=0.2,
    pad_width=0.1,
    pad_height=0.5,
    add_walls=True,
    flat_ending=True,
):
    # Calculate the total number of stairs needed to reach the desired height
    num_stairs = int(np.ceil(total_height / stair_height)) * 2  # up and down
    goals = np.zeros((num_goals + 2, 2))
    mid_y = terrain.length // 2

    # Scale adjustments
    dis_x_min = round((x_range[0] + stair_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + stair_height) / terrain.horizontal_scale)
    stair_height_scaled = round(stair_height / terrain.vertical_scale)
    half_valid_width_scaled = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    # Platform settings
    platform_len_scaled = round(platform_len / terrain.horizontal_scale)
    platform_height_scaled = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_scaled, :] = platform_height_scaled

    dis_x = platform_len_scaled
    current_height = 0
    goals[0] = [platform_len_scaled - round(1 / terrain.horizontal_scale), mid_y]

    stair_xs = []

    # Construct the stairs
    for i in range(num_stairs):
        rand_x = np.random.randint(dis_x_min, dis_x_max)

        # Ascend first half, then descend
        if i < num_stairs // 2:
            current_height += stair_height_scaled
        else:
            if not flat_ending:
                current_height -= stair_height_scaled

        terrain.height_field_raw[dis_x : dis_x + rand_x, mid_y - half_valid_width_scaled : mid_y + half_valid_width_scaled] = current_height
        stair_xs.append(dis_x + rand_x // 2)
        dis_x += rand_x

    max_stair_x = max(stair_xs, key=lambda x: x if x < terrain.width else 0)
    last_stair_idx = stair_xs.index(max_stair_x) - 2  # don't go too close to the end

    # Set the goals
    step_per_goal = (last_stair_idx) // num_goals
    for i in range(1, num_goals + 1):
        stair_index = i * step_per_goal - 1 if i * step_per_goal - 1 < num_stairs else num_stairs - 1
        y_noise = np.random.randint(-half_valid_width_scaled, half_valid_width_scaled) // 4
        goals[i] = [stair_xs[stair_index], mid_y + y_noise]

    goals[-1] = [stair_xs[last_stair_idx], mid_y]

    # store each step's (x,y,z) center and (width / length / height) so we can reconstruct later
    # stair_boxes = []
    # for step in len(stair_xs):
    #     x = stair_xs[step]
    #     y = mid_y
    #     z = stair_height * (step + 1) / 2
    #     width = 1
    #     length = 1
    #     height = stair_height
    #     stair_boxes.append([x, y, z, width, length, height])

    terrain.goals = goals * terrain.horizontal_scale

    # Pad edges
    pad_width_scaled = int(pad_width // terrain.horizontal_scale)
    pad_height_scaled = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_scaled] = pad_height_scaled
    terrain.height_field_raw[:, -pad_width_scaled:] = pad_height_scaled
    terrain.height_field_raw[:pad_width_scaled, :] = pad_height_scaled
    terrain.height_field_raw[-pad_width_scaled:, :] = pad_height_scaled

    if add_walls:
        # spikes

        wall_height = 1.25 * total_height
        wall_height_scaled = round(wall_height / terrain.vertical_scale)

        # Calculate the y-coordinates for the side walls
        terrain_width = terrain.height_field_raw.shape[1]
        mid_y = terrain.length // 2
        half_valid_width_scaled = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

        left_wall_y = mid_y - half_valid_width_scaled
        right_wall_y = mid_y + half_valid_width_scaled

        # Create walls directly next to the stairs
        left_side_shape = terrain.height_field_raw[:, :left_wall_y].shape
        right_side_shape = terrain.height_field_raw[:, right_wall_y:].shape

        left_spikes = np.random.randint(0, wall_height_scaled, size=left_side_shape)
        right_spikes = np.random.randint(0, wall_height_scaled, size=right_side_shape)

        terrain.height_field_raw[:, :left_wall_y] = left_spikes
        terrain.height_field_raw[:, right_wall_y:] = right_spikes

        # terrain.height_field_raw[:, :left_wall_y] = wall_height_scaled
        # terrain.height_field_raw[:, right_wall_y:] = wall_height_scaled

    return terrain


def parkour_step_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    #    x_range=[1.5, 2.4],
    x_range=[0.2, 0.4],
    y_range=[-0.15, 0.15],
    half_valid_width=[0.45, 0.5],
    step_height=0.2,
    pad_width=0.1,
    pad_height=0.5,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round((x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x : dis_x + rand_x,] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = 0

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2

    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + hurdle_depth, round(mid_y - hurdle_width / 2) : round(mid_y + hurdle_width / 2)
    ] = hurdle_height

    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length + first_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + first_step_depth, round(mid_y - first_step_width / 2) : round(mid_y + first_step_width / 2)
    ] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length + second_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + second_step_depth, round(mid_y - second_step_width / 2) : round(mid_y + second_step_width / 2)
    ] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)

    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length + third_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + third_step_depth, round(mid_y - third_step_width / 2) : round(mid_y + third_step_width / 2)
    ] = third_step_height

    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length + forth_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + forth_step_depth, round(mid_y - forth_step_width / 2) : round(mid_y + forth_step_width / 2)
    ] = forth_step_height

    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size

    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)

    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)

    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length + slope_depth / 2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length : platform_length + slope_depth, left_y - slope_width // 2 : left_y + slope_width // 2] = (
        heights.astype(int) + platform_height
    )

    platform_length += slope_depth + gap_size
    goals[6] = [platform_length + slope_depth / 2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length : platform_length + slope_depth, right_y - slope_width // 2 : right_y + slope_width // 2] = (
        heights.astype(int) + platform_height
    )

    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int * terrain.horizontal_scale, 0.0, max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')


def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """

    def get_rand_dis_int(scale):
        return np.random.randint(int(-scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))

    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        move_corners[1:num_rows, 1:num_cols] -= hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles, move_x != 0
