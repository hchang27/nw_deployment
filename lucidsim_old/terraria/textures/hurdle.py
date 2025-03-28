"""
Generate a Vuer geometry with textures that resembles the given heightmap.
"""
import asyncio
import os

import numpy as np
from params_proto import PrefixProto
from vuer import Vuer
from vuer.schemas import Box, DefaultScene, Plane, group, Sphere


class Hurdle(PrefixProto):
    dataset_prefix = "hurdle/scene_99999"
    root = "http://luma01.csail.mit.edu:4000"
    prefix = "scenes"

    # curb_texture = "curb.jpeg"
    # ground_texture = "grass.png"

    ground_z: np.uint16 = 0

    num_hurdles = 8

    horizontal_scale = 0.025
    vertical_scale = 0.005

    def __post_init__(self, **deps):
        from ml_logger import ML_Logger
        self.logger = ML_Logger(root=self.root, prefix=self.prefix)

        try:
            with self.logger.Prefix(os.path.join(self.dataset_prefix, "assets")):
                self.height_field_raw, = self.logger.load_pkl("smooth_height_field_raw.npy")
                self.vertices, = self.logger.load_pkl("vertices.npy")
                self.triangles, = self.logger.load_pkl("triangles.npy")
                self.goals, = self.logger.load_pkl("original_goals.npy")

        except:
            print("Some files are not found, expected to be added later.")

    def compute_hurdle_length(self, mask):
        """Compute the length of the hurdle in the given mask."""
        # Find the first non-zero element in each row
        first_nonzero = np.argmax(mask, axis=1)

        # Find the last non-zero element in each row
        last_nonzero = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1)

        # Compute the length of the hurdle
        hurdle_length = last_nonzero - first_nonzero

        return hurdle_length

    def compute_hurdle_width(self, mask):
        """Compute the width of the hurdle in the given mask."""
        # Count the number of rows that contain a non-zero value
        rows_with_hurdles = np.any(mask > 0, axis=1)
        hurdle_width = np.sum(rows_with_hurdles)

        return hurdle_width

    def _get_mesh(self, curb_material, ground_material):
        ground_idxs = np.where(self.height_field_raw == self.ground_z)

        hurdle_mask = np.ones_like(self.height_field_raw, dtype=np.uint8)
        hurdle_mask[ground_idxs] = 0

        first_nonzero = np.argmax(hurdle_mask, axis=1)
        last_nonzero = hurdle_mask.shape[1] - np.argmax(hurdle_mask[:, ::-1], axis=1)

        hurdle_length = (last_nonzero - first_nonzero).min() * self.horizontal_scale
        hurdle_width = (np.sum(np.any(hurdle_mask > 0, axis=1)) / self.num_hurdles) * self.horizontal_scale
        hurdle_height = self.height_field_raw.max() * self.vertical_scale

        centers = []

        goals_px = self.goals / self.horizontal_scale
        dis_x = goals_px[0, 0] + 1

        for i in range(1, len(goals_px) - 1):
            y_center = goals_px[i][1]
            rand_x = 2 * (goals_px[i][0] - dis_x)
            x_center = dis_x + rand_x
            dis_x = x_center
            centers.append((x_center, y_center))

        centers = np.array(centers) * self.horizontal_scale

        hurdles = []
        for i, (x_center, y_center) in enumerate(centers):
            hurdles.append(
                Box(args=[hurdle_width, hurdle_length, hurdle_height, 100, 100, 100],
                    position=[x_center, y_center, hurdle_height / 2],
                    key=f"hurdle_{i}",
                    **curb_material))

        return group(*hurdles, Plane(args=[500, 500, 10, 10], key="ground",
                                     **ground_material), key="hurdles")

    def __call__(self, curb_material, ground_material):
        return self._get_mesh(curb_material, ground_material)


if __name__ == '__main__':

    app = Vuer(static_root=os.environ["HOME"], queries=dict(grid=False))

    hurdles = Hurdle()

    curb_texture = "grass.png"
    ground_texture = "bricks.jpeg"
    background_texture = "snowy.jpeg"

    bricks_path = f"http://localhost:8012/static/{ground_texture}"
    grass_path = f"http://localhost:8012/static/{curb_texture}"
    background_path = f"http://localhost:8012/static/{background_texture}"

    ground_material = dict(materialType="standard", material=dict(map=grass_path, mapRepeat=[100, 100]))
    curb_material = dict(materialType="standard",
                         material=dict(map=bricks_path, mapRepeat=[0.1, 1]))  # , mapRepeat=[100, 100]))


    # ground_material = dict()

    @app.spawn
    async def main(sess):
        sess.set @ DefaultScene(
            # Box(args=[1, 1, 1], position=[0, 0.5, 0], color="red", materialType="standard", key="hurdle-1")
            hurdles(curb_material, ground_material),
            Sphere(
                key="background",
                args=[50, 32, 32],
                position=[0, 0, 0],
                rotation=[np.pi / 2, 0, 0],
                material=dict(side=2, map=background_path),
            )
            # TriMesh(vertices=np.array(hurdles.vertices), faces=np.array(hurdles.triangles),
            #         position=[-15, -15, 0]),

        )
        # sess.add @ DefaultScene()

        while True:
            await asyncio.sleep(1.0)


    app.run()
