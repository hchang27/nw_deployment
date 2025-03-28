from pathlib import Path

import numpy as np
import open3d as o3d
from typing import List

from vuer import Vuer
from vuer.events import Set
from vuer.schemas import DefaultScene, Ply, PointCloud, Box, CoordsMarker, Movable, Plane, group
from vuer.serdes import b64jpg

import matplotlib.pyplot as plt

assets_folder = Path(__file__).parent / "/Users/alanyu/Downloads/"
test_file = "filtered_pcd.ply"

# trimesh has issue loading large pointclouds.
pcd = o3d.io.read_point_cloud(str(assets_folder / test_file))

app = Vuer(static_root=assets_folder, port=8013, uri="ws://localhost:8013")


def plane_from_fit(fit: List[float]):
    a, b, c = fit.T[0]

    normal = np.array([a, b, -1])

    r = np.linalg.norm(normal[:2])

    yz = np.linalg.norm(normal[1:])
    x_rot = -np.arcsin(-normal[1] / yz)

    xz = np.linalg.norm(normal[[0, 2]])
    y_rot = -np.arcsin(-normal[0] / xz)

    xy = np.linalg.norm(normal[:2])
    z_rot = np.arccos(normal[0] / xy)

    position = [0, 0, c]
    # x = 0
    # y = np.arctan2(np.sqrt(a ** 2 + b ** 2), -1)
    # z = np.arctan2(b, a)

    # x = np.arctan2(-b, np.sqrt(a ** 2 + 1))
    # y = np.arctan2(a, np.sqrt(b ** 2 + 1))
    # z = 0

    return position, [x_rot, y_rot, z_rot]


def fit(pts):
    # pts: N x 3

    # A: N x 3
    A = np.hstack([pts[:, :2], np.ones((pts.shape[0], 1))])
    b = pts[:, 2:]  # N x 1

    sol = np.linalg.inv(A.T @ A) @ A.T @ b
    predictions = A @ sol
    errors = b - predictions
    residual = np.linalg.norm(errors)
    print("solution: %f x + %f y + %f = z" % (sol[0], sol[1], sol[2]))

    # print("errors: \n", errors)
    # print("predictions: \n", predictions)
    # print("residual:", residual)

    def func(x, y):
        return sol[0] * x + sol[1] * y + sol[2]

    return sol, func


from scipy.spatial.transform import Rotation as R


def plane_to_threejs_position_rotation(normal, point):
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Position is the same as the point on the plane
    position = point

    # Three.js Z-axis vector (up direction in your coordinate system)
    z_vector = np.array([0, 0, 1])

    # Rotation axis is the cross product of the Z-axis vector and the normal
    rotation_axis = np.cross(z_vector, normal)
    if np.linalg.norm(rotation_axis) < 1e-6:
        # Handle the special case where the normal is parallel or anti-parallel to the Z-axis
        rotation_axis = np.array([1, 0, 0])
        angle = 0 if normal[2] > 0 else np.pi
    else:
        # Normalize the rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # Angle between the Z-axis vector and the normal
        angle = np.arccos(np.clip(np.dot(z_vector, normal), -1.0, 1.0))

    # Calculate the rotation in radians
    rotation = R.from_rotvec(rotation_axis * angle)

    # Convert rotation to Euler angles (Three.js default is XYZ order)
    rotation_euler = rotation.as_euler('XYZ', degrees=False)

    return position, rotation_euler


@app.spawn(start=True)
async def main(proxy):
    points = np.asarray(pcd.points)

    y_max = -5.5
    fit_points = points[points[:, 1] < y_max]

    sol, func = fit(fit_points)

    sol = sol.T[0]

    point_on_plane = np.array([0, 0, sol[2]])

    normal = np.array([sol[0], sol[1], -1])

    pos, rot = plane_to_threejs_position_rotation(normal, point_on_plane)

    print(pos, rot)

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    heights = (Z - func(X, Y))

    from scipy.interpolate import griddata

    num_bins = 50
    # Create grid coordinates for discretized X and Y
    grid_x, grid_y = np.mgrid[X.min():X.max():complex(num_bins), Y.min():Y.max():complex(num_bins)]

    # Interpolate Z values onto the grid
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')

    grid_z = np.nan_to_num(grid_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Plotting the heatmap (heightmap)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_z, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Z Intensity')
    plt.xlabel('Discretized X')
    plt.ylabel('Discretized Y')
    plt.title('Heatmap (Heightmap) of Z Intensity with Discretized X and Y')
    plt.show()

    width_px = grid_z.shape[0]
    length_px = grid_z.shape[1]

    width = width_px * 0.1
    length = length_px * 0.1
    
    scale = grid_z.max() - grid_z.min()
    shift = grid_z.min()
    heightmap = (grid_z - shift) / scale
    heightmap_uint8 = (heightmap * 255).astype(np.uint8)

    proxy @ Set(
        DefaultScene(
            PointCloud(
                key="pointcloud",
                vertices=np.array(pcd.points),
                position=[0, 0, 0],
                size=0.05,
            ),
            Plane(
                key="ground-plane",
                args=[20, 20, 10, 10],
                position=pos,
                rotation=rot,
                materialType="standard",
                material=dict(color="green", side=2),
            ),
            Plane(
                args=[length, width, length_px, width_px],
                key='heightmap',
                materialType="standard",
                material=dict(
                    displacementMap=b64jpg(heightmap_uint8),
                    displacementScale=scale,
                    displacementBias=shift,
                ),
                rotation=[0, 0, np.pi / 2],
            ),
            up=[0, 0, 1],
        ),
    )


"""
 -0.001869 x + -0.035755 y + 0.888523 = z
errors: 
"""
