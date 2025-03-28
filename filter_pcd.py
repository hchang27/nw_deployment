import open3d as o3d
import trimesh
import numpy as np

pcd = o3d.io.read_point_cloud("/Users/alanyu/Downloads/point_cloud.ply")
pcd.estimate_normals()

radius = 7
center = np.array([0 , 0, 0])

coords_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)

points = np.asarray(pcd.points)

R = pcd.get_rotation_matrix_from_xyz(np.deg2rad([242, 21, 5]))
pcd = pcd.rotate(R, center=(0,0,2.32))
pcd = pcd.voxel_down_sample(voxel_size=0.1)

points = np.asarray(pcd.points)

# Calculate the Euclidean distance from each point to the center
distances = np.linalg.norm(points - center, axis=1)

# Create a boolean mask for points within the specified radius
within_radius_mask = distances <= radius

# Filter points using the mask
filtered_points = points[within_radius_mask]

# Create a new point cloud from the filtered points
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Define the parameters for statistical outlier removal
nb_neighbors = 10      # Number of neighbors to consider for each point
std_ratio = 0.5   # Standard deviation ratio to determine outliers

# Remove outliers from the point cloud
filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=std_ratio)
filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.2)
filtered_pcd.estimate_normals()

# Save the result
# o3d.io.write_point_cloud("/Users/alanyu/Downloads/filtered_pcd.ply", filtered_pcd)

# # Create a mesh from the depth image using Marching Cubes
# mesh = o3d.geometry.TriangleMesh.create_from_depth_image(
#     o3d.geometry.Image(depth_image), 
#     o3d.geometry.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
#  Apply the Poisson surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_pcd)[0]
mesh.compute_vertex_normals()
mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

# plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.2,
#                                          ransac_n=3,
#                                          num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# 
# # viz 
# plane_cloud = filtered_pcd.select_by_index(inliers)
# plane_cloud.paint_uniform_color([1.0, 0, 0])
# 
# noneplane_cloud = filtered_pcd.select_by_index(inliers, invert=True)
# noneplane_cloud.paint_uniform_color([0, 0, 1.0])

# o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud])
# 
# # Optionally, simplify the mesh
viz = o3d.visualization.Visualizer()
viz.create_window()
viz.add_geometry(filtered_pcd)
bbox = pcd.get_axis_aligned_bounding_box()  # or pcd.get_oriented_bounding_box()
viz.add_geometry(coords_marker)
viz.add_geometry(mesh)
viz.run()
viz.close()
