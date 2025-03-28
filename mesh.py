import numpy as np
import open3d as o3d

# Replace 'path_to_mesh_file' with the path to your mesh file
mesh_path = '/Users/alanyu/Downloads/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply'
mesh = o3d.io.read_triangle_mesh(mesh_path)

# # Compute vertex normals if they haven't been computed already
# if not mesh.has_vertex_normals():
#     mesh.compute_vertex_normals()
# 
# # Reverse normals
# # Flip the normals (invert face winding order)
# mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
# 
# # Visualize the corrected mesh
# o3d.visualization.draw_geometries([mesh])


# Compute vertex normals if they haven't been computed already
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Reverse normals by flipping the winding order of triangles
mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])

# Define the distance threshold
distance_threshold = 6 # Replace with your desired distance

# Calculate distances of vertices from origin
vertices = np.asarray(mesh.vertices)
distances = np.linalg.norm(vertices, axis=1)

# Find indices of vertices within the distance threshold
valid_vertex_indices = np.where(distances <= distance_threshold)[0]
valid_vertex_indices_set = set(valid_vertex_indices)

# Filter out triangles that have vertices outside the threshold
triangles = np.asarray(mesh.triangles)
valid_triangles = np.array([triangle for triangle in triangles 
                            if set(triangle).issubset(valid_vertex_indices_set)])

# Update the mesh with filtered triangles
mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
mesh.remove_unreferenced_vertices()

# save the mesh
o3d.io.write_triangle_mesh("/Users/alanyu/Downloads/red_stairs.ply", mesh)

# Visualize the corrected mesh
o3d.visualization.draw_geometries([mesh])
