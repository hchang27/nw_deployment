import numpy as np
# from jaxtyping import Array, Float


# def transform_points(pts: Float[Array, 'batch 3'], matrix: Float[Array, '16']):
def transform_points(pts, matrix):
    """Transforms a list of points by a transformation matrix in the three.js convention.

    Args:
        pts (): Tensor("batch_size, 3")
        matrix (): Tensor(16)

    Returns:

    """
    # Convert the list of points to a numpy array with an additional dimension for the homogeneous coordinate
    pts_homogeneous = np.hstack((pts, np.ones((len(pts), 1))))

    # Apply the transformation matrix to each point
    transformed_pts = pts_homogeneous @ matrix

    # Convert back to 3D points from homogeneous coordinates
    transformed_pts = transformed_pts[:, :3]
    return transformed_pts
