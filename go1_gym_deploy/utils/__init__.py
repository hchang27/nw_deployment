import torch
from waterbear import Bear


def class_to_bear(obj) -> Bear:
    if not hasattr(obj, "__dict__"):
        return obj
    result = Bear()
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_bear(item))
        else:
            element = class_to_bear(val)
        result[key] = element
    return result


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(r), -torch.sin(r)], [0, torch.sin(r), torch.cos(r)]]
        , device=rpy.device)

    R_y = torch.tensor(
        [[torch.cos(p), 0, torch.sin(p)], [0, 1, 0], [-torch.sin(p), 0, torch.cos(p)]]
        , device=rpy.device)

    R_z = torch.tensor(
        [[torch.cos(y), -torch.sin(y), 0], [torch.sin(y), torch.cos(y), 0], [0, 0, 1]], device=rpy.device
    )

    # rot = torch.dot(R_z, torch.dot(R_y, R_x))
    rot = R_z @ (R_y @ R_x)
    return rot
