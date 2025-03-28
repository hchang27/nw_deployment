def sphere(args):
    '''
    Radius of the sphere.
    '''
    return " ".join([str(args[0]), '0', '0'])


def box(args):
    '''
    X half-size; Y half-size; Z half-size.
    '''
    return " ".join([str(args[0] / 2), str(args[1] / 2), str(args[2] / 2)])


def plane(args):
    '''
    X half-size; Y half-size; grid spacing.
    '''
    return " ".join([str(args[0] / 2), str(args[1] / 2), '1'])


VUER_PRIMITIVES = {"sphere": sphere, "box": box, "plane": plane}
