import warnings

try:
    import pyzed
    import uvc
    ON_ORIN = True
except ImportError:
    warnings.warn("Can't import pyzed! OK if testing on local")
    ON_ORIN = False
