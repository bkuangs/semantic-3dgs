"""Scene-graph dynamic & semantic 3D Gaussian Splatting for driving logs."""

__version__ = "0.1.0"

from semantic_3dgs._cuda import configure_cuda_toolkit as _configure_cuda_toolkit

_configure_cuda_toolkit()
