"""Point gsplat's JIT build at the pip-installed CUDA toolkit (``nvidia-cuda-nvcc``).

The system ``nvcc`` may be too old for gsplat (C++20). torch's cpp_extension wants a
conventional ``CUDA_HOME`` layout, so we assemble one from symlinks in the user cache.
Nothing happens if ``CUDA_HOME`` is already set or the pip toolkit is not installed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _symlink(source: Path, dest: Path) -> None:
    if dest.is_symlink():
        if dest.resolve() == source.resolve():
            return
        dest.unlink()
    elif dest.exists():
        raise RuntimeError(f"CUDA toolkit cache path already exists: {dest}")
    dest.symlink_to(source, target_is_directory=source.is_dir())


def configure_cuda_toolkit() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or spec.submodule_search_locations is None:
        return
    root = next((Path(loc) / "cu13" for loc in spec.submodule_search_locations
                 if (Path(loc) / "cu13" / "bin" / "nvcc").is_file()), None)
    if root is None:
        return
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    home = cache / "semantic-3dgs" / "cuda-13.0"
    try:
        home.mkdir(parents=True, exist_ok=True)
        for name in ("bin", "include", "nvvm"):
            _symlink(root / name, home / name)
        lib = home / "lib"
        lib.mkdir(exist_ok=True)
        for f in (root / "lib").iterdir():
            if f.is_file():
                _symlink(f, lib / f.name)
        _symlink(root / "lib" / "libcudart.so.13", lib / "libcudart.so")
    except OSError:
        return
    os.environ["CUDA_HOME"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
