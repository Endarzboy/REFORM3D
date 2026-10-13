"""Resolve optional upstream sources without depending on the research checkout."""

import os
import sys
from pathlib import Path


def source_root(method: str, bundled: Path) -> Path:
    key = f"REFORM3D_{method.upper()}_SOURCE_ROOT"
    return Path(os.environ.get(key, str(bundled))).expanduser().resolve()


def add_source_path(root: Path, required_file: str, packages: tuple[str, ...]) -> None:
    if not (root / required_file).is_file():
        raise FileNotFoundError(
            f"Required upstream source is missing: {root / required_file}. "
            "Set this method's REFORM3D_<METHOD>_SOURCE_ROOT to its documented source directory."
        )
    for name in packages:
        module = sys.modules.get(name)
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        paths = [Path(origin)] if origin else [Path(p) for p in getattr(module, "__path__", ())]
        if not paths or any(not path.resolve().is_relative_to(root) for path in paths):
            raise RuntimeError(
                f"Module {name!r} is already loaded from outside {root}; "
                "run this external encoder in a fresh process."
            )
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
