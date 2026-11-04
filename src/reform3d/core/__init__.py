"""Shared infrastructure for REFORM-3D evaluation."""

from importlib import import_module

__all__ = ["checkpoints", "features", "paths", "reproducibility", "runtime"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
