from __future__ import annotations
from typing import Callable, Dict

from envs.presets.config import PresetConfig

_REGISTRY: Dict[str, PresetConfig] = {}


def register_preset(name: str) -> Callable:
    """
    Decorator that registers a PresetConfig factory.

    Usage:
        @register_preset("hispania")
        def _():
            return PresetConfig(name="hispania", ...)
    """

    def decorator(factory_fn: Callable[[], PresetConfig]) -> Callable:
        _REGISTRY[name] = factory_fn()
        return factory_fn

    return decorator


def get_preset(name: str) -> PresetConfig:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_presets() -> list[str]:
    return list(_REGISTRY.keys())


def _load_presets() -> None:
    """Auto-import preset modules so they self-register via @register_preset."""
    import importlib
    import pkgutil
    import envs.presets as _pkg

    for _, module_name, _ in pkgutil.iter_modules(_pkg.__path__):
        if module_name not in ("registry", "config"):
            importlib.import_module(f"envs.presets.{module_name}")


_load_presets()
