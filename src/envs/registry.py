from __future__ import annotations
from typing import Callable, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.entities import Tile, Unit

BoardFactory = Callable[[], Dict[int, "Tile"]]
UnitsFactory = Callable[[Dict[int, "Tile"]], Dict[int, "Unit"]]

_REGISTRY: Dict[str, Tuple[BoardFactory, UnitsFactory]] = {}


def register_preset(name: str):
    """Decorator to register a preset by name.

    Usage at the bottom of a preset module:

        @register_preset("hispania")
        def _register():
            return create_hispania_board, create_hispania_units
    """

    def decorator(factory_fn: Callable[[], Tuple[BoardFactory, UnitsFactory]]):
        board_fn, units_fn = factory_fn()
        _REGISTRY[name] = (board_fn, units_fn)
        return factory_fn

    return decorator


def get_preset(name: str) -> Tuple[BoardFactory, UnitsFactory]:
    """Return (board_factory, units_factory) for a named preset."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_presets() -> list[str]:
    """Return all registered preset names."""
    return list(_REGISTRY.keys())


def _load_presets():
    """Auto-import all modules under envs/presets/ so they self-register."""
    import importlib
    import pkgutil
    import envs.presets as _presets_pkg

    for _, module_name, _ in pkgutil.iter_modules(_presets_pkg.__path__):
        importlib.import_module(f"envs.presets.{module_name}")


_load_presets()
