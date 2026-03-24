"""Module register."""

import importlib
import logging
import pkgutil
from inspect import getmembers
from typing import Generic, Type, TypeVar

from ..schemas import LLM, GroupRewardModel, RewardModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Register(Generic[T]):
    """Module register"""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __call__(self, *arg, **kwarg):
        return self.register(*arg, **kwarg)

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def __getitem__(self, key) -> Type[T]:
        try:
            return self._dict[key]
        except Exception as e:
            logger.error(f"module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    @property
    def keys(self):
        """key"""
        return self._dict.keys()

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)


class registers:
    """All module registers."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    reward_model = Register[RewardModel | GroupRewardModel]("reward_models")
    llm = Register[LLM]("llms")


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logger.warning(f"Module {name} import failed: {err}")
    logger.fatal("Please check these modules.")


def find_modules():
    """Recursively find all modules under path."""

    from .paths import package_dir

    directories = [
        register._name
        for name, register in getmembers(registers)
        if isinstance(register, Register)
    ]
    for directory in directories:
        assert (package_dir / directory).exists()

    all_modules = [
        name
        for directory in directories
        for importer, name, ispkg in pkgutil.walk_packages(
            [str(package_dir / directory)], f"{package_dir.name}.{directory}."
        )
    ]

    return all_modules


def import_all_modules_for_register():
    """Import all modules for register."""

    all_modules = find_modules()
    logger.debug(f"All modules: {all_modules}")

    errors = []
    for module_name in all_modules:
        try:
            importlib.import_module(module_name)
            logger.debug(f"{module_name} loaded.")
        except ImportError as error:
            errors.append((module_name, error))
    _handle_errors(errors)
