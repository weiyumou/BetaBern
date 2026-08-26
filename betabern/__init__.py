from importlib.metadata import version

#: Read from the installed distribution so ``pyproject.toml`` is the single source.
__version__ = version("betabern")
