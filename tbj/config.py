"""Project configuration: a YAML file whose paths resolve relative to its folder."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config.yaml"


class Config(dict):
    """The parsed config.yaml plus helpers for project paths."""

    def __init__(self, data: dict, root: Path):
        super().__init__(data)
        self.root = Path(root).resolve()

    def path(self, key: str) -> Path:
        """Absolute path for an entry under `paths:`."""
        return (self.root / self["paths"][key]).resolve()

    def setup(self, name: str) -> dict:
        try:
            return self["setups"][name]
        except KeyError:
            raise SystemExit(f"Unknown setup '{name}'. Known: {', '.join(self['setups'])}")

    def plan(self, name: str) -> dict:
        try:
            return self["plans"][name]
        except KeyError:
            raise SystemExit(f"Unknown plan '{name}'. Known: {', '.join(self['plans'])}")


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("TBJ_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}. Run commands from the repo folder or pass --config.")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data, path.parent)
