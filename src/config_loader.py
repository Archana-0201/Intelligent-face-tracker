"""
config_loader.py
----------------
Loads and validates the configuration from config.json.
Provides a singleton Config object accessible throughout the application.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


class ConfigLoader:
    """Singleton configuration loader for the face tracker system."""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[dict] = None

    def __new__(cls, config_path: str = "config.json"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str):
        """Load configuration from JSON file."""
        abs_path = os.path.abspath(config_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Config file not found: {abs_path}")
        with open(abs_path, "r") as f:
            self._config = json.load(f)

    def get(self, *keys, default=None):
        """Get a nested config value by dot-separated keys or positional args."""
        data = self._config
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return default
            if data is None:
                return default
        return data

    @property
    def video(self) -> dict:
        return self._config.get("video", {})

    @property
    def detection(self) -> dict:
        return self._config.get("detection", {})

    @property
    def recognition(self) -> dict:
        return self._config.get("recognition", {})

    @property
    def tracking(self) -> dict:
        return self._config.get("tracking", {})

    @property
    def database(self) -> dict:
        return self._config.get("database", {})

    @property
    def logging_cfg(self) -> dict:
        return self._config.get("logging", {})

    @property
    def ui(self) -> dict:
        return self._config.get("ui", {})

    @property
    def frame_skip(self) -> int:
        """Number of frames to skip between detection cycles (configurable)."""
        return self._config.get("detection", {}).get("frame_skip", 2)

    def __repr__(self):
        return f"ConfigLoader(config={self._config})"


# Module-level singleton accessor
def get_config(config_path: str = "config.json") -> ConfigLoader:
    return ConfigLoader(config_path)
