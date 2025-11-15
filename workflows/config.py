from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform


here = Path(__file__).resolve().parent


def _ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class DataPaths:
    """Central object holding key paths for data and local assets."""

    source_data: Path
    input_data: Path
    scratch: Path
    logs: Path
    blueprints: Path
    model_config: Path


def get_data_paths() -> DataPaths:
    """
    Return canonical data and project paths with sensible defaults.
    """
    home = Path(os.environ.get("HOME", str(Path.home())))

    # Optionally adapt defaults for macOS
    if platform.system() == "Darwin":
        source_data = home / "data" / "source_data"
        input_data = home / "data" / "input_data"
        scratch = home / "data" / "scratch"

    # Local project assets
    model_config = here / "model-configs" / "opt_base_roms-marbl-cson-default"
    logs_dir = here / "logs"
    blueprints_dir = here / "blueprints"

    # Ensure all directories exist
    for p in (source_data, input_data, scratch, logs_dir, blueprints_dir):
        _ensure_dir(p)


    return DataPaths(
        source_data=source_data,
        input_data=input_data,
        scratch=scratch,
        logs=logs_dir,
        blueprints=blueprints_dir,
        model_config=model_config,
    )


# Initialize and export a single canonical instance
paths = get_data_paths()
