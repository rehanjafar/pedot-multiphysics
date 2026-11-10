"""
I/O utilities for saving/loading simulation data.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    """Save dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)


def load_json(path: Path) -> dict:
    """Load JSON file to dict."""
    with open(Path(path)) as f:
        return json.load(f)


def save_npz(arrays: dict[str, np.ndarray], path: Path) -> None:
    """Save dict of arrays to compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load NPZ file to dict of arrays."""
    with np.load(Path(path)) as data:
        return dict(data)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_run_dir(base: Path, campaign_id: str, run_id: str) -> Path:
    """Get the output directory for a specific run."""
    return ensure_dir(Path(base) / "simulated_raw" / campaign_id / run_id)


def get_campaign_dir(base: Path, campaign_id: str) -> Path:
    """Get the aggregated output directory for a campaign."""
    return ensure_dir(Path(base) / "simulated_agg" / campaign_id)


def get_report_dir(base: Path, campaign_id: str) -> Path:
    """Get the report directory for a campaign."""
    return ensure_dir(Path(base) / "reports" / campaign_id)


def save_metadata(
    run_dir: Path,
    config: Any,
    runtime_seconds: float,
    warnings: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
) -> None:
    """Save runtime metadata for a run."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": runtime_seconds,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "warnings": warnings or [],
        "errors": errors or [],
    }
    save_json(meta, Path(run_dir) / "metadata.json")


class RunManifest:
    """Track completed and failed runs for a campaign."""

    def __init__(self, campaign_dir: Path) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.manifest_path = self.campaign_dir / "manifest.json"
        self.completed: list[str] = []
        self.failed: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.manifest_path.exists():
            data = load_json(self.manifest_path)
            self.completed = data.get("completed", [])
            self.failed = data.get("failed", [])

    def save(self) -> None:
        save_json(
            {"completed": self.completed, "failed": self.failed},
            self.manifest_path,
        )

    def mark_completed(self, run_id: str) -> None:
        if run_id not in self.completed:
            self.completed.append(run_id)
        if run_id in self.failed:
            self.failed.remove(run_id)
        self.save()

    def mark_failed(self, run_id: str) -> None:
        if run_id not in self.failed:
            self.failed.append(run_id)
        self.save()

    def is_completed(self, run_id: str) -> bool:
        return run_id in self.completed


def validate_id(value: str) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or ".." in value:
        raise ValueError("IDs must be safe names containing letters, digits, _, - or .")
    return value
