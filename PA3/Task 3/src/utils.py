"""Shared helpers for Task 3: config, paths, atomic JSON/CSV I/O, logging, environment capture.

Kept free of torch so the offline analysis runs on a CPU-only machine; torch is imported lazily by src/live.py.
"""
import os
import json
import time
import hashlib
import logging
import platform
import subprocess
from typing import Any, Dict

import numpy as np
import yaml

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)


def load_config(path: str = "configs/task3.yaml") -> Dict[str, Any]:
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(TASK_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_path(*parts: str) -> str:
    p = os.path.join(*parts)
    return p if os.path.isabs(p) else os.path.join(TASK_ROOT, p)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def atomic_json_dump(obj: Any, path: str, indent: int = 1) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_frame(df, path: str) -> None:
    """CSV (git-friendly, diffable) for small tables, parquet for the large per-order tables."""
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    if path.endswith(".parquet"):
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


TEXT_COLUMNS = ("sample_id", "extracted", "group_key", "gold", "gold_key", "answer", "initial_answer")


def read_frame(path: str):
    """CSV ids and answers stay strings (MathVista ids like '10' and answers like '40' must not become numbers)."""
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={c: str for c in TEXT_COLUMNS})


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_int(*parts: Any, bits: int = 63) -> int:
    """Deterministic integer from strings (SHA-256), used for per-call seeds and example selection."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << bits) - 1)


def package_versions() -> Dict[str, str]:
    from importlib import metadata
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "pyarrow", "scipy", "math-verify", "latex2sympy2_extended", "sympy",
                "torch", "transformers", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = "not_installed"
    return out


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "platform": platform.platform(), "packages": package_versions()}
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                            check=True, cwd=TASK_ROOT).stdout.strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info
