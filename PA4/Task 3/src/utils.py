"""Shared helpers: paths, config, seeding, JSON I/O, timing, environment record."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

TASK_DIR = Path(__file__).resolve().parents[1]          # PA4/Task 3
PA4_DIR = TASK_DIR.parent
RESULTS = TASK_DIR / "results"


def log(*msg):
    print(time.strftime("[%H:%M:%S]"), *msg, flush=True)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else TASK_DIR / "configs" / "task3.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def data_repo(cfg) -> Path:
    p = Path(cfg["data"]["repo_dir"])
    return p if p.is_absolute() else (TASK_DIR / p).resolve()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else str(v)
    if isinstance(o, float):
        return o if np.isfinite(o) else str(o)
    if isinstance(o, np.ndarray):
        return _to_jsonable(o.tolist())
    if isinstance(o, torch.Tensor):
        return _to_jsonable(o.detach().cpu().tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_jsonable(r)) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(store: dict, key: str):
    """Wall time of a block with CUDA synchronised on both sides (seconds)."""
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def gpu_mem() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved()}


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def free_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(TASK_DIR), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_record(cfg=None) -> dict:
    import importlib.metadata as md
    rec = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "git_commit": git_commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    for pkg in ["transformers", "huggingface_hub", "datasets", "tokenizers", "safetensors", "numpy",
                "scipy", "accelerate", "peft"]:
        try:
            rec[pkg] = md.version(pkg)
        except Exception:
            rec[pkg] = None
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        rec["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_memory_bytes": p.total_memory,
                      "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "count": torch.cuda.device_count()}
    if cfg is not None:
        rec["config_sha256"] = cfg.get("_sha256")
    return rec
