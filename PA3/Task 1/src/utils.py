"""Small shared helpers: atomic I/O, seeding, chat templating, environment capture."""
import os
import json
import time
import random
import hashlib
import logging
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)


def load_config(path: str = "configs/task1.yaml") -> Dict[str, Any]:
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(TASK_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def atomic_torch_save(obj: Any, path: str) -> None:
    """torch.save to a temp file, then rename, so a crash never leaves a half-written file."""
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [s.cpu() for s in state["torch_cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            torch.cuda.set_rng_state(cuda_states[0])


def apply_chat(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """Qwen3 chat template with thinking disabled (enable_thinking=False), as the manual requires.

    With add_generation_prompt=True the rendered prompt ends with
    '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' (the non-thinking prefix)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )


NO_THINK_SUFFIX = "<think>\n\n</think>\n\n"


def assert_no_think_prompt(prompt: str) -> None:
    if not prompt.endswith(NO_THINK_SUFFIX):
        raise AssertionError(f"Prompt does not end with the non-thinking prefix: {prompt[-80:]!r}")


def eos_token_ids(tokenizer: Any, model: Optional[Any] = None) -> List[int]:
    """All ids that end a Qwen3 chat turn (<|im_end|>, <|endoftext|>, plus generation_config eos)."""
    ids = set()
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    if model is not None and getattr(model, "generation_config", None) is not None:
        cfg_eos = model.generation_config.eos_token_id
        if isinstance(cfg_eos, int):
            ids.add(cfg_eos)
        elif isinstance(cfg_eos, (list, tuple)):
            ids.update(int(x) for x in cfg_eos)
    return sorted(ids)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_size_mb(path: str, suffixes: Optional[tuple] = None) -> float:
    total = 0
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.endswith(".tmp"):
                continue
            if suffixes is None or fn.endswith(suffixes):
                total += os.path.getsize(os.path.join(root, fn))
    return total / 2**20


def resolve_model_revision(model_name: str) -> str:
    """Return the Hub commit hash of the checkpoint actually downloaded (for the manifest)."""
    try:
        from huggingface_hub import model_info
        return model_info(model_name).sha or "unknown"
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for pkg in ("torch", "transformers", "peft", "datasets", "accelerate", "lm_eval", "numpy"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            info[pkg] = "not_installed"
    if info.get("lm_eval") == "installed":
        try:
            from importlib.metadata import version
            info["lm_eval"] = version("lm_eval")
        except Exception:
            pass
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        info["cuda_version"] = torch.version.cuda
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=TASK_ROOT
        ).stdout.strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


def free_cuda() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
