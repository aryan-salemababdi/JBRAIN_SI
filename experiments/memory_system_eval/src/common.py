"""Shared paths, tokenizer, hashing, resource checks and state snapshots for the memory-system evaluation.
Read-only with respect to everything outside experiments/memory_system_eval/."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]           # experiments/memory_system_eval
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))

DATA, FEAT, RESULTS, LOGS = HERE / "data", HERE / "features", HERE / "results", HERE / "logs"
STAGES, FIG, INSPECT, MODELS = HERE / "stages", HERE / "figures", HERE / "inspect", HERE / "models"
CKPT = Path(r"D:\aryan\JBRAIN\Models\JBR1_V2_CE\jbr_v2_ce_weights.pt")
CKPT_SHA256 = "f735065aac46203c0e9152f4b96a64c04f7d2416409c01579eaf021e1b1ca850"
SPEC = dict(vocab_size=100277, d_model=768, n_heads=12, n_layers=12)
SEEDS = (0, 1, 2)
WAEM = REPO / "experiments" / "waem_controlled_retrieval"
OLD = Path(r"D:\aryan\JBRAIN\experiments")

_ENC = None


def tokenizer():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def enc(text: str) -> list[int]:
    return tokenizer().encode(text, disallowed_special=())


def dec(ids) -> str:
    return tokenizer().decode(list(ids))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def write_json(path: Path, obj, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def disk_free_gb(path: Path = HERE) -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def ram_available_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def vram_free_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / 2 ** 30
    except Exception:
        pass
    return 0.0


def check_resources(need_disk_gb: float = 0.0, need_ram_gb: float = 0.0, need_vram_gb: float = 0.0, label: str = "") -> dict:
    """Pre-flight check before an expensive step; raises if a requirement cannot be met (never silently proceeds)."""
    r = {"label": label, "disk_free_gb": round(disk_free_gb(), 2), "ram_available_gb": round(ram_available_gb(), 2),
         "vram_free_gb": round(vram_free_gb(), 2), "need": {"disk": need_disk_gb, "ram": need_ram_gb, "vram": need_vram_gb}}
    if r["disk_free_gb"] < need_disk_gb + 5.0:            # keep a 5 GB margin on the drive
        raise RuntimeError(f"insufficient disk for {label}: {r}")
    if r["ram_available_gb"] < need_ram_gb + 2.0:
        raise RuntimeError(f"insufficient RAM for {label}: {r}")
    if need_vram_gb and r["vram_free_gb"] < need_vram_gb:
        raise RuntimeError(f"insufficient VRAM for {label}: {r}")
    return r


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout


def snapshot_state(tag: str, hash_checkpoint: bool = True) -> dict:
    """Record git state and hashes of production files and the checkpoint (inspect/STATE_<tag>.json)."""
    out = INSPECT / f"STATE_{tag}.json"
    prod = sorted((REPO / "jbrain_model").glob("*.py"))
    state = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": git("rev-parse", "HEAD").strip(),
             "git_status_short": git("status", "--short").splitlines(),
             "core_py_diff_vs_head_sha256": sha256_text(git("diff", "HEAD", "--", "jbrain_model/core.py")),
             "production_sources_sha256": {p.name: sha256_file(p) for p in prod}}
    if hash_checkpoint:
        state["checkpoint_sha256"] = sha256_file(CKPT)
        state["checkpoint_ok"] = state["checkpoint_sha256"] == CKPT_SHA256
    write_json(out, state, overwrite=True)
    return state


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0
