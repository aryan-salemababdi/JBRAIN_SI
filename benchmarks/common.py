"""Shared helpers: hashing, canonical config hashing, uncertainty, environment capture, manifest, resume compatibility, per-sample validation."""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CODE_FILES = ("__init__.py", "__main__.py", "common.py", "adapter.py", "metrics.py", "run.py")


class CompatibilityError(RuntimeError):
    """Raised when a resume or comparison would silently mix incompatible configurations."""


class ValidationError(RuntimeError):
    """Raised when per-example results are missing, duplicated or invalid."""


# ------------------------------------------------------------------ hashing / json
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Key-order independent, whitespace independent JSON (floats are rendered by repr, so equal floats hash equally)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode("utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_sha256() -> str:
    """Hash of the benchmark package sources (the package is untracked in git, so a commit id alone would not identify it)."""
    h = hashlib.sha256()
    for name in CODE_FILES:
        p = PKG_DIR / name
        if p.is_file():
            h.update(name.encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def git_info(root: Path = REPO_ROOT) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    head = run("rev-parse", "HEAD")
    dirty = [l for l in run("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    return {"head": head or None, "tracked_files_modified": len(dirty), "note": "benchmarks/ is untracked; see code_sha256 for its exact content"}


# ------------------------------------------------------------------ statistics
def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_mean_ci(values: list[float], iters: int = 2000, seed: int = 20260101, alpha: float = 0.05) -> tuple[float, float]:
    import numpy as np
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(iters, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


# ------------------------------------------------------------------ environment
PACKAGES = ("torch", "numpy", "tiktoken", "lm_eval", "datasets", "transformers", "huggingface_hub", "pyarrow", "tqdm", "scipy")


def environment_info(device: str) -> dict:
    import torch
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = _md.version(p)
        except Exception:
            pk[p] = None
    env = {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_runtime": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        "device_requested": device, "packages": pk,
        "offline_flags": {k: os.environ.get(k) for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
        "torch_num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return env


# ------------------------------------------------------------------ manifest
MANIFEST_FILES = ("config.json", "environment.json", "checkpoint.json", "results.json", "per_sample_results.jsonl", "run.log", "loglik_cache.jsonl", "compute_dtype.json")


def build_manifest(run_dir: Path, extra: dict) -> dict:
    files = {}
    for name in MANIFEST_FILES:
        p = run_dir / name
        if p.is_file():
            files[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {"schema_version": SCHEMA_VERSION, "files": files, **extra}


def verify_manifest(run_dir: Path) -> list[str]:
    """Return a list of problems (empty = all hashes match)."""
    run_dir = Path(run_dir)
    mp = run_dir / "manifest.json"
    if not mp.is_file():
        return ["manifest.json missing (run incomplete?)"]
    man = read_json(mp)
    problems = []
    for name, meta in man.get("files", {}).items():
        p = run_dir / name
        if not p.is_file():
            problems.append(f"{name}: missing")
        elif sha256_file(p) != meta["sha256"]:
            problems.append(f"{name}: sha256 mismatch")
    cfg = run_dir / "config.json"
    if cfg.is_file() and "config_hash" in man and config_hash(read_json(cfg)) != man["config_hash"]:
        problems.append("config.json: config_hash mismatch")
    return problems


# ------------------------------------------------------------------ resume compatibility
def diff_configs(old: dict, new: dict, prefix: str = "") -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "<absent>"), new.get(k, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff_configs(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k}: {a!r} -> {b!r}")
    return out


def check_resume_compat(run_dir: Path, new_cfg: dict, new_code_sha: str, new_ckpt_sha: str) -> None:
    run_dir = Path(run_dir)
    if not (run_dir / "config.json").is_file():
        raise CompatibilityError(f"{run_dir} has no config.json; cannot resume")
    if (run_dir / "manifest.json").is_file():
        raise CompatibilityError(f"{run_dir} is already complete (manifest.json exists); start a new run instead")
    old_cfg = read_json(run_dir / "config.json")
    problems = diff_configs(old_cfg, new_cfg)
    old_ck = read_json(run_dir / "checkpoint.json").get("sha256") if (run_dir / "checkpoint.json").is_file() else None
    if old_ck != new_ckpt_sha:
        problems.append(f"checkpoint sha256: {old_ck} -> {new_ckpt_sha}")
    st = read_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
    if st.get("code_sha256") not in (None, new_code_sha):
        problems.append(f"benchmark code changed since the run started: {st.get('code_sha256')} -> {new_code_sha}")
    if problems:
        raise CompatibilityError("refusing to resume; configuration differs:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------ per-example validation
def validate_per_sample(records: list[dict], task: str, expected_n: int) -> dict:
    """Detect missing, duplicate and invalid examples. Raises ValidationError; returns a small summary when clean."""
    problems = []
    ids = [r.get("doc_id") for r in records]
    seen, dup = set(), []
    for i in ids:
        if i in seen:
            dup.append(i)
        seen.add(i)
    if dup:
        problems.append(f"{len(dup)} duplicate doc_id(s), e.g. {sorted(set(dup))[:5]}")
    bad_ids = [i for i in seen if not isinstance(i, int) or isinstance(i, bool)]
    if bad_ids:
        problems.append(f"non-integer doc_id(s): {bad_ids[:5]}")
    good = {i for i in seen if isinstance(i, int) and not isinstance(i, bool)}
    expected = set(range(expected_n))
    missing = sorted(expected - good)
    extra = sorted(good - expected)
    if missing:
        problems.append(f"{len(missing)} missing doc_id(s), e.g. {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected doc_id(s), e.g. {extra[:5]}")
    invalid = []
    for r in records:
        why = None
        lls = r.get("choice_loglik") if r.get("kind") == "multiple_choice" else [r.get("loglik")]
        if lls is None or any((x is None) or not isinstance(x, (int, float)) or not math.isfinite(x) or x > 1e-6 for x in lls):
            why = "non-finite or positive log-likelihood"
        elif r.get("kind") == "multiple_choice":
            g, n = r.get("gold_index"), len(lls)
            if not isinstance(g, int) or not (0 <= g < n):
                why = f"gold_index {g!r} outside 0..{n - 1}"
        elif r.get("kind") == "word_prediction" and (not isinstance(r.get("target"), str) or not r["target"]):
            why = "empty target"
        if why:
            invalid.append((r.get("doc_id"), why))
    if invalid:
        problems.append(f"{len(invalid)} invalid record(s), e.g. {invalid[:3]}")
    if problems:
        raise ValidationError(f"[{task}] " + "; ".join(problems))
    return {"task": task, "n": len(records), "expected": expected_n}
