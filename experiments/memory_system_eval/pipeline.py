"""Sequential, resumable orchestrator for the remaining stages (one GPU job at a time; a step is skipped when its output exists).
Steps that need Stage 5 outputs wait for results/s5/metrics.json. Run detached:  python pipeline.py  (log: logs/pipeline.log)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
STEPS = [("s6_longseq", "exp/s6_longseq.py", "results/s6/longseq.json", False),
         ("s5_vjb", "exp/s5_vjb.py", "results/s5/metrics_vjb.json", True),
         ("s7_efficiency", "exp/s7_efficiency.py", "results/s7/efficiency.json", False)]


def log(msg):
    with open(HERE / "logs" / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def main():
    for name, script, out, needs_s5 in STEPS:
        if (HERE / out).exists():
            log(f"[skip] {name}: {out} exists")
            continue
        if needs_s5:
            while not (HERE / "results/s5/metrics.json").exists():
                time.sleep(30)
        log(f"[run ] {name}")
        t0 = time.time()
        with open(HERE / "logs" / f"{name}.log", "a", encoding="utf-8") as f:
            r = subprocess.run([PY, script], cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                                                                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        ok = r.returncode == 0 and (HERE / out).exists()
        log(f"[{'ok' if ok else 'FAIL'}] {name} rc={r.returncode} {time.time() - t0:.0f}s")
        if not ok:
            break


if __name__ == "__main__":
    main()
