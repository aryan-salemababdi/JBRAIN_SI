"""Stage 7b: clean resident-memory measurement of the on-disk memory at query time. For each N, a fresh process opens the store built
by Stage 7 (features/s7/store_N) and answers the Stage 7 queries with HIER, IBF-soft and VEL-IVF (VEL with an exact scan for N <= 10^4).
RSS is read before opening the store (after loading the question analyses and J-BRAIN's input embeddings, which belong to J-BRAIN, not
to the memory), after opening it, and after each system's queries. The memory-side cost is the increase. Output: results/s7/rss.json"""
from __future__ import annotations

import gzip
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "bench"))
from common import CKPT, MODELS, RESULTS, tokenizer, write_json  # noqa: E402

WORK = HERE / "features" / "s7"
NS = (10, 100, 1000, 10000, 100000, 1000000)


def worker(N):
    import psutil
    import torch
    import systems as SY
    from analyse import analyse_question
    from diskstore import DiskStore
    from extract import GenderModel
    proc = psutil.Process()
    enc = tokenizer()
    with gzip.open(WORK / "phaseA.pkl.gz", "rb") as f:
        A = pickle.load(f)
    queries = A["queries"][N]
    del A
    import gc
    gc.collect()
    ck = torch.load(MODELS / "heads_seed0.pt", map_location="cpu", weights_only=False)
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["embeddings.weight"].float().numpy()
    gm = GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], raw, enc)
    del ck
    gc.collect()
    m = lambda: proc.memory_info().rss / 2 ** 20  # noqa: E731
    out = {"N": N, "rss_before_open_mb": m()}
    ds = DiskStore(WORK / f"store_{N}", n_buckets=max(1, N // 64))
    ds.load_ivf()
    out["rss_after_open_mb"] = m()

    def hier(q):
        an = analyse_question(ds, q["ids"], q["labels"], 0, np.asarray(q["attr_p"], np.float64), enc, gm)
        if an["status"] != "resolved":
            return None
        p = an["attr_p"]
        for ai in np.argsort(-p)[:4]:
            v = ds.get(an["eids"][0], SY.ATTR_IDS[ai])
            if v is not None:
                return v
        return None

    def ibf(q):
        an = analyse_question(ds, q["ids"], q["labels"], 0, np.asarray(q["attr_p"], np.float64), enc, gm)
        if an["status"] != "resolved":
            return None
        p = an["attr_p"]
        r = ds.ibf_read({an["eids"][0]: 1.0}, {SY.ATTR_IDS[i]: float(p[i]) for i in range(len(p)) if p[i] > 1e-3})
        return r[0] if r else None

    def dense(q):
        return ds.dense_topk(q["dense"], 5, nprobe=16 if ds.ivf is not None else None)
    for name, fn in (("HIER", hier), ("IBF-soft", ibf), ("dense-channel" + ("-IVF" if ds.ivf is not None else "-exact"), dense)):
        t = time.perf_counter()
        for q in queries:
            fn(q)
        out[f"rss_after_{name}_mb"] = m()
        out[f"{name}_seconds"] = time.perf_counter() - t
    out["peak_wset_mb"] = proc.memory_info().peak_wset / 2 ** 20
    ds.close()
    print(json.dumps(out))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
        return
    res = {}
    for N in NS:
        if not (WORK / f"store_{N}").exists():
            continue
        r = subprocess.run([sys.executable, __file__, "--worker", str(N)], capture_output=True, text=True,
                           env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(r.stderr[-2000:])
            continue
        res[N] = json.loads(line[-1])
        d = res[N]
        print(N, {k: round(v, 1) for k, v in d.items() if isinstance(v, float)}, flush=True)
    write_json(RESULTS / "s7" / "rss.json", res, overwrite=True)


if __name__ == "__main__":
    main()
