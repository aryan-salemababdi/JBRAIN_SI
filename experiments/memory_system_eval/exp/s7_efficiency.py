"""Stage 7 (Experiment 6): on-device efficiency and scaling of the on-disk memory, N = 10 ... 10^6 records.

Phase A (main process, GPU): a master list of ~1.05M person records (all first x last name pairs, 5 attributes each, values from
all pools), nested stores N = first N records; queries per N (answerable: sampled stored records; absent: near-miss attributes of
stored entities and unknown persons); question analysis with the seed-0 heads; dense record keys: J-BRAIN key features -> dense
head for the first 10^5 records, 'embedding-only' keys (mean input embedding, fixed random projection) beyond (documented: the dense
channel's accuracy is not meaningful past 10^5, its cost is). Also J-BRAIN cost per question / per note line on GPU (batch 1,
batch 64, with nvidia-smi power sampling) and on CPU (batch 1).
Phase B (one fresh subprocess per N, CPU only): build the disk store (insert throughput), disk bytes, RSS, per-query latency
(p50/p95) and accuracy for DICT (in-RAM dict), HIER, IBF-soft, BM25, VEL (exact dense scan) and VEL-IVF, bytes read per query,
update/delete latency and correctness. Output: results/s7/efficiency.json"""
from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "bench"))
sys.path.insert(0, str(HERE / "exp"))
from common import CKPT, FEAT, MODELS, RESULTS, tokenizer, write_json  # noqa: E402

NS = (10, 100, 1000, 10000, 100000, 1000000)
WORK = HERE / "features" / "s7"
OUT = RESULTS / "s7"
N_ANS, N_ABS = 400, 133


# ------------------------------------------------------------------------------------------------------------------- phase A
def master_records(seed=0):
    import gen_membench as G
    import lexicons as LX
    rng = np.random.default_rng(seed)
    firsts = [(f, "f") for f in LX.FEMALE] + [(m, "m") for m in LX.MALE]
    names = [(f, g, l) for f, g in firsts for l in G.LAST]
    names = [names[i] for i in rng.permutation(len(names))]
    reserve, names = names[:2000], names[2000:]
    pairs = []
    for ei in range(len(names)):
        for a in rng.choice(G.PERSON_ATTRS, size=5, replace=False):
            pairs.append((ei, str(a)))
    pairs = [pairs[i] for i in rng.permutation(len(pairs))][:NS[-1]]
    splits = G.SPLITS
    vals = [G.draw_value(rng, LX.ATTRS[a]["vtype"], splits[int(rng.integers(4))]) for _, a in pairs]
    return names, reserve, pairs, vals


def make_queries(N, names, reserve, pairs, vals, rng):
    import lexicons as LX
    held = {}
    for ei, a in pairs[:N]:
        held.setdefault(ei, set()).add(a)
    qs = []
    for i in rng.choice(N, min(N_ANS, N), replace=False if N >= N_ANS else True):
        ei, a = pairs[int(i)]
        qs.append({"ei": ei, "attr": a, "answerable": True, "value": vals[int(i)]})
    ents = list(held)
    for j in range(N_ABS):
        if j % 2 == 0:
            ei = ents[int(rng.integers(len(ents)))]
            cand = [a for a in LX.ATTRS if LX.ATTRS[a]["owner"] == "person" and a not in held[ei]]
            qs.append({"ei": ei, "attr": str(rng.choice(cand)), "answerable": False, "kind": "near_miss"})
        else:
            r = reserve[int(rng.integers(len(reserve)))]
            qs.append({"ei": None, "name": r, "attr": str(rng.choice([a for a in LX.ATTRS if LX.ATTRS[a]["owner"] == "person"])),
                       "answerable": False, "kind": "unknown_entity"})
    for q in qs:
        f, g, l = names[q["ei"]] if q["ei"] is not None else q["name"]
        a = LX.ATTRS[q["attr"]]
        u = rng.random()
        if u < 0.4:
            t = str(rng.choice(LX.GEN_Q["train"])).replace("{alias}", str(rng.choice(a["alias_tr"])))
        else:
            t = str(rng.choice(a["q_tr"]))
        pron = {"pos": "her" if g == "f" else "his", "obj": "her" if g == "f" else "him", "sub": "she" if g == "f" else "he"}
        q["text"] = t.replace("{N}", f"{f} {l}").replace("{pos}", pron["pos"]).replace("{obj}", pron["obj"])
        q["name_str"] = f"{f} {l}"
    return qs


def gpu_phase(enc):
    import torch
    import jbrain_frozen as J
    from heads import pack_features, q_pool
    from s4_infer import apply_mlp, load_heads, tag
    import lexicons as LX
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / "phaseA.pkl.gz").exists():
        return
    names, reserve, pairs, vals = master_records()
    rng = np.random.default_rng(1)
    queries = {N: make_queries(N, names, reserve, pairs, vals, rng) for N in NS}
    h, ck = load_heads(0)
    norm = (ck["norm_mu"], ck["norm_sd"])
    model = J.load_backbone()
    allq = [q for N in NS for q in queries[N]]
    ids = [enc.encode(q["text"], disallowed_special=()) for q in allq]
    Xq, oq, lq = pack_features(model, ids, norm=norm)
    qp = q_pool(Xq, oq, lq, [0] * len(ids))
    tags = tag(h["tagger"], Xq, oq, lq)
    ap = apply_mlp(h["q_attr"], qp)
    with torch.no_grad():
        qd = h["dense"].forward_q(torch.as_tensor(qp, device=J.DEV)).cpu().numpy()
    for q, i, t, p, d in zip(allq, ids, tags, ap, qd):
        q.update({"ids": i, "labels": t, "attr_p": p.astype(np.float32), "dense": d.astype(np.float32)})
    # dense keys: J-BRAIN features for the first 1e5 records, embedding-only beyond
    n_real = 100000
    kt = [f"{names[ei][0]} {names[ei][2]} {LX.ATTRS[a]['canon']}" for ei, a in pairs[:n_real]]
    Xk, ok, lk = pack_features(model, [enc.encode(t, disallowed_special=()) for t in kt], norm=norm, chunk=8192)
    kx = np.stack([np.concatenate([Xk[o:o + n].astype(np.float32).mean(0), Xk[o + n - 1].astype(np.float32)]) for o, n in zip(ok, lk)])
    del Xk
    with torch.no_grad():
        keys_real = np.concatenate([h["dense"].forward_r(torch.as_tensor(kx[s:s + 8192], device=J.DEV)).cpu().numpy()
                                    for s in range(0, len(kx), 8192)])
    E = model.embeddings.weight.detach().float().cpu().numpy()
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    proj = np.random.default_rng(7).standard_normal((E.shape[1], 256)).astype(np.float32) / math.sqrt(E.shape[1])
    rest = []
    for ei, a in pairs[n_real:]:
        v = En[enc.encode(f"{names[ei][0]} {names[ei][2]} {LX.ATTRS[a]['canon']}")].mean(0) @ proj
        rest.append(v / (np.linalg.norm(v) + 1e-9))
    keys = np.concatenate([keys_real, np.array(rest, dtype=np.float32)]).astype(np.float16)
    np.save(WORK / "dense_keys.npy", keys)
    cost = jbrain_cost(model, h, enc, norm)
    with gzip.open(WORK / "phaseA.pkl.gz", "wb") as f:
        pickle.dump({"names": names, "reserve": reserve, "pairs": pairs, "vals": vals, "queries": queries, "jbrain_cost": cost}, f)


def power_sampler(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=5)
            out.append((time.perf_counter(), float(r.stdout.strip().splitlines()[0])))
        except Exception:
            pass
        time.sleep(0.1)


def jbrain_cost(model, h, enc, norm):
    """Per-question analysis and per-line extraction cost: J-BRAIN forward (blocks 0-7 needed for features; the full model is run)
    + heads. GPU batch 1 / batch 64 with power sampling; CPU batch 1 (fp32)."""
    import torch
    import jbrain_frozen as J
    from heads import pack_features
    import lexicons as LX
    qtext = [f"What is Sara Kim's {a['canon']}?" for a in LX.ATTRS.values()] * 10
    ltext = [f"You can reach Sara Kim at 617-555-{1000 + i}." for i in range(220)]
    res = {}
    for label, texts in (("question", qtext), ("line", ltext)):
        ids = [enc.encode(t) for t in texts]
        ntok = float(np.mean([len(x) for x in ids]))
        for bs, trunc in ((1, False), (1, True), (64, False), (64, True)):
            torch.cuda.synchronize()
            idle = []
            stop = threading.Event()
            th = threading.Thread(target=power_sampler, args=(stop, idle))
            th.start()
            time.sleep(1.5)
            stop.set()
            th.join()
            samples = []
            stop = threading.Event()
            th = threading.Thread(target=power_sampler, args=(stop, samples))
            th.start()
            t0 = time.perf_counter()
            reps = 0
            while time.perf_counter() - t0 < 6.0:
                for s in range(0, len(ids), bs):
                    pack_features(model, ids[s:s + bs], norm=norm, truncate=trunc)
                    reps += len(ids[s:s + bs])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stop.set()
            th.join()
            p_idle = float(np.median([p for _, p in idle])) if idle else None
            p_run = float(np.mean([p for _, p in samples])) if samples else None
            res[f"{label}_gpu_bs{bs}_{'trunc7' if trunc else 'full'}"] = {"items": reps, "tokens_per_item": ntok, "ms_per_item": 1000 * dt / reps,
                                          "power_idle_w": p_idle, "power_run_w": p_run,
                                          "energy_mj_per_item_above_idle": (None if p_idle is None else
                                                                             1000 * (p_run - p_idle) * dt / reps),
                                          "power_samples": len(samples)}
    cpu = J.load_backbone(device="cpu", verify_hash=False)
    torch.set_num_threads(8)
    for label, texts in (("question", qtext[:22]), ("line", ltext[:22])):
        ids = [enc.encode(t) for t in texts]
        for stop in (None, 7):
            with torch.no_grad():
                J.run_layers(cpu, torch.tensor([ids[0]]), None, (5, 7), stop_after=stop)
                t0 = time.perf_counter()
                for x in ids:
                    J.run_layers(cpu, torch.tensor([x]), None, (5, 7), stop_after=stop)
                dt = time.perf_counter() - t0
            res[f"{label}_cpu_bs1_8threads_{'trunc7' if stop else 'full'}"] = {"items": len(ids), "ms_per_item": 1000 * dt / len(ids),
                                                                               "tokens_per_item": float(np.mean([len(x) for x in ids]))}
    del cpu
    return res


# ------------------------------------------------------------------------------------------------------------------- phase B
def worker(N):
    import psutil
    import lexicons as LX
    from analyse import analyse_question
    from diskstore import DiskStore
    from extract import GenderModel
    import systems as SY
    import torch
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    enc = tokenizer()
    A = pickle.load(gzip.open(WORK / "phaseA.pkl.gz", "rb"))
    names, pairs, vals, queries = A["names"], A["pairs"], A["vals"], A["queries"][N]
    keys = np.load(WORK / "dense_keys.npy", mmap_mode="r")
    ck = torch.load(MODELS / "heads_seed0.pt", map_location="cpu", weights_only=False)
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["embeddings.weight"].float().numpy()
    gm = GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], raw, enc)
    vel = pickle.load(open(MODELS / "verifiers_oracle_seed0.pkl", "rb"))["VEL-feat"]
    m5 = json.loads((RESULTS / "s5" / "metrics.json").read_text(encoding="utf-8"))["results"]
    cal_h, cal_v = m5["HIER/oracle/seed0"]["calibration"], m5["VEL-feat/oracle/seed0"]["calibration"]

    def lg(p):
        p = min(max(float(p), 1e-12), 1 - 1e-12)
        return math.log(p / (1 - p))
    path = WORK / f"store_{N}"
    import shutil
    if path.exists():
        shutil.rmtree(path)
    B = max(1, N // 64)
    rss_pre = proc.memory_info().rss
    ds = DiskStore(path, n_buckets=B)
    t0 = time.perf_counter()
    seen = set()
    for i in range(N):
        ei, a = pairs[i]
        f, g, l = names[ei]
        eid = f"p{ei}"
        if eid not in seen:
            ds.add_entity(eid, f"{f} {l}", f, l, g)
            seen.add(eid)
        alias = LX.ATTRS[a]["alias_tr"][0]
        ds.write(eid, a, enc.encode(" " + vals[i]), i, cued=False,
                 doc_text=f"{f} {l} {LX.ATTRS[a]['canon']} {f} {l}'s {alias} is {vals[i]}.", dense_vec=keys[i])
        if (i + 1) % 20000 == 0:
            ds.commit()
    ds.commit()
    build_s = time.perf_counter() - t0
    ivf_s = None
    if N >= 10000:
        t1 = time.perf_counter()
        ds.build_ivf(n_lists=min(1024, N // 40))
        ivf_s = time.perf_counter() - t1
    ds.close()
    disk = {k: v for k, v in DiskStore(path, n_buckets=B).disk_bytes().items()}
    del raw
    import gc
    gc.collect()
    ds = DiskStore(path, n_buckets=B)
    ds.load_ivf()
    rss_open = proc.memory_info().rss
    dict_store = None
    out = {"N": N, "buckets": B, "build_seconds": build_s, "insert_per_s": N / build_s, "ivf_build_seconds": ivf_s, "disk_bytes": disk,
           "rss_baseline_mb": rss0 / 2 ** 20, "rss_before_open_mb": rss_pre / 2 ** 20, "rss_after_open_mb": rss_open / 2 ** 20,
           "systems": {}}

    def analysed(q):
        an = analyse_question(ds, q["ids"], q["labels"], 0, np.asarray(q["attr_p"], np.float64), enc, gm)
        return an

    def gold(q):
        return (f"p{q['ei']}", q["attr"]) if q["answerable"] else None

    def run(name, fn, qset):
        io0 = proc.io_counters().read_bytes
        lat, ok, far, n_abs = [], 0, 0, 0
        ent_err = 0
        for q in qset:
            t = time.perf_counter()
            res = fn(q)
            lat.append(time.perf_counter() - t)
            if q["answerable"]:
                ok += res is not None and res[1] == enc.encode(" " + q["value"]) and res[0] == gold(q)
                ent_err += res is not None and res[0] != gold(q)
            else:
                n_abs += 1
                far += res is not None
        io1 = proc.io_counters().read_bytes
        lat = np.array(lat) * 1000
        n_ans = sum(q["answerable"] for q in qset)
        out["systems"][name] = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                                "mean_ms": float(lat.mean()), "first_pass_p95_ms": None, "exact": ok / n_ans,
                                "binding_or_wrong_record_rate": ent_err / n_ans, "far": far / max(n_abs, 1),
                                "read_bytes_per_query": (io1 - io0) / len(qset), "rss_mb_after": proc.memory_info().rss / 2 ** 20,
                                "n_queries": len(qset), "n_answerable": n_ans}

    # DICT reference: in-RAM dict keyed by (full name, attribute) - perfect analysis assumed for names, attribute = argmax
    rss_d0 = proc.memory_info().rss
    dict_store = {}
    for i in range(N):
        ei, a = pairs[i]
        dict_store[(f"{names[ei][0]} {names[ei][2]}", a)] = enc.encode(" " + vals[i])
    out["dict_rss_mb"] = (proc.memory_info().rss - rss_d0) / 2 ** 20

    def f_dict(q):
        a = SY.ATTR_IDS[int(np.argmax(q["attr_p"]))]
        v = dict_store.get((q["name_str"], a))
        return None if v is None else ((f"p{q['ei']}" if q["ei"] is not None else None, a), v)

    def f_hier(q):
        an = analysed(q)
        if an["status"] != "resolved":
            return None
        e = an["eids"][0]
        p = an["attr_p"]
        hits = []
        for ai in np.argsort(-p)[:4]:
            v = ds.get(e, SY.ATTR_IDS[ai])
            if v is not None:
                hits.append((SY.ATTR_IDS[ai], float(p[ai]), v))
        if not hits:
            return None
        s1, s2 = lg(hits[0][1]), (lg(hits[1][1]) if len(hits) > 1 else lg(0.0))
        return ((e, hits[0][0]), hits[0][2]) if (s1 >= cal_h["tau"] and s1 - s2 >= cal_h["delta"]) else None

    def f_ibf(q):
        an = analysed(q)
        if an["status"] != "resolved":
            return None
        p = an["attr_p"]
        res = ds.ibf_read({an["eids"][0]: 1.0}, {SY.ATTR_IDS[i]: float(p[i]) for i in range(len(p)) if p[i] > 1e-3})
        if not res or res[0][2] < 1.0:
            return None
        (e, a), rid, s = res[0]
        if e != an["eids"][0] or p[SY.ATTR_IDS.index(a)] < 0.5:
            return None
        return (e, a), ds.record(rid)["payload"]

    def f_bm25(q):
        r = ds.bm25(SY.words(q["text"]), 1)
        if not r:
            return None
        rec = ds.record(r[0][0])
        return (rec["eid"], rec["attr"]), rec["payload"]

    def make_vel(nprobe):
        def f_vel(q):
            an = analysed(q)
            p = an["attr_p"]
            qw = SY.words(q["text"])
            lex = ds.bm25(qw, 50)
            dense = ds.dense_topk(q["dense"], 5, nprobe=nprobe)
            cand = {rid for rid, _ in lex[:5]} | {rid for rid, _ in dense}
            if an["status"] in ("resolved", "ambiguous"):
                for e in an["eids"]:
                    for ai in np.argsort(-p)[:4]:
                        r = ds.db.execute("SELECT id FROM records WHERE eid=? AND attr=? AND status=1", (e, SY.ATTR_IDS[ai])).fetchone()
                        if r:
                            cand.add(r[0])
            if not cand:
                return None
            bm = {rid: s for rid, s in lex}
            bmr = {rid: i for i, (rid, _) in enumerate(lex)}
            dr = {rid: i for i, (rid, _) in enumerate(dense)}
            qs = set(qw)
            amax = SY.ATTR_IDS[int(np.argmax(p))]
            rows, recs = [], []
            for rid in cand:
                rec = ds.record(rid)
                e, a = rec["eid"], rec["attr"]
                nw = SY.words(ds.entities[e]["name"])
                pa = float(p[SY.ATTR_IDS.index(a)])
                rows.append([bm.get(rid, 0.0), 1 / (1 + bmr[rid]) if rid in bmr else 0.0, float(np.asarray(ds.dense[rid], np.float32) @ q["dense"]),
                             1 / (1 + dr.get(rid, 5)), pa, math.log(pa + 1e-6), float(a == amax),
                             float(an["status"] == "resolved" and e in an["eids"]), float(an["status"] == "ambiguous" and e in an["eids"]),
                             float(an["status"] == "unknown"), 0.0, len(set(nw) & qs) / len(nw), float(rec["conflict"]),
                             math.log(rec["nver"]), float(an["status"] in ("resolved", "ambiguous") and e in an["eids"])])
                recs.append(rec)
            pr = vel.predict_proba(np.array(rows, dtype=np.float32))[:, 1]
            o = np.argsort(-pr)
            s1, s2 = lg(pr[o[0]]), (lg(pr[o[1]]) if len(o) > 1 else lg(0.0))
            if not (s1 >= cal_v["tau"] and s1 - s2 >= cal_v["delta"]):
                return None
            r = recs[o[0]]
            return (r["eid"], r["attr"]), r["payload"]
        return f_vel

    systems = [("DICT", f_dict), ("HIER", f_hier), ("IBF-soft", f_ibf), ("BM25", f_bm25), ("VEL", make_vel(None))]
    if ds.ivf is not None:
        systems.append(("VEL-IVF", make_vel(16)))
    sub = [q for q in queries if q["answerable"]][:100] + [q for q in queries if not q["answerable"]][:33]
    for name, fn in systems:
        qset = sub if (N >= 100000 and name in ("BM25", "VEL", "VEL-IVF")) else queries
        first = []
        for q in qset[:30]:
            t = time.perf_counter()
            fn(q)
            first.append(time.perf_counter() - t)
        run(name, fn, qset)
        out["systems"][name]["first_pass_p95_ms"] = float(np.percentile(np.array(first) * 1000, 95))
    # updates and deletes (cost and correctness)
    rng = np.random.default_rng(3)
    idx = rng.choice(N, min(200, N), replace=False)
    up, de, ok_u, ok_d = [], [], 0, 0
    for i in idx[: len(idx) // 2]:
        ei, a = pairs[int(i)]
        newv = enc.encode(" UPDATED-" + str(int(i)))
        t = time.perf_counter()
        ds.write(f"p{ei}", a, newv, N + int(i), cued=True, doc_text=f"updated {i}")
        ds.commit()
        up.append(time.perf_counter() - t)
        ok_u += ds.get(f"p{ei}", a) == newv
    for i in idx[len(idx) // 2:]:
        ei, a = pairs[int(i)]
        t = time.perf_counter()
        ds.delete(f"p{ei}", a, N + int(i))
        ds.commit()
        de.append(time.perf_counter() - t)
        ok_d += ds.get(f"p{ei}", a) is None and all(r[0][0] != f"p{ei}" or r[0][1] != a
                                                    for r in ds.ibf_read({f"p{ei}": 1.0}, {a: 1.0}) if r[2] >= 1.0)
    out["update"] = {"n": len(up), "p50_ms": float(np.percentile(np.array(up) * 1000, 50)), "p95_ms": float(np.percentile(np.array(up) * 1000, 95)),
                     "correct": ok_u / max(1, len(up))}
    out["delete"] = {"n": len(de), "p50_ms": float(np.percentile(np.array(de) * 1000, 50)), "p95_ms": float(np.percentile(np.array(de) * 1000, 95)),
                     "correct": ok_d / max(1, len(de))}
    out["rss_peak_mb"] = proc.memory_info().peak_wset / 2 ** 20 if hasattr(proc.memory_info(), "peak_wset") else None
    out["rss_end_mb"] = proc.memory_info().rss / 2 ** 20
    ds.close()
    write_json(OUT / f"worker_{N}.json", out, overwrite=True)
    print(json.dumps({k: v for k, v in out.items() if k != "disk_bytes"})[:2000], flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
        return
    from common import check_resources
    print(check_resources(need_disk_gb=6, need_ram_gb=6, need_vram_gb=3, label="stage 7"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    enc = tokenizer()
    t0 = time.time()
    gpu_phase(enc)
    print("phase A done", round(time.time() - t0), flush=True)
    res = {}
    for N in NS:
        if (OUT / f"worker_{N}.json").exists():
            res[N] = json.loads((OUT / f"worker_{N}.json").read_text(encoding="utf-8"))
            continue
        r = subprocess.run([sys.executable, __file__, "--worker", str(N)], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
        res[N] = json.loads((OUT / f"worker_{N}.json").read_text(encoding="utf-8"))
    A = pickle.load(gzip.open(WORK / "phaseA.pkl.gz", "rb"))
    write_json(OUT / "efficiency.json", {"per_N": res, "jbrain_cost": A["jbrain_cost"], "seconds": round(time.time() - t0, 1)}, overwrite=True)


if __name__ == "__main__":
    main()
