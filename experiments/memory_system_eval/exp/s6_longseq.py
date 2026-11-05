"""Stage 6 (Experiment 3): accuracy against sequence length (512-32,768 tokens), target position and distance, two regimes.
Write modes: oracle-write; system-write with LOCAL features (every line re-encoded from a fresh state, as trained); system-write
with STREAMING features (block outputs of one chunked pass over the whole stream: free at write time, but beyond J-BRAIN's 512-token
training window). Systems: S2L, S3, HIER, IBF-hash, IBF-soft (entity-partitioned; plus a single-field ablation), VEL-feat,
Hybrid-feat, ORACLE (seeds 0-2; heads, verifiers and thresholds from Stages 4-5, no re-calibration), and J-BRAIN-only (greedy).
Accuracy = payload exact AND correct entity AND correct attribute (values may repeat across facts in dense long streams).
GPU part cached per file in features/longseq_*.pkl.gz (resumable). Output: results/s6/longseq.json"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jbrain_frozen as J  # noqa: E402
from analyse import analyse_question  # noqa: E402
from common import CKPT, DATA, FEAT, MODELS, RESULTS, SEEDS, check_resources, tokenizer, write_json  # noqa: E402
from extract import GenderModel, decode_spans, extract_episode  # noqa: E402
from heads import line_pool, pack_features, q_pool, span_feat  # noqa: E402
from ibf import Codes  # noqa: E402
from membench import LX, line_items, question_items  # noqa: E402
from s4_infer import apply_mlp, load_heads, tag  # noqa: E402
from s5_eval import gold_key, row_for  # noqa: E402
from s5_jbrain_only import HEADER  # noqa: E402
from store import oracle_store  # noqa: E402
import systems as SY  # noqa: E402

LENGTHS = (512, 2048, 4096, 8192, 16384, 32768)
REGIMES = ("density", "fixed")
SOURCES = ("local", "stream")
SYSTEMS = ["S2L", "S3", "HIER", "IBF-hash", "IBF-soft", "IBF-soft-single", "IBF-hash-analytic", "IBF-soft-analytic",
           "IBF-soft-single-analytic", "VEL-feat", "Hybrid-feat", "ORACLE"]
DEV = J.DEV
OUT = RESULTS / "s6"


def load_file(regime, L):
    return [json.loads(x) for x in open(DATA / "longseq" / f"{regime}_{L}.jsonl", encoding="utf-8")]


def preds_for_lines(h, X, off, ln):
    tags = tag(h["tagger"], X, off, ln)
    intent = apply_mlp(h["intent"], line_pool(X, off, ln))
    sx, sk = [], []
    for i, labs in enumerate(tags):
        for a, b, typ in decode_spans(labs):
            if typ in ("VAL", "ATTR"):
                sx.append(span_feat(X, off[i], ln[i], a, b))
                sk.append((i, a, b))
    sp = apply_mlp(h["span_attr"], np.stack(sx)) if sx else []
    sa = [dict() for _ in off]
    for (i, a, b), p in zip(sk, sp):
        sa[i][(a, b)] = p.astype(np.float32)
    return [{"labels": tags[i], "intent_p": intent[i], "span_attr": sa[i]} for i in range(len(off))]


@torch.no_grad()
def gpu_part(regime, L, model, heads, norm, enc, nl_any, emb_raw):
    path = FEAT / f"longseq_{regime}_{L}.pkl.gz"
    if path.exists():
        return pickle.load(gzip.open(path, "rb"))
    eps = load_file(regime, L)
    out, t0 = [], time.time()
    mu, sd = torch.tensor(norm[0], device=DEV), torch.tensor(norm[1], device=DEV)
    head_ids = enc.encode(HEADER, disallowed_special=())
    for ep in eps:
        L_items, Q_items = line_items(ep), question_items(ep)
        Xl, ol, ll = pack_features(model, [it["ids"] for it in L_items], norm=norm, max_tokens=4096)
        _, caps = J.prefill_stream(model, ep["ctx_ids"], capture=(5, 7))
        Xs = ((torch.cat([caps[5].float(), caps[7].float()], 1).to(DEV) - mu) / sd).half().cpu().numpy()
        os_ = np.array([ep["lines"][i]["start"] for i in range(len(ep["lines"]))], dtype=np.int64)
        ls_ = np.array([ep["lines"][i]["end"] - ep["lines"][i]["start"] for i in range(len(ep["lines"]))], dtype=np.int64)
        Xq, oq, lq = pack_features(model, [it["ids"] for it in Q_items], norm=norm, max_tokens=4096)
        qp = q_pool(Xq, oq, lq, [it["q_start"] for it in Q_items])
        rec = {"seeds": {}}
        for s in SEEDS:
            h, ck = heads[s]
            gm = GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], emb_raw, enc)
            lp_local = preds_for_lines(h, Xl, ol, ll)
            lp_stream = preds_for_lines(h, Xs, os_, ls_)
            for src, lp in (("local", lp_local), ("stream", lp_stream)):
                for it, p in zip(L_items, lp):
                    p["ids"] = it["ids"]
            qt = tag(h["tagger"], Xq, oq, lq)
            qa = apply_mlp(h["q_attr"], qp)
            qd = h["dense"].forward_q(torch.as_tensor(qp, device=DEV)).cpu().numpy()
            rec["seeds"][s] = {"local": lp_local, "stream": lp_stream,
                               "questions": [{"labels": qt[i], "attr_p": qa[i], "dense": qd[i]} for i in range(len(Q_items))]}
        keys = set()
        for st in [oracle_store(ep, enc)] + [extract_episode(rec["seeds"][s][src], enc, GenderModel(
                heads[s][1]["gender"]["coef"], heads[s][1]["gender"]["intercept"], emb_raw, enc)) for s in SEEDS for src in SOURCES]:
            for k in st.live():
                keys.add((SY.ent_name(st, k[0]), k[1]))
        keys = sorted(keys)
        Xk, ok, lk = pack_features(model, [enc.encode(f"{n} {LX.ATTRS[a]['canon']}", disallowed_special=()) for n, a in keys], norm=norm,
                                   max_tokens=4096)
        kx = np.stack([np.concatenate([Xk[o:o + n].astype(np.float32).mean(0), Xk[o + n - 1].astype(np.float32)]) for o, n in zip(ok, lk)])
        for s in SEEDS:
            rv = heads[s][0]["dense"].forward_r(torch.as_tensor(kx, device=DEV)).cpu().numpy()
            rec["seeds"][s]["rec_vec"] = dict(zip(keys, rv))
        state, _ = J.prefill_stream(model, head_ids + ep["ctx_ids"], keep_caps=False)
        prompts = [{"prompt_ids": q["ctx_ids"] + enc.encode("Q:") + enc.encode(" " + q["q_text"]) + enc.encode("\n") + enc.encode("A:")}
                   for q in ep["questions"]]
        rec["jbrain_only"] = [a["generated"] for a in J.answer_queries_lm(model, state, prompts, nl_any, max_new=16)]
        out.append(rec)
        del Xl, Xs, Xq, Xk, state
        torch.cuda.empty_cache()
    with gzip.open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"  gpu {regime} {L}: {time.time() - t0:.0f}s", flush=True)
    return out


def cpu_part(regime, L, G, enc, emb_raw, cal, verif, codes):
    eps = load_file(regime, L)
    rows = []
    for ep, rec in zip(eps, G):
        tgt = ep["targets"]
        for s in SEEDS:
            gm = GenderModel(*_gender[s], emb_raw, enc)
            P = rec["seeds"][s]
            for mode in ("oracle",) + tuple(f"system-{src}" for src in SOURCES):
                st = oracle_store(ep, enc) if mode == "oracle" else extract_episode(P[mode.split("-")[1]], enc, gm)
                cmode = "oracle" if mode == "oracle" else "system"
                ix = SY.EpisodeIndex(st, P["rec_vec"], codes)
                ix1 = SY.EpisodeIndex(st, P["rec_vec"], codes, partition=False)
                gold_live = {(r["name"] if r["eid"] != "self" else "my", r["attr"]) for r in ep["records"] if r["status"] == "live"}
                for qi, (it, q) in enumerate(zip(question_items(ep), ep["questions"])):
                    pq = P["questions"][qi]
                    an = analyse_question(st, it["ids"], pq["labels"], it["q_start"], np.asarray(pq["attr_p"], np.float64), enc, gm)
                    qq = {"ep": 0, "qi": qi, "words": SY.words(q["ctx_text"] + " " + q["q_text"]), "an": an,
                          "dense_vec": np.asarray(pq["dense"], np.float32), "gold_key": gold_key(st, q, cmode),
                          "answerable": q["answerable"], "kind": q["absent_type"] or "answerable", "family": q["family"], "tags": q["tags"],
                          "payload": q["payload_ids"], "old_payloads": [], "gold_name": q["gold_name"], "attr": q["attr"]}
                    res = {"S2L": SY.s2l(ix, qq), "S3": SY.s3(ix, qq), "HIER": SY.hier(ix, qq), "IBF-hash": SY.ibf(ix, qq, "bind", False),
                           "IBF-soft": SY.ibf(ix, qq, "bind", True), "IBF-soft-single": SY.ibf(ix1, qq, "bind", True),
                           "ORACLE": SY.oracle(ix, qq)}
                    for name, with_ibf in (("VEL-feat", False), ("Hybrid-feat", True)):
                        keys, X = SY.vel_candidates(ix, qq, with_ibf=with_ibf)
                        res[name] = SY.ranked(list(zip(keys, verif[(cmode, s)][name].predict_proba(X)[:, 1].tolist()))) if keys else []
                    for name in ("IBF-hash", "IBF-soft", "IBF-soft-single"):
                        res[name + "-analytic"] = res[name]
                    t = q.get("target")
                    for name, r in res.items():
                        row = row_for(ix, qq, r, name)
                        if name.startswith("IBF") and row["top"] is not None:
                            if not SY.ibf_ledger_ok(qq, tuple(row["top"]), not name.startswith("IBF-hash")):
                                row["s1"] = SY.NEG / 10
                        if name.endswith("-analytic"):           # preregistered analytic rule: r / r*(M_bucket) >= 1 and ledger agreement
                            acc = row["top"] is not None and row["s1"] >= 1.0
                        else:
                            c = cal.get((name if name != "IBF-soft-single" else "IBF-soft", cmode, s))
                            acc = SY.decide(row["top"], row["s1"], row["s2"], c["tau"], c["delta"]) if c else row["top"] is not None
                        rows.append({"regime": regime, "L": L, "seed": s, "mode": mode, "system": name, "answerable": q["answerable"],
                                     "target": t, "frac": tgt[t]["frac_actual"] if t is not None else None,
                                     "distance": tgt[t]["distance"] if t is not None else None, "accepted": bool(acc),
                                     "correct": bool(acc and row["top_exact"] and row["ent_ok"] and row["attr_ok"]),
                                     "exact": bool(acc and row["top_exact"]),
                                     "extracted": bool(t is not None and (("my" if q["eid"] == "self" else q["gold_name"]), q["attr"]) in
                                                       {(SY.ent_name(st, k[0]), k[1]) for k, rr in st.live().items()
                                                        if rr.payload == q["payload_ids"]}),
                                     "n_records": len(ix.keys)})
                ix.close()
                ix1.close()
        for s in SEEDS[:1]:
            for q, gen in zip(ep["questions"], rec["jbrain_only"]):
                txt = enc.decode(gen).strip()
                acc = txt not in ("UNKNOWN", "")
                t = q.get("target")
                rows.append({"regime": regime, "L": L, "seed": 0, "mode": "none", "system": "JBRAIN-only", "answerable": q["answerable"],
                             "target": t, "frac": ep["targets"][t]["frac_actual"] if t is not None else None,
                             "distance": ep["targets"][t]["distance"] if t is not None else None, "accepted": acc,
                             "correct": bool(q["answerable"] and acc and gen == q["payload_ids"]),
                             "exact": bool(q["answerable"] and acc and gen == q["payload_ids"]), "extracted": None,
                             "n_records": None})
    return rows


_gender = {}


def main():
    t0 = time.time()
    print(check_resources(need_disk_gb=1, need_ram_gb=4, need_vram_gb=3, label="stage 6"), flush=True)
    torch.cuda.set_per_process_memory_fraction(0.75)
    enc = tokenizer()
    nl_any = set()
    for i in range(enc.n_vocab):
        try:
            if b"\n" in enc.decode_single_token_bytes(i):
                nl_any.add(i)
        except KeyError:
            pass
    emb_raw = torch.load(CKPT, map_location="cpu", weights_only=True)["embeddings.weight"].float().numpy()
    heads = {s: load_heads(s) for s in SEEDS}
    for s in SEEDS:
        _gender[s] = (heads[s][1]["gender"]["coef"], heads[s][1]["gender"]["intercept"])
    norm = (heads[0][1]["norm_mu"], heads[0][1]["norm_sd"])
    model = J.load_backbone()
    G = {}
    for regime in REGIMES:
        for L in LENGTHS:
            G[(regime, L)] = gpu_part(regime, L, model, heads, norm, enc, nl_any, emb_raw)
            del G[(regime, L)]
    del model
    torch.cuda.empty_cache()
    while not (RESULTS / "s5" / "metrics.json").exists():              # thresholds and verifiers come from Stage 5
        time.sleep(30)
    m5 = json.loads((RESULTS / "s5" / "metrics.json").read_text(encoding="utf-8"))["results"]
    cal = {}
    for key, v in m5.items():
        name, mode, sd = key.split("/")
        cal[(name, mode, int(sd[4:]))] = v["calibration"]
    verif = {(m, s): pickle.load(open(MODELS / f"verifiers_{m}_seed{s}.pkl", "rb")) for m in ("oracle", "system") for s in SEEDS}
    codes = Codes(1024)
    all_rows = []
    for regime in REGIMES:
        for L in LENGTHS:
            ts = time.time()
            Gp = pickle.load(gzip.open(FEAT / f"longseq_{regime}_{L}.pkl.gz", "rb"))
            all_rows += cpu_part(regime, L, Gp, enc, emb_raw, cal, verif, codes)
            print(f"  cpu {regime} {L}: {time.time() - ts:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "rows.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    summ = {}
    import itertools
    keys = sorted({(r["regime"], r["system"], r["mode"]) for r in all_rows})
    for regime, system, mode in keys:
        for L in LENGTHS:
            R = [r for r in all_rows if r["regime"] == regime and r["system"] == system and r["mode"] == mode and r["L"] == L]
            per_seed = {}
            for s in sorted({r["seed"] for r in R}):
                Rs = [r for r in R if r["seed"] == s]
                A = [r for r in Rs if r["answerable"]]
                U = [r for r in Rs if not r["answerable"]]
                per_seed[s] = {"correct": np.mean([r["correct"] for r in A]), "exact": np.mean([r["exact"] for r in A]),
                               "far": np.mean([r["accepted"] for r in U]) if U else None,
                               "by_position": [float(np.mean([r["correct"] for r in A if r["target"] == t])) for t in range(5)],
                               "extracted": float(np.mean([r["extracted"] for r in A])) if A and A[0]["extracted"] is not None else None,
                               "n_answerable": len(A), "n_absent": len(U),
                               "n_records_mean": float(np.mean([r["n_records"] for r in Rs if r["n_records"] is not None])) if Rs and Rs[0]["n_records"] is not None else None}
            summ[f"{regime}|{system}|{mode}|{L}"] = {
                "per_seed": per_seed,
                "correct_mean": float(np.mean([v["correct"] for v in per_seed.values()])),
                "correct_seed_sd": float(np.std([v["correct"] for v in per_seed.values()], ddof=1)) if len(per_seed) > 1 else None,
                "far_mean": float(np.mean([v["far"] for v in per_seed.values() if v["far"] is not None])),
                "by_position_mean": [float(np.mean([v["by_position"][t] for v in per_seed.values()])) for t in range(5)]}
    write_json(OUT / "longseq.json", {"summary": summ, "seconds": round(time.time() - t0, 1)}, overwrite=True)
    for regime in REGIMES:
        print(f"== {regime}")
        for system, mode in sorted({(k.split("|")[1], k.split("|")[2]) for k in summ if k.startswith(regime)}):
            print(f"  {system:<16} {mode:<14}" + " ".join(f"{summ[f'{regime}|{system}|{mode}|{L}']['correct_mean']:.3f}" for L in LENGTHS))


if __name__ == "__main__":
    main()
