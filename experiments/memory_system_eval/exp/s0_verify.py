"""Stage 0: verify claims from earlier J-BRAIN memory experiments against their RAW result files (read-only).
Each check re-derives a number from raw data and compares it with the value stated in the earlier report.
Output: results/s0_verification.json (+ printed table)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OLD, REPO, RESULTS, WAEM, read_json, write_json  # noqa: E402

EXP = REPO / "experiments"
checks = []


def check(cid, claim, source, reported, measured, tol=None, note=""):
    if measured is None:
        status = "unverifiable (raw data missing or unreadable)"
    elif tol is None:
        status = "confirmed" if measured == reported else "differs"
    else:
        ok = all(abs(a - b) <= tol for a, b in zip(reported, measured)) if isinstance(reported, (list, tuple)) else abs(reported - measured) <= tol
        status = "confirmed" if ok else "differs"
    checks.append({"id": cid, "claim": claim, "source": str(source), "reported": reported, "measured": measured, "status": status, "note": note})


def safe(fn):
    try:
        return fn()
    except Exception as e:                                    # noqa: BLE001
        return None


# C01 pretrained LM key recall vs delay (soroush WS2-A)
p = OLD / "soroush-three-experiments/results/ws2/lm_probe_base_ce_4B_fp32.json"
t = safe(lambda: read_json(p)["tasks"]["T1_natural"])
check("C01", "pretrained LM single-key recall falls 0.300 -> 0.123 -> 0.037 -> 0.028 -> 0.020 at delays 0/16/64/256/1024", p,
      [0.300, 0.123, 0.037, 0.028, 0.020], safe(lambda: [round(t[k]["acc"], 3) for k in ("0", "16", "64", "256", "1024")]), tol=0.001)
check("C01b", "linear probe (all layers, pre-cue, held-out distractors) decodes the key at delay 16 with accuracy 1.000, 0.353 at 64",
      p, [1.0, 0.353], safe(lambda: [round(t[n]["probe"]["pre_cue_Lall"]["test_acc"], 3) for n in ("16", "64")]), tol=0.002,
      note=safe(lambda: f"label-permuted control at 16: {t['16']['probe']['pre_cue_all_permuted_labels']['test_acc']}"))

# C02 decisive recall
p = OLD / "jbrain-decisive-recall/results/stage1_decision.json"
d = safe(lambda: read_json(p))
check("C02", "single-key recall learned by J-BRAIN arm A in 3/3 seeds; ~11x slower (median T95 ratio vs RG-LRU)", p, [3, 11.0],
      safe(lambda: [d["confirm"]["A"]["n_learned"], d["speed_ratio_vs_C"]["A"]]), tol=0)

# C03 phase retention
p1 = OLD / "jbrain-phase-retention-study/results/decision_P1.json"
p2 = OLD / "jbrain-phase-retention-study/results/decision_P2.json"
a1 = safe(lambda: read_json(p1)["A"]["A1_speed"])
check("C03", "removing rotation (dphi=0): learned 5/5 paired seeds; median T95 ratio 0.09 (P1)", p1, [5, 0.091],
      safe(lambda: [a1["learned"][1], round(a1["median_ratio"], 3)]), tol=0.001)
e2 = safe(lambda: read_json(p2)["P2_dphi0"]["E2_fixed"])
check("C03b", "P2 (long training delays): dphi=0 raises the 2048/4096 extrapolation score by a median +0.38", p2, 0.381,
      safe(lambda: round(e2["median"], 3)), tol=0.002)

# C04 context investigation
p = OLD / "jbrain-v2-context-investigation/results/s02_lambada.json"
check("C04", "LAMBADA accuracy of the base model 0.0266 (independent scorer)", p, 0.0266,
      safe(lambda: read_json(p)["models"]["base_ce_4B"]["conditions"]["full"]["acc"]), tol=1e-4)
p = OLD / "jbrain-v2-context-investigation/results/s05_bf16_numerics.json"
dead = safe(lambda: round(sum(v["bf16_autocast"]["frac_heads_dead_all_positions"] * 12 for v in read_json(p)["layers"].values())))
check("C04b", "bf16 autocast makes 33/144 heads write nothing at every position", p, 33, dead, tol=0)

# C05 MVP postmortem copy test
p = OLD / "jbrain-mvp-postmortem/results/induction_results.json"
ind = safe(lambda: read_json(p)["models"]["base"]["random_repeat_L32"])
check("C05", "32 random tokens shown twice: 14.2 -> 13.3 nats/token (copy benefit ~0.9 nats)", p, [14.2, 13.3],
      safe(lambda: [round(ind["first_copy_loss"], 1), round(ind["second_copy_loss"], 1)]), tol=0.05)

# C06 retrieval audit binding test
p = EXP / "jbrain-retrieval-audit/analysis/behavior_models.json"
bm = safe(lambda: read_json(p)["models"])
two = safe(lambda: {m: {k: round(v["acc_2way"], 3) for k, v in mv.items() if k.startswith("two_fact")} for m, mv in bm.items()})
check("C06", "two-fact binding test at chance for all 6 model variants (base/SFT x phase on/off): 2-way accuracy 0.47-0.55 for 0-256 filler tokens, n=96 each",
      p, [0.47, 0.55], safe(lambda: [min(min(d.values()) for d in two.values()), max(max(d.values()) for d in two.values())]), tol=0.003,
      note=safe(lambda: f"base_phase_on alone: {two['base_phase_on']}"))

# C07 C1/C3 MQAR16 learned counts (learned = reached 50% training accuracy; eval N=8 also reported)
def c1c3(variant):
    fs = sorted(glob.glob(str(EXP / f"c1c3_multimode_memory/results/mqar16/{variant}_lr0.001_s*.json")))
    rs = [read_json(f) for f in fs]
    return [sum("0.5" in r.get("steps_to_train_acc", {}) for r in rs), len(rs), [round(r["eval"]["N=8"]["acc"], 3) for r in rs]]
for v, rep in (("baseline", 4), ("c1", 0), ("c3", 5), ("c1c3", 3)):
    m = safe(lambda: c1c3(v))
    check(f"C07-{v}", f"MQAR16 runs that learned: {v} {rep}/5", EXP / "c1c3_multimode_memory/results/mqar16", rep,
          None if m is None else m[0], tol=0, note=None if m is None else f"n={m[1]}, eval N=8 acc per seed {m[2]}")

# C08 controlled C1+C3 runs (E0/E1/E3)
def ctl(cond):
    out = []
    for s in (0, 1, 2):
        r = read_json(EXP / f"retrieval_c3_c1c3_transformer/diag_c1c3/controlled/runs/{cond}_s{s}.json")
        out.append(round(r["log"][-1]["task"]["N8/train_symbols"]["acc"], 3))
    return out
for cond, rep in (("E0", [0.020, 0.012, 0.008]), ("E1", [0.107, 0.009, 0.120]), ("E3", [0.018, 0.012, 0.009])):
    check(f"C08-{cond}", f"controlled ~105K C1+C3 {cond}: final MQAR-8 top-1 on training symbols {rep}",
          EXP / "retrieval_c3_c1c3_transformer/diag_c1c3/controlled/runs", rep, safe(lambda: ctl(cond)), tol=0.0015)

# C09 WAEM metrics recomputed from per-query records
def waem_recompute(run, seed):
    recs = [json.loads(x) for x in open(WAEM / f"runs/waem/{run}/seed{seed}/records/test_main.jsonl", encoding="utf-8")]
    a = [r for r in recs if r["answerable"]]
    u = [r for r in recs if not r["answerable"]]
    ex = sum(r["payload_exact"] for r in a) / len(a)
    key = sum(bool(r["key_correct"]) for r in a) / len(a)
    far = sum(r["out_kind"] != "unknown" for r in u) / len(u)
    m = read_json(WAEM / f"runs/waem/{run}/seed{seed}/metrics.json")["sets"]["test_main"]["summary"]
    return [round(ex, 4), round(key, 4), round(far, 4)], [round(m["payload_exact"]["rate"], 4), round(m["key_acc"]["rate"], 4),
                                                         round(m["false_accept"]["rate"], 4)]
for run in ("phase_on_phase128_hid", "phase_on_real256_hid", "phase_on_phase128_emb"):
    for s in (0, 1, 2):
        r = safe(lambda: waem_recompute(run, s))
        check(f"C09-{run}-s{s}", f"WAEM {run} seed {s}: exact/top-1/FAR on test_main as reported in metrics.json", WAEM / f"runs/waem/{run}",
              None if r is None else r[1], None if r is None else r[0], tol=1e-4)

# C10 WAEM LM-answering baselines: 0% exact
def lm_exact(cond, seed):
    recs = [json.loads(x) for x in open(WAEM / f"runs/lm_eval/{cond}/seed{seed}/records/test_main.jsonl", encoding="utf-8")]
    a = [r for r in recs if r["answerable"]]
    return round(sum(r["payload_exact"] for r in a) / len(a), 4)
for cond in ("A0", "A-FT", "C0", "C-FT"):
    m = safe(lambda: [lm_exact(cond, s) for s in (0, 1, 2)])
    check(f"C10-{cond}", f"WAEM LM answering {cond}: 0% exact payload on test_main (3 seeds)", WAEM / f"runs/lm_eval/{cond}", [0.0, 0.0, 0.0], m, tol=0)

# C11 WAEM LM preservation: B identical to A0
def ppl_pair(a, b):
    x = read_json(WAEM / f"results/lm_preservation/ppl_{a}/result.json")
    y = read_json(WAEM / f"results/lm_preservation/ppl_{b}/result.json")
    return [round(x["result"]["perplexity"], 3), round(y["result"]["perplexity"], 3),
            max(abs(p - q) for p, q in zip(x["result"]["per_passage_nll"], y["result"]["per_passage_nll"]))]
check("C11", "WAEM attached (B) leaves LAMBADA passage perplexity identical to A0: 102.944 / 102.944 / max diff 0",
      WAEM / "results/lm_preservation", [102.944, 102.944, 0.0], safe(lambda: ppl_pair("A0", "B")), tol=1e-3)

# C12 WAEM LM probes: gold-vs-competitor preference at chance
def pref():
    r = read_json(WAEM / "results/audit/lm_probes.json")["result"]
    return [round(v["pairwise_preference"]["gold_preferred"], 3) for v in r.values()]
check("C12", "LM gold-vs-competitor preference at chance (0.483-0.501) for A0, C0, A-FT s0, C-FT s0", WAEM / "results/audit/lm_probes.json",
      [0.489, 0.483, 0.501, 0.484], safe(pref), tol=0.001)

if __name__ == "__main__":
    out = RESULTS / "s0_verification.json"
    write_json(out, {"checks": checks}, overwrite=True)
    for c in checks:
        print(f"{c['id']:<34} {c['status']:<12} reported={c['reported']} measured={c['measured']} {(c['note'] or '')[:90]}")
    print({s: sum(c["status"] == s for c in checks) for s in sorted({c["status"] for c in checks})})
