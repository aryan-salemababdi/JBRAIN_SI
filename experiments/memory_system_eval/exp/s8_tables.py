"""Stage 8: aggregate saved results into the report tables (mean and 95% t-interval over seeds). Reads only results/*.json.
Output: results/s8/tables.json and results/s8/tables.md"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import RESULTS, write_json  # noqa: E402

SYS_ORDER = ["S1", "S2", "S2L", "S3", "S4", "HIER", "IBF-hash", "IBF-hash-analytic", "IBF-soft", "IBF-soft-analytic", "IBF-hash-bundle",
             "VEL-feat", "VEL-JB", "Hybrid-feat", "ORACLE"]
CHAIN_KEYS = ["top1", "accept_rate", "exact", "exact_given_accepted", "far", "frr", "substitution", "stale", "unknown_precision",
              "unknown_recall", "entity_acc", "attribute_acc", "binding_acc", "binding_error"]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = float(np.mean(xs))
    h = float(stats.t.ppf(0.975, len(xs) - 1) * np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None
    return {"mean": m, "ci95": h, "n_seeds": len(xs)}


def fmt(x, pct=True, ci=True):
    if x is None:
        return "–"
    m = x["mean"] * (100 if pct else 1)
    if ci and x["ci95"] is not None:
        return f"{m:.1f} ± {x['ci95'] * (100 if pct else 1):.1f}" if pct else f"{m:.3f} ± {x['ci95']:.3f}"
    return f"{m:.1f}" if pct else f"{m:.3f}"


def main():
    R = json.loads((RESULTS / "s5" / "metrics.json").read_text(encoding="utf-8"))
    res = dict(R["results"])
    vjb = RESULTS / "s5" / "metrics_vjb.json"
    if vjb.exists():
        res.update(json.loads(vjb.read_text(encoding="utf-8"))["results"])
    jo = json.loads((RESULTS / "s5" / "jbrain_only.json").read_text(encoding="utf-8"))["results"]
    out = {"main": {}, "by_family": {}, "by_kind": {}, "by_tag": {}, "ece": {}, "aurc": {}}
    md = []
    for mode in ("oracle", "system"):
        for split in ("test_id", "test_ood"):
            md.append(f"\n### {mode}-write, {split}\n")
            md.append("| System | top-1 | exact | exact given accepted | FAR | FRR | substitution | binding error | UNKNOWN precision | ECE (raw) |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for s in SYS_ORDER:
                runs = [v for k, v in res.items() if k.startswith(f"{s}/{mode}/")]
                if not runs:
                    continue
                agg = {k: ms([r[split]["chain"].get(k) for r in runs]) for k in CHAIN_KEYS}
                agg["ece_raw"] = ms([r[split].get("ece_raw") for r in runs])
                agg["ece_platt_val"] = ms([r[split].get("ece_platt_val") for r in runs])
                agg["aurc"] = ms([r[split]["risk_coverage"]["aurc"] for r in runs])
                out["main"][f"{s}|{mode}|{split}"] = agg
                fams = sorted(runs[0][split]["by_family"])
                out["by_family"][f"{s}|{mode}|{split}"] = {f: ms([r[split]["by_family"][f]["exact"] for r in runs]) for f in fams}
                kinds = sorted(runs[0][split]["by_kind"])
                out["by_kind"][f"{s}|{mode}|{split}"] = {k: ms([r[split]["by_kind"][k]["far"] if k != "answerable" else
                                                                r[split]["by_kind"][k]["exact"] for r in runs]) for k in kinds}
                tags = sorted(runs[0][split]["by_tag"])
                out["by_tag"][f"{s}|{mode}|{split}"] = {t: ms([r[split]["by_tag"][t]["exact"] for r in runs if t in r[split]["by_tag"]])
                                                        for t in tags}
                md.append(f"| {s} | {fmt(agg['top1'])} | {fmt(agg['exact'])} | {fmt(agg['exact_given_accepted'])} | {fmt(agg['far'])} | "
                          f"{fmt(agg['frr'])} | {fmt(agg['substitution'])} | {fmt(agg['binding_error'])} | {fmt(agg['unknown_precision'])} | "
                          f"{fmt(agg['ece_raw'], pct=False)} |")
            if mode == "oracle":
                c = jo[split]["chain"]
                md.append(f"| J-BRAIN only (no memory, reads the stream) | – | {100 * c['exact']:.1f} | – | {100 * c['far']:.1f} | "
                          f"{100 * c['frr']:.1f} | – | – | – | – |")
    out["jbrain_only"] = jo
    out["mcnemar_seed0"] = R.get("mcnemar_seed0")
    ext = RESULTS / "s5" / "extraction.json"
    if ext.exists():
        E = json.loads(ext.read_text(encoding="utf-8"))
        out["extraction"] = {split: {k: ms([E[f"{split}/seed{s}"][k] for s in (0, 1, 2) if f"{split}/seed{s}" in E])
                                     for k in ("record_precision", "record_recall", "span_precision", "span_recall", "intent_acc")}
                             for split in ("val", "test_id", "test_ood")}
        md.append("\n### Extraction (system-write)\n")
        md.append("| Split | record precision | record recall | span precision | span recall | intent accuracy |")
        md.append("|---|---|---|---|---|---|")
        for split, v in out["extraction"].items():
            md.append(f"| {split} | " + " | ".join(fmt(v[k]) for k in ("record_precision", "record_recall", "span_precision", "span_recall",
                                                                       "intent_acc")) + " |")
    (RESULTS / "s8").mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "s8" / "tables.json", out, overwrite=True)
    (RESULTS / "s8" / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
