"""Figures for the report, drawn only from saved result files (results/*). Output: figures/*.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
LENGTHS = (512, 2048, 4096, 8192, 16384, 32768)


def load(p):
    p = RES / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_longseq():
    d = load("s6/longseq.json")
    if d is None:
        return
    S = d["summary"]
    show = [("HIER", "system-local"), ("HIER", "system-stream"), ("VEL-feat", "system-local"), ("IBF-soft-analytic", "system-local"),
            ("IBF-soft", "system-local"), ("IBF-soft-single-analytic", "system-local"), ("S2L", "system-local"), ("ORACLE", "oracle"),
            ("JBRAIN-only", "none")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, regime in zip(axes, ("density", "fixed")):
        for system, mode in show:
            ys = [S.get(f"{regime}|{system}|{mode}|{L}", {}).get("correct_mean") for L in LENGTHS]
            if any(y is not None for y in ys):
                ax.plot(LENGTHS, [np.nan if y is None else y for y in ys], marker="o", label=f"{system} ({mode})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, [str(L) for L in LENGTHS])
        ax.set_title(f"Experiment 3: {regime} regime")
        ax.set_xlabel("sequence length (tokens)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("exact payload + correct entity and attribute")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_length.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for system, mode in (("HIER", "system-local"), ("HIER", "system-stream"), ("VEL-feat", "system-local")):
        ys = [S.get(f"density|{system}|{mode}|32768", {}).get("by_position_mean", [np.nan] * 5)]
        ax.plot([5, 25, 50, 75, 95], ys[0], marker="o", label=f"{system} ({mode}), 32,768 tokens")
    ax.set_xlabel("target position in the stream (%)")
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_position_32k.png", dpi=140)
    plt.close(fig)


def fig_efficiency():
    d = load("s7/efficiency.json")
    if d is None:
        return
    P = d["per_N"]
    Ns = sorted(int(n) for n in P)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    comps = {"SQLite (records, entities, versions, FTS5, indexes)": lambda r: sum(v for k, v in r["disk_bytes"].items() if k.startswith("mem.sqlite")),
             "dense keys (float16 memmap)": lambda r: r["disk_bytes"].get("dense.f16", 0),
             "IBF field (complex64 memmap)": lambda r: r["disk_bytes"].get("ibf.c64", 0),
             "IVF index": lambda r: r["disk_bytes"].get("ivf.npz", 0)}
    for label, f in comps.items():
        ax.plot(Ns, [f(P[str(n)]) / 2 ** 20 for n in Ns], marker="o", label=label)
    rss = load("s7/rss.json") or {}
    if rss:
        rn = sorted(int(n) for n in rss)
        for key, lab in (("rss_after_HIER_mb", "RSS increase: open + HIER queries"), ("rss_after_IBF-soft_mb", "... + IBF queries"),
                         (None, "... + VEL dense channel (IVF / exact)")):
            ys = []
            for n in rn:
                r = rss[str(n)]
                k = key or [x for x in r if x.startswith("rss_after_dense")][0]
                ys.append(max(r[k] - r["rss_before_open_mb"], 0.1))
            ax.plot(rn, ys, marker="s", ls="--", label=lab)
    ax.plot(Ns, [P[str(n)]["dict_rss_mb"] for n in Ns], marker="^", ls=":", label="in-RAM dict of the same records (RSS)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("records")
    ax.set_ylabel("MB")
    ax.set_title("Memory and disk vs number of records")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "memory_vs_records.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for system in ("DICT", "HIER", "IBF-soft", "BM25", "VEL", "VEL-IVF"):
        xs = [n for n in Ns if system in P[str(n)]["systems"]]
        if xs:
            ax.plot(xs, [P[str(n)]["systems"][system]["p95_ms"] for n in xs], marker="o", label=f"{system} p95")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("records")
    ax.set_ylabel("memory-side latency per query, p95 (ms, CPU)")
    ax.set_title("Retrieval latency vs memory size (J-BRAIN feature cost excluded)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "latency_vs_records.png", dpi=140)
    plt.close(fig)


def fig_risk_coverage():
    d = load("s5/metrics.json")
    if d is None:
        return
    R = d["results"]
    v = load("s5/metrics_vjb.json")
    if v:
        R.update(v["results"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, split in zip(axes, ("test_id", "test_ood")):
        for s in ("HIER", "IBF-soft", "VEL-feat", "VEL-JB", "Hybrid-feat", "S3"):
            k = f"{s}/system/seed0"
            if k in R:
                rc = R[k][split]["risk_coverage"]
                ax.plot(rc["coverage"], rc["risk"], marker=".", label=f"{s} (AURC {rc['aurc']:.3f})")
        ax.set_title(f"Risk-coverage, system-write, {split} (seed 0)")
        ax.set_xlabel("coverage")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("risk (wrong among answered)")
    fig.tight_layout()
    fig.savefig(FIG / "risk_coverage.png", dpi=140)
    plt.close(fig)


def fig_breakdowns():
    t = load("s8/tables.json")
    if t is None:
        return
    fams = ["canon", "alias", "para", "partial", "pronoun", "multi_pron", "generic_ood"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, split in zip(axes, ("test_id", "test_ood")):
        systems = ["S2L", "HIER", "IBF-soft", "VEL-feat", "Hybrid-feat"]
        w = 0.15
        for i, s in enumerate(systems):
            d = t["by_family"].get(f"{s}|system|{split}", {})
            ys = [d.get(f, {}).get("mean", np.nan) if d.get(f) else np.nan for f in fams]
            ax.bar(np.arange(len(fams)) + i * w, ys, w, label=s)
        ax.set_xticks(np.arange(len(fams)) + 2 * w, fams, rotation=20)
        ax.set_title(f"Exact by question family, system-write, {split}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("exact payload (answerable)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "exact_by_family.png", dpi=140)
    plt.close(fig)
    kinds = ["unknown_entity", "attr_missing", "near_miss", "similar_attr", "similar_name", "deleted", "ambiguous_partial", "mention_only"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, split in zip(axes, ("test_id", "test_ood")):
        systems = ["S2L", "HIER", "IBF-soft", "VEL-feat", "Hybrid-feat"]
        w = 0.15
        for i, s in enumerate(systems):
            d = t["by_kind"].get(f"{s}|system|{split}", {})
            ys = [d.get(k, {}).get("mean", np.nan) if d.get(k) else np.nan for k in kinds]
            ax.bar(np.arange(len(kinds)) + i * w, ys, w, label=s)
        ax.set_xticks(np.arange(len(kinds)) + 2 * w, kinds, rotation=25, fontsize=8)
        ax.set_title(f"False acceptance by absent type, system-write, {split}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("false acceptance rate")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "far_by_absent_type.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_longseq, fig_efficiency, fig_risk_coverage, fig_breakdowns):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:                                         # noqa: BLE001
            print("FAILED", f.__name__, repr(e), file=sys.stderr)
