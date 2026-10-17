"""Command-line runner: `python -m benchmarks {run,verify,compare,historical,list-tasks} ...` (see benchmarks/README.md)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from .common import (REPO_ROOT, SCHEMA_VERSION, CompatibilityError, ValidationError, build_manifest, check_resume_compat, code_sha256, config_hash,
                     canonical_json, diff_configs, environment_info, git_info, read_json, sha256_bytes, sha256_file, utc_now, validate_per_sample,
                     verify_manifest, write_json)

DEFAULT_CHECKPOINT = REPO_ROOT / "Models" / "JBR1_V2_CE" / "jbr_v2_ce_weights.pt"
DEFAULT_OUT_ROOT = REPO_ROOT / "benchmarks" / "runs"
SEEDS = {"random_seed": 0, "numpy_random_seed": 1234, "torch_random_seed": 1234, "fewshot_random_seed": 1234}
log = logging.getLogger("jbrain.bench")

CONTAMINATION_NOTE = ("Training-data overlap with these benchmarks has NOT been checked. The model was pretrained on FineWeb-Edu (sample-10BT, web text); "
                      "ARC, PIQA, SciQ and HellaSwag text could appear in web crawls. No decontamination or n-gram overlap analysis was performed, so overlap is neither "
                      "claimed nor ruled out.")


# ------------------------------------------------------------------ helpers
def _jsonable(o):
    return json.loads(json.dumps(o, default=str))


def dataset_revisions(dataset_path: str) -> dict:
    """Revision(s) of the locally cached Hugging Face dataset snapshot(s), if the standard hub cache layout is present."""
    try:
        from huggingface_hub import constants
        d = Path(constants.HF_HUB_CACHE) / ("datasets--" + dataset_path.replace("/", "--"))
        snaps = sorted(p.name for p in (d / "snapshots").iterdir()) if (d / "snapshots").is_dir() else []
        ref = (d / "refs" / "main").read_text().strip() if (d / "refs" / "main").is_file() else None
        return {"hub_cache_dir": str(d), "snapshots": snaps, "refs_main": ref}
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}


def requests_sha256(samples: list[dict]) -> str:
    from .metrics import _pairs
    rows = [[int(s["doc_id"]), _pairs(s["arguments"])] for s in sorted(samples, key=lambda s: int(s["doc_id"]))]
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def build_config(args, tok_info: dict, harness_version: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "tasks": list(args.tasks),
        "num_fewshot": 0, "batch_size": 1, "limit": args.limit, "bootstrap_iters": 0, **SEEDS,
        "max_length": args.max_length, "precision": args.precision,
        "model_spec": {"vocab_size": args.vocab_size, "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers},
        "tokenizer": {"name": tok_info["name"], "n_vocab": tok_info["n_vocab"], "eot_token": tok_info["eot_token"], "tiktoken_version": tok_info["tiktoken_version"],
                      "probe_encodings_sha256": tok_info["probe_encodings_sha256"], "encode_policy": tok_info["encode_policy"]},
        "scoring": {"method": "lm-eval loglikelihood requests; per-request sum of fp32 log-softmax over continuation tokens",
                    "context_truncation": "left-truncate context so that context+continuation <= max_length; error if the continuation alone does not fit",
                    "special_tokens": "none added; special-token strings in text are encoded as ordinary text",
                    "answer_normalisation": "none (acc); acc_norm divides by choice character length"},
        "harness": {"name": "lm-eval", "version": harness_version},
    }


def _setup_logging(path: Path):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh, sh = logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stderr)
    for h in (fh, sh):
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    return fh


# ------------------------------------------------------------------ run
def cmd_run(args) -> int:
    if not args.allow_network:
        for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            os.environ.setdefault(k, "1")
    import torch
    import lm_eval
    from lm_eval import simple_evaluate
    from .adapter import JBrainEvalLM, RequestCache, load_model, load_tokenizer, tokenizer_info
    from .metrics import METRIC_DEFINITIONS, SUPPORTED_TASKS, aggregate, build_records, consistency

    bad = [t for t in args.tasks if t not in SUPPORTED_TASKS]
    if bad:
        print(f"error: unsupported task(s) {bad}; supported: {list(SUPPORTED_TASKS)}", file=sys.stderr)
        return 2
    args.tasks = list(dict.fromkeys(args.tasks))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda requested but CUDA is unavailable", file=sys.stderr)
        return 2
    if args.precision == "bf16" and device != "cuda":
        print("error: bf16 autocast is only supported on cuda (use --precision fp32 on cpu)", file=sys.stderr)
        return 2

    enc_probe = load_tokenizer()
    tok_info = tokenizer_info(enc_probe)
    cfg = build_config(args, tok_info, lm_eval.__version__)
    chash = config_hash(cfg)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    print(f"hashing checkpoint {ckpt} ...", file=sys.stderr)
    ck_sha = sha256_file(ckpt)
    code_sha = code_sha256()

    if args.resume:
        run_dir = Path(args.resume)
        check_resume_compat(run_dir, cfg, code_sha, ck_sha)
        resumed = True
    else:
        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{chash[:8]}_{ck_sha[:8]}" + (f"_{args.name}" if args.name else "")
        run_dir = Path(args.out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        resumed = False
    (run_dir / "per_task").mkdir(exist_ok=True)
    fh = _setup_logging(run_dir / "run.log")
    log.info("%s run in %s | config_hash %s | checkpoint sha256 %s | code_sha256 %s | device %s", "RESUMING" if resumed else "STARTING", run_dir, chash, ck_sha, code_sha, device)

    status_path = run_dir / "status.json"
    if resumed:
        status = read_json(status_path)
        status["resumes"] = status.get("resumes", 0) + 1
    else:
        write_json(run_dir / "config.json", cfg)
        status = {"state": "running", "created_utc": utc_now(), "code_sha256": code_sha, "config_hash": chash, "resumes": 0}
    env = environment_info(device)
    if not (run_dir / "environment.json").is_file():
        write_json(run_dir / "environment.json", env)
    else:
        old = read_json(run_dir / "environment.json")
        for k in ("torch", "cuda_runtime", "cuda_device_name"):
            if old.get(k) != env.get(k):
                log.warning("environment differs from the original run: %s %r -> %r (results may differ numerically)", k, old.get(k), env.get(k))
        write_json(run_dir / f"environment.resume{status['resumes']}.json", env)
    write_json(status_path, status)

    model, ck_info = load_model(ckpt, device, cfg["model_spec"])
    if ck_info["sha256"] != ck_sha:
        raise CompatibilityError("checkpoint changed while the run was starting")
    ck_info["tokenizer"] = {k: tok_info[k] for k in ("name", "n_vocab", "eot_token")}
    if not (run_dir / "checkpoint.json").is_file():
        write_json(run_dir / "checkpoint.json", ck_info)
    log.info("checkpoint loaded strictly: %d parameters, sha256 %s", ck_info["n_parameters"], ck_sha)

    cache = RequestCache(run_dir / "loglik_cache.jsonl")
    lm = JBrainEvalLM(model, enc_probe, cache, max_length=args.max_length, precision=args.precision, device=device, cross_check_every=args.cross_check_every)
    dtype_probe = lm.probe_compute_dtype()
    log.info("compute dtype probe: %s", dtype_probe)
    if args.precision == "bf16" and not dtype_probe["bf16_actually_executed"]:
        raise CompatibilityError("--precision bf16 requested but no bfloat16 tensor was produced in the probe forward pass; refusing to label results bf16")
    if args.precision == "fp32" and "torch.bfloat16" in dtype_probe["submodule_output_dtype_counts"]:
        raise CompatibilityError("--precision fp32 but bfloat16 activations appeared; refusing")
    ck_info["compute_dtype_probe"] = dtype_probe
    write_json(run_dir / "compute_dtype.json", dtype_probe)
    t_run = time.time()
    task_dirs = {}
    for task in args.tasks:
        marker, jl = run_dir / "per_task" / f"{task}.complete.json", run_dir / "per_task" / f"{task}.jsonl"
        if marker.is_file():
            m = read_json(marker)
            if m.get("config_hash") == chash and jl.is_file() and sha256_file(jl) == m.get("per_sample_sha256"):
                log.info("[%s] already complete (verified); skipping", task)
                task_dirs[task] = m
                continue
            log.warning("[%s] completion marker is stale or inconsistent; re-running (cached requests are reused)", task)
        log.info("[%s] evaluating (limit=%s) ...", task, args.limit)
        n_new0, n_hit0, cc0 = lm.n_new, lm.n_cached_hits, len(lm.cross_checks)
        t0 = time.time()
        res = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, batch_size=1, limit=args.limit, log_samples=True, write_out=False, bootstrap_iters=0, **SEEDS)
        samples = res["samples"][task]
        tcfg = _jsonable(res["configs"][task])
        md = tcfg.get("metadata") or {}
        if "config_source" in md:  # absolute local path; keep only the task yaml name
            md["task_yaml"] = Path(md.pop("config_source")).name
            tcfg["metadata"] = md
        records = build_records(task, tcfg["output_type"], samples, cache.entries)
        n_expected = int(res["n-samples"][task]["effective"])
        validate_per_sample(records, task, n_expected)
        agg = aggregate(task, records)
        cons = consistency(task, agg, res["results"][task], records)
        if not cons["ok"]:
            log.error("[%s] recomputed metrics disagree with lm-eval: %s", task, cons)
        with open(jl, "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        info = {"task": task, "config_hash": chash, "per_sample_sha256": sha256_file(jl), "aggregate": agg, "lm_eval_reported": _jsonable(res["results"][task]),
                "consistency": cons, "n_expected": n_expected, "n_original_docs": int(res["n-samples"][task]["original"]),
                "task_version": res.get("versions", {}).get(task), "task_config": {k: tcfg.get(k) for k in (
                    "dataset_path", "dataset_name", "training_split", "validation_split", "test_split", "fewshot_split", "output_type", "doc_to_text",
                    "doc_to_target", "doc_to_choice", "metric_list", "num_fewshot", "metadata")},
                "dataset_revision": dataset_revisions(tcfg.get("dataset_path", "")), "requests_sha256": requests_sha256(samples),
                "n_new_requests": lm.n_new - n_new0, "n_cache_hits": lm.n_cached_hits - n_hit0, "cross_checks": lm.cross_checks[cc0:],
                "elapsed_seconds_this_session": time.time() - t0}
        write_json(marker, info)
        task_dirs[task] = info
        log.info("[%s] done: n=%d acc=%.4f%s (%.0fs)", task, agg["n"], agg["acc"], f" acc_norm={agg['acc_norm']:.4f}" if "acc_norm" in agg else f" perplexity={agg['perplexity']:.2f}", time.time() - t0)
    cache.close()

    # ---------------- assemble final artifacts
    per_sample = run_dir / "per_sample_results.jsonl"
    with open(per_sample, "w", encoding="utf-8", newline="\n") as out:
        for task in args.tasks:
            out.write(open(run_dir / "per_task" / f"{task}.jsonl", encoding="utf-8").read())
    ccs = [c for t in args.tasks for c in task_dirs[t]["cross_checks"]]
    results = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "partial_evaluation": args.limit is not None, "limit": args.limit,
        "run_id": run_dir.name, "finished_utc": utc_now(), "config_hash": chash, "code_sha256": code_sha, "checkpoint_sha256": ck_sha, "precision": args.precision,
        "compute_dtype": dtype_probe,
        "tasks": {t: {"aggregate": task_dirs[t]["aggregate"], "lm_eval_reported": task_dirs[t]["lm_eval_reported"], "consistency": task_dirs[t]["consistency"],
                      "n_expected": task_dirs[t]["n_expected"], "n_original_docs": task_dirs[t]["n_original_docs"], "task_version": task_dirs[t]["task_version"],
                      "task_config": task_dirs[t]["task_config"], "dataset_revision": task_dirs[t]["dataset_revision"], "requests_sha256": task_dirs[t]["requests_sha256"]}
                  for t in args.tasks},
        "headline": {t: ({"acc": task_dirs[t]["aggregate"]["acc"], "acc_norm": task_dirs[t]["aggregate"]["acc_norm"]} if "acc_norm" in task_dirs[t]["aggregate"]
                         else {"acc": task_dirs[t]["aggregate"]["acc"], "perplexity": task_dirs[t]["aggregate"]["perplexity"]}) for t in args.tasks},
        "independent_cross_check": {"n_requests_rescored": len(ccs), "max_abs_loglik_diff": max((c["abs_diff"] for c in ccs), default=None),
                                    "greedy_disagreements": sum(1 for c in ccs if not c["greedy_agrees"]),
                                    "method": "every Nth newly scored request is re-scored by a separate step-by-step state-carrying implementation (no autocast)",
                                    "note": "meaningful as an exact check only for --precision fp32"},
        "consistency_all_ok": all(task_dirs[t]["consistency"]["ok"] for t in args.tasks),
        "metric_definitions": METRIC_DEFINITIONS, "contamination": CONTAMINATION_NOTE,
        "determinism": "log-likelihoods are deterministic for fixed code/weights/device/torch build; across hardware or torch builds expect ~1e-4 nat differences "
                       "and rare argmax flips on near-ties. Compare runs with `python -m benchmarks compare`.",
        "uncertainty": "intervals describe sampling error over the benchmark examples only; they do not include checkpoint-to-checkpoint or seed variability",
        "elapsed_seconds_this_session": time.time() - t_run, "git": git_info(),
    }
    write_json(run_dir / "results.json", results)
    status.update(state="complete", finished_utc=utc_now())
    write_json(status_path, status)
    fh.flush()
    logging.shutdown()
    man = build_manifest(run_dir, {"run_id": run_dir.name, "config_hash": chash, "code_sha256": code_sha, "checkpoint_sha256": ck_sha, "created_utc": status["created_utc"],
                                   "finished_utc": status["finished_utc"], "git": git_info(), "partial_evaluation": args.limit is not None,
                                   "datasets": {t: {"dataset_path": task_dirs[t]["task_config"]["dataset_path"], "dataset_name": task_dirs[t]["task_config"]["dataset_name"],
                                                    "test_split": task_dirs[t]["task_config"]["test_split"], "validation_split": task_dirs[t]["task_config"]["validation_split"],
                                                    "revision": task_dirs[t]["dataset_revision"], "requests_sha256": task_dirs[t]["requests_sha256"]} for t in args.tasks}})
    write_json(run_dir / "manifest.json", man)
    print(json.dumps(results["headline"], indent=2))
    print(f"run directory: {run_dir}")
    if not results["consistency_all_ok"]:
        print("WARNING: recomputed metrics disagree with lm-eval for at least one task; see results.json", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------ verify / compare / historical
def cmd_verify(args) -> int:
    problems = verify_manifest(Path(args.run_dir))
    if problems:
        print("FAILED:\n  " + "\n  ".join(problems))
        return 1
    res = read_json(Path(args.run_dir) / "results.json")
    print(f"OK: manifest hashes match for {args.run_dir}\n  checkpoint_sha256 {res['checkpoint_sha256']}\n  config_hash {res['config_hash']}")
    if args.checkpoint:
        got = sha256_file(args.checkpoint)
        print(f"  --checkpoint {args.checkpoint}: sha256 {'MATCHES' if got == res['checkpoint_sha256'] else 'DIFFERS: ' + got}")
        return 0 if got == res["checkpoint_sha256"] else 1
    return 0


def _load_records(run_dir: Path):
    recs = {}
    for line in open(run_dir / "per_sample_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        recs[(r["task"], r["doc_id"])] = r
    return recs


def cmd_compare(args) -> int:
    a, b = Path(args.run_a), Path(args.run_b)
    ca, cb = read_json(a / "config.json"), read_json(b / "config.json")
    d = diff_configs(ca, cb)
    ra, rb = read_json(a / "results.json"), read_json(b / "results.json")
    print(f"config_hash: {ra['config_hash'][:12]} vs {rb['config_hash'][:12]} -> {'IDENTICAL' if not d else 'DIFFERENT'}")
    for x in d:
        print("   config difference:", x)
    print(f"checkpoint sha256: {'IDENTICAL' if ra['checkpoint_sha256'] == rb['checkpoint_sha256'] else 'DIFFERENT'}")
    print(f"benchmark code sha256: {'IDENTICAL' if ra['code_sha256'] == rb['code_sha256'] else 'DIFFERENT'}")
    ea, eb = read_json(a / "environment.json"), read_json(b / "environment.json")
    for k in ("torch", "cuda_runtime", "cuda_device_name", "platform"):
        if ea.get(k) != eb.get(k):
            print(f"environment {k}: {ea.get(k)!r} vs {eb.get(k)!r}")
    if (d or ra["checkpoint_sha256"] != rb["checkpoint_sha256"]) and not args.allow_config_diff:
        print("refusing to compare per-example results across different configurations/checkpoints (use --allow-config-diff to override)")
        return 2
    A, B = _load_records(a), _load_records(b)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    print(f"examples: {len(set(A) & set(B))} in both, {len(only_a)} only in A, {len(only_b)} only in B")
    worst = 0
    for t in dict.fromkeys(k[0] for k in A):
        ids = sorted(i for (tt, i) in A if tt == t and (t, i) in B)
        flips = max_d = pf = 0
        for i in ids:
            x, y = A[(t, i)], B[(t, i)]
            if x["kind"] == "multiple_choice":
                flips += (x["acc"] != y["acc"]) + (x["acc_norm"] != y["acc_norm"])
                pf += x["pred_acc"] != y["pred_acc"]
                max_d = max(max_d, max(abs(p - q) for p, q in zip(x["choice_loglik"], y["choice_loglik"])))
            else:
                flips += x["acc"] != y["acc"]
                max_d = max(max_d, abs(x["loglik"] - y["loglik"]))
        ma, mb = ra["tasks"][t]["aggregate"], rb["tasks"][t]["aggregate"]
        extra = f"perplexity {ma['perplexity']:.3f} vs {mb['perplexity']:.3f}" if "perplexity" in ma else f"acc_norm {ma['acc_norm']:.4f} vs {mb['acc_norm']:.4f}"
        print(f"  {t:15s} n={len(ids):6d} acc {ma['acc']:.4f} vs {mb['acc']:.4f} | {extra} | correctness flips {flips} | max |d loglik| {max_d:.3e}")
        worst = max(worst, flips)
    return 0 if not (only_a or only_b) else 1


def cmd_historical(args) -> int:
    """Side-by-side of a new run and the legacy benchmark JSON. Read-only; reported scores are read from the file, never hardcoded."""
    leg = read_json(args.legacy)
    if args.key not in leg:
        print(f"error: key {args.key!r} not in {args.legacy}; available: {[k for k in leg if k.startswith('lm_eval')]}", file=sys.stderr)
        return 2
    node = leg[args.key]["results"]
    new = read_json(Path(args.run_dir) / "results.json")
    print(f"legacy file: {args.legacy} [{args.key}] config={node.get('config', {}).get('model')} git_hash={node.get('git_hash')} date={node.get('date')}")
    print(f"new run    : {args.run_dir} (checkpoint {new['checkpoint_sha256'][:12]}, partial_evaluation={new['partial_evaluation']})")
    print("NOTE: the legacy file records no checkpoint hash; agreement supports, but does not prove, that the same weights were evaluated.")
    for t, v in new["tasks"].items():
        old = node["results"].get(t)
        if not old:
            print(f"  {t}: not in legacy file")
            continue
        n_old = node.get("n-samples", {}).get(t, {}).get("effective")
        a = v["aggregate"]
        parts = [f"n {a['n']} (legacy {n_old})"]
        for name, key in (("acc", "acc,none"), ("acc_norm", "acc_norm,none"), ("perplexity", "perplexity,none")):
            if name in a and key in old:
                parts.append(f"{name} new {a[name]:.6f} legacy {old[key]:.6f} diff {a[name] - old[key]:+.2e}")
        print(f"  {t:15s} " + " | ".join(parts))
    return 0


def cmd_list(_args) -> int:
    from .metrics import SUPPORTED_TASKS
    print("\n".join(SUPPORTED_TASKS))
    return 0


def main(argv=None) -> int:
    from .metrics import SUPPORTED_TASKS
    p = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run benchmarks (resumable)")
    r.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS), metavar="TASK", help=f"subset of {list(SUPPORTED_TASKS)} (default: all)")
    r.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    r.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    r.add_argument("--device", choices=["cpu", "cuda"], default=None)
    r.add_argument("--max-length", type=int, default=512)
    r.add_argument("--limit", type=int, default=None, help="evaluate only the first N documents per task (smoke test; results are flagged partial)")
    r.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    r.add_argument("--name", default=None, help="optional suffix for the run directory name")
    r.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run (configuration, checkpoint and code must be unchanged)")
    r.add_argument("--cross-check-every", type=int, default=200, help="re-score every Nth new request with an independent implementation (0 = off)")
    r.add_argument("--allow-network", action="store_true", help="allow dataset downloads (default: offline, cached data only)")
    for k, v in (("vocab-size", 100277), ("d-model", 768), ("n-heads", 12), ("n-layers", 12)):
        r.add_argument(f"--{k}", type=int, default=v)
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verify", help="verify manifest hashes of a finished run")
    v.add_argument("run_dir")
    v.add_argument("--checkpoint", default=None, help="also compare this checkpoint file's sha256 with the run's")
    v.set_defaults(fn=cmd_verify)
    c = sub.add_parser("compare", help="compare two runs example by example")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--allow-config-diff", action="store_true")
    c.set_defaults(fn=cmd_compare)
    h = sub.add_parser("historical", help="print a new run next to the legacy benchmark JSON (read-only)")
    h.add_argument("run_dir")
    h.add_argument("--legacy", default=str(REPO_ROOT / "evaluation" / "jbrain_v2" / "benchmark_final_488281_complete.json"))
    h.add_argument("--key", default="lm_eval_fp32")
    h.set_defaults(fn=cmd_historical)
    sub.add_parser("list-tasks").set_defaults(fn=cmd_list)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (CompatibilityError, ValidationError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
