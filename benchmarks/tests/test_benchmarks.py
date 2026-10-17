"""Lightweight, CPU-only tests for the benchmark suite. They use a tiny randomly initialised model and synthetic fixtures; no benchmark dataset,
real checkpoint or GPU is needed.   Run from the repository root:   python -m unittest benchmarks.tests.test_benchmarks -v"""
import collections
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if os.environ.get("JBR_TEST_CUDA") != "1":  # CPU-only by default; set JBR_TEST_CUDA=1 to also run the CUDA bf16 test
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmarks import common
from benchmarks.adapter import JBrainEvalLM, RequestCache, load_model, load_tokenizer, request_key, tokenizer_info
from benchmarks.metrics import aggregate, build_records, choice_chars, consistency

SPEC = {"vocab_size": 100277, "d_model": 32, "n_heads": 2, "n_layers": 1}
Req = collections.namedtuple("Req", "args")


def tiny_checkpoint(dirpath: Path) -> Path:
    from architecture_v2 import JBRLanguageModel
    torch.manual_seed(0)
    m = JBRLanguageModel(SPEC["vocab_size"], d_model=32, n_heads=2, n_layers=1)
    p = dirpath / "tiny.pt"
    torch.save(m.state_dict(), p)
    return p


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.ckpt = tiny_checkpoint(cls.tmp)
        cls.model, cls.info = load_model(cls.ckpt, "cpu", SPEC)
        cls.enc = load_tokenizer()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def lm(self, max_length=512, cache_name="c.jsonl", **kw):
        lm = JBrainEvalLM(self.model, self.enc, RequestCache(self.tmp / cache_name), max_length=max_length, **kw)
        self.addCleanup(lm.cache.close)
        return lm


class LoadingTests(Base):
    def test_strict_load_records_hash_size_and_params(self):
        self.assertEqual(self.info["sha256"], common.sha256_file(self.ckpt))
        self.assertEqual(self.info["bytes"], self.ckpt.stat().st_size)
        self.assertEqual(self.info["n_parameters"], sum(p.numel() for p in self.model.parameters()))
        self.assertTrue(self.info["strict_load"] and self.info["weights_only_load"])
        self.assertFalse(self.model.training)

    def test_loading_never_modifies_the_checkpoint(self):
        before, mt = common.sha256_file(self.ckpt), self.ckpt.stat().st_mtime_ns
        load_model(self.ckpt, "cpu", SPEC)
        self.assertEqual(before, common.sha256_file(self.ckpt))
        self.assertEqual(mt, self.ckpt.stat().st_mtime_ns)

    def test_missing_key_unexpected_key_and_shape_mismatch_fail_loudly(self):
        sd = torch.load(self.ckpt, weights_only=True)
        k = next(iter(sd))
        bad = {a: b for a, b in sd.items() if a != k}
        p1 = self.tmp / "missing.pt"; torch.save(bad, p1)
        with self.assertRaises(RuntimeError):
            load_model(p1, "cpu", SPEC)
        extra = dict(sd); extra["bogus.weight"] = torch.zeros(1)
        p2 = self.tmp / "extra.pt"; torch.save(extra, p2)
        with self.assertRaises(RuntimeError):
            load_model(p2, "cpu", SPEC)
        with self.assertRaises(RuntimeError):
            load_model(self.ckpt, "cpu", {**SPEC, "d_model": 64})

    def test_training_checkpoint_dict_and_missing_file_rejected(self):
        p = self.tmp / "train.pt"; torch.save({"step": 1, "model_state_dict": torch.load(self.ckpt, weights_only=True)}, p)
        with self.assertRaises(ValueError):
            load_model(p, "cpu", SPEC)
        with self.assertRaises(FileNotFoundError):
            load_model(self.tmp / "nope.pt", "cpu", SPEC)


class TokenizerTests(Base):
    def test_cl100k_identity(self):
        self.assertEqual(self.enc.n_vocab, 100277)
        self.assertEqual(self.enc.eot_token, 100257)
        info = tokenizer_info(self.enc)
        self.assertEqual(info["name"], "cl100k_base")
        self.assertEqual(info["probe_encodings_sha256"], tokenizer_info(load_tokenizer())["probe_encodings_sha256"])

    def test_special_token_text_is_ordinary_text_and_no_bos_added(self):
        lm = self.lm(cache_name="tok.jsonl")
        ids = lm.tok_encode("a <|endoftext|> b")  # must not raise, must not produce the EOT id
        self.assertNotIn(self.enc.eot_token, ids)
        self.assertEqual(lm.tok_encode("The capital"), self.enc.encode("The capital"))
        self.assertEqual(lm.tok_encode("x y z", left_truncate_len=2), self.enc.encode("x y z")[-2:])
        self.assertEqual(lm.tok_decode(lm.tok_encode(" Paris")), " Paris")

    def test_continuation_keeps_its_leading_space_token(self):
        self.assertEqual(self.enc.encode("is Paris"), self.enc.encode("is") + self.enc.encode(" Paris"))
        self.assertNotEqual(self.enc.encode(" Paris"), self.enc.encode("Paris"))


class PrecisionTests(Base):
    def test_fp32_probe_reports_no_bf16(self):
        p = self.lm(cache_name="p1.jsonl", precision="fp32").probe_compute_dtype(16)
        self.assertEqual(p["precision"], "fp32")
        self.assertFalse(p["bf16_actually_executed"])
        self.assertEqual(p["parameter_dtypes"], ["torch.float32"])

    def test_bf16_on_cpu_is_not_reported_as_bf16(self):
        # autocast is only enabled on cuda; on CPU the probe must show that bf16 did NOT execute
        p = self.lm(cache_name="p2.jsonl", precision="bf16").probe_compute_dtype(16)
        self.assertFalse(p["bf16_actually_executed"])

    def test_invalid_precision_rejected(self):
        c = RequestCache(self.tmp / "p3.jsonl")
        self.addCleanup(c.close)
        with self.assertRaises(ValueError):
            JBrainEvalLM(self.model, self.enc, c, precision="fp16")

    @unittest.skipUnless(torch.cuda.is_available(), "needs cuda")
    def test_bf16_probe_on_cuda_executes_bf16_and_differs_slightly(self):
        m, _ = load_model(self.ckpt, "cuda", SPEC)
        lm = JBrainEvalLM(m, self.enc, RequestCache(self.tmp / "p4.jsonl"), precision="bf16", device="cuda")
        self.addCleanup(lm.cache.close)
        self.assertTrue(lm.probe_compute_dtype(16)["bf16_actually_executed"])
        c, k = lm.tok_encode("The quick brown fox jumps over the"), lm.tok_encode(" lazy dog")
        lm.precision = "fp32"
        a = lm.score(c, k)["loglik"]
        lm.precision = "bf16"
        b = lm.score(c, k)["loglik"]
        self.assertNotEqual(a, b)
        self.assertLess(abs(a - b), 0.5)

    def test_precision_changes_config_hash_and_blocks_resume(self):
        base = {"tasks": ["arc_easy"], "limit": None, "precision": "fp32", "max_length": 512}
        self.assertNotEqual(common.config_hash(base), common.config_hash({**base, "precision": "bf16"}))


class ScoringTests(Base):
    def test_score_matches_manual_log_softmax_and_independent_scorer(self):
        lm = self.lm(cache_name="s.jsonl")
        c, k = lm.tok_encode("The quick brown fox jumps over the"), lm.tok_encode(" lazy dog")
        r = lm.score(c, k)
        with torch.no_grad():
            lg, _ = self.model(torch.tensor([c + k]))
        lp = F.log_softmax(lg[0].float(), -1)
        manual = sum(lp[len(c) - 1 + j, t].item() for j, t in enumerate(k))
        self.assertAlmostEqual(r["loglik"], manual, places=4)
        ind, greedy = lm.score_independent(c, k)
        self.assertAlmostEqual(r["loglik"], ind, places=3)
        self.assertEqual(r["greedy"], greedy)
        self.assertLess(r["loglik"], 0)

    def test_is_greedy_true_only_when_every_token_is_argmax(self):
        lm = self.lm(cache_name="g.jsonl")
        c = lm.tok_encode("Hello there my friend")
        with torch.no_grad():
            lg, _ = self.model(torch.tensor([c]))
        top = int(lg[0, -1].argmax())
        self.assertTrue(lm.score(c, [top])["greedy"])
        worst = int(lg[0, -1].argmin())
        self.assertFalse(lm.score(c, [worst])["greedy"])

    def test_loglikelihood_uses_cache_on_second_call_and_returns_same_values(self):
        lm = self.lm(cache_name="l.jsonl")
        reqs = [Req(("The sky is", " blue")), Req(("The sky is", " green")), Req(("Two plus two is", " four"))]
        a = lm.loglikelihood(reqs)
        self.assertEqual((lm.n_new, lm.n_cached_hits), (3, 0))
        b = lm.loglikelihood(reqs)
        self.assertEqual((lm.n_new, lm.n_cached_hits), (3, 3))
        self.assertEqual(a, b)
        self.assertEqual(len(lm.cache.entries), 3)

    def test_cross_check_hook_records_small_differences(self):
        lm = self.lm(cache_name="x.jsonl", cross_check_every=1)
        lm.loglikelihood([Req(("Once upon a", " time"))])
        self.assertEqual(len(lm.cross_checks), 1)
        self.assertLess(lm.cross_checks[0]["abs_diff"], 1e-3)
        self.assertTrue(lm.cross_checks[0]["greedy_agrees"])


class TruncationTests(Base):
    def test_left_truncation_keeps_tail_and_full_continuation(self):
        lm = self.lm(max_length=16, cache_name="t.jsonl")
        ctx, cont = list(range(1000, 1040)), [5, 6, 7]
        used, dropped = lm.prepare(ctx, cont)
        self.assertEqual(used, ctx[-13:])
        self.assertEqual(dropped, 27)
        self.assertEqual(len(used) + len(cont), 16)

    def test_no_truncation_when_it_fits_and_flag_in_score(self):
        lm = self.lm(max_length=16, cache_name="t2.jsonl")
        self.assertEqual(lm.prepare([1, 2, 3], [4])[1], 0)
        r = lm.score(list(range(1000, 1040)), [5, 6, 7])
        self.assertTrue(r["truncated"])
        self.assertEqual((r["n_context_tokens"], r["n_context_used"], r["n_context_dropped"]), (40, 13, 27))

    def test_truncated_score_equals_scoring_only_the_kept_tail(self):
        lm = self.lm(max_length=16, cache_name="t3.jsonl")
        ctx, cont = list(range(1000, 1040)), [5, 6, 7]
        a = lm.score(ctx, cont)["loglik"]
        b = lm.score(ctx[-13:], cont)["loglik"]
        self.assertAlmostEqual(a, b, places=6)

    def test_invalid_inputs_raise_instead_of_misaligning(self):
        lm = self.lm(max_length=16, cache_name="t4.jsonl")
        with self.assertRaises(ValueError):
            lm.prepare([], [1])
        with self.assertRaises(ValueError):
            lm.prepare([1, 2], [])
        with self.assertRaises(ValueError):
            lm.prepare([1, 2], list(range(16)))
        with self.assertRaises(NotImplementedError):
            lm.generate_until([])
        with self.assertRaises(NotImplementedError):
            lm.loglikelihood_rolling([])


def mc_sample(doc_id, pairs, gold):
    return {"doc_id": doc_id, "doc_hash": "d", "prompt_hash": "p", "arguments": [list(p) for p in pairs], "target": gold, "acc": None, "acc_norm": None}


def entry(ll, n_ctx=5, n_cont=2, trunc=False):
    return {"loglik": ll, "greedy": False, "n_context_tokens": n_ctx, "n_context_used": n_ctx, "n_context_dropped": 0, "n_continuation_tokens": n_cont, "truncated": trunc}


class LabelMappingAndMetricsTests(unittest.TestCase):
    def test_choice_chars_strips_only_the_single_leading_space(self):
        self.assertEqual(choice_chars(" abc"), 3)
        self.assertEqual(choice_chars("abc"), 3)
        self.assertEqual(choice_chars("  abc"), 4)

    def test_gold_index_and_acc_vs_acc_norm_are_distinct(self):
        ctx = "Q?"
        conts = [" no", " a much longer correct answer", " nope"]       # gold = index 1
        lls = [-3.0, -9.0, -5.0]                                          # raw: index 0 wins; per-char: 1 wins (-9/28 > -3/2)
        cache = {request_key(ctx, c): entry(l) for c, l in zip(conts, lls)}
        recs = build_records("arc_easy", "multiple_choice", [mc_sample(0, [(ctx, c) for c in conts], 1)], cache)
        r = recs[0]
        self.assertEqual((r["pred_acc"], r["pred_acc_norm"]), (0, 1))
        self.assertEqual((r["acc"], r["acc_norm"]), (0, 1))
        self.assertEqual(r["gold_choice"], "a much longer correct answer")
        self.assertEqual(r["choice_loglik"], lls)

    def test_ties_choose_first_like_numpy_argmax(self):
        conts = [" x", " y"]
        cache = {request_key("c", k): entry(-1.0) for k in conts}
        r = build_records("piqa", "multiple_choice", [mc_sample(0, [("c", k) for k in conts], 1)], cache)[0]
        self.assertEqual(r["pred_acc"], 0)

    def test_multiple_choice_aggregation_known_values(self):
        recs = []
        for i in range(10):
            recs.append({"task": "t", "kind": "multiple_choice", "doc_id": i, "acc": int(i < 3), "acc_norm": int(i < 6), "n_choices": 4, "truncated": i == 0})
        a = aggregate("t", recs)
        self.assertEqual((a["n"], a["acc"], a["acc_norm"], a["chance"], a["n_truncated_docs"]), (10, 0.3, 0.6, 0.25, 1))
        lo, hi = a["acc_ci95"]
        self.assertTrue(lo < 0.3 < hi)
        self.assertAlmostEqual(common.wilson_interval(0, 50)[0], 0.0, places=9)
        self.assertLess(common.wilson_interval(50, 50)[0], 1.0)

    def test_lambada_aggregation_word_vs_token_perplexity(self):
        recs = [{"task": "lambada_openai", "kind": "word_prediction", "doc_id": 0, "acc": 1, "word_nll": 1.0, "n_target_tokens": 1, "truncated": False},
                {"task": "lambada_openai", "kind": "word_prediction", "doc_id": 1, "acc": 0, "word_nll": 5.0, "n_target_tokens": 2, "truncated": False}]
        a = aggregate("lambada_openai", recs)
        self.assertAlmostEqual(a["perplexity"], math.exp(3.0), places=9)          # per word: exp(mean(1,5))
        self.assertAlmostEqual(a["token_perplexity"], math.exp(6.0 / 3), places=9)  # per token: exp(total / tokens)
        self.assertEqual(a["acc"], 0.5)
        self.assertEqual(a["acc_by_target_tokens"], {"1": [1, 1], "2": [0, 1]})

    def test_lambada_record_from_cache(self):
        s = {"doc_id": 3, "doc_hash": "d", "prompt_hash": "p", "arguments": [["the cat sat on the", " mat"]], "target": " mat", "acc": 1.0, "perplexity": 2.0}
        cache = {request_key("the cat sat on the", " mat"): dict(entry(-0.7, n_cont=1), greedy=True)}
        r = build_records("lambada_openai", "loglikelihood", [s], cache)[0]
        self.assertEqual((r["acc"], r["word_nll"], r["n_target_tokens"], r["target"]), (1, 0.7, 1, " mat"))

    def test_consistency_detects_disagreement_with_the_harness(self):
        recs = [{"kind": "multiple_choice", "acc": 1, "acc_norm": 0, "lm_eval_acc": 1.0, "lm_eval_acc_norm": 0.0}]
        ok = consistency("t", {"acc": 1.0, "acc_norm": 0.0}, {"acc,none": 1.0, "acc_norm,none": 0.0}, recs)
        self.assertTrue(ok["ok"])
        bad = consistency("t", {"acc": 1.0, "acc_norm": 0.0}, {"acc,none": 0.9, "acc_norm,none": 0.0}, [dict(recs[0], lm_eval_acc=0.0)])
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["flag_mismatches"], 1)
        self.assertEqual(len(bad["aggregate_mismatches"]), 1)


class ConfigHashAndManifestTests(unittest.TestCase):
    def test_hash_is_order_independent_and_sensitive(self):
        a = {"tasks": ["a", "b"], "limit": None, "x": {"p": 1, "q": 2.5}}
        b = {"x": {"q": 2.5, "p": 1}, "limit": None, "tasks": ["a", "b"]}
        self.assertEqual(common.config_hash(a), common.config_hash(b))
        self.assertNotEqual(common.config_hash(a), common.config_hash({**a, "limit": 10}))
        self.assertNotEqual(common.config_hash(a), common.config_hash({**a, "tasks": ["b", "a"]}))  # list order is meaningful
        self.assertEqual(len(common.config_hash(a)), 64)
        with self.assertRaises(ValueError):
            common.config_hash({"x": float("nan")})

    def test_code_hash_is_stable(self):
        self.assertEqual(common.code_sha256(), common.code_sha256())

    def test_manifest_roundtrip_and_tamper_detection(self):
        d = Path(tempfile.mkdtemp())
        try:
            common.write_json(d / "config.json", {"a": 1})
            common.write_json(d / "results.json", {"r": 2})
            (d / "per_sample_results.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            man = common.build_manifest(d, {"config_hash": common.config_hash({"a": 1})})
            common.write_json(d / "manifest.json", man)
            self.assertEqual(common.verify_manifest(d), [])
            self.assertEqual(set(man["files"]), {"config.json", "results.json", "per_sample_results.jsonl"})
            (d / "results.json").write_text("{}", encoding="utf-8")
            self.assertTrue(any("results.json" in p for p in common.verify_manifest(d)))
            (d / "per_sample_results.jsonl").unlink()
            self.assertTrue(any("missing" in p for p in common.verify_manifest(d)))
            common.write_json(d / "config.json", {"a": 2})
            self.assertTrue(any("config" in p for p in common.verify_manifest(d)))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_verify_without_manifest_reports_incomplete(self):
        d = Path(tempfile.mkdtemp())
        try:
            self.assertTrue(common.verify_manifest(d))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.cfg = {"tasks": ["arc_easy"], "limit": None, "precision": "fp32", "max_length": 512}
        common.write_json(self.d / "config.json", self.cfg)
        common.write_json(self.d / "checkpoint.json", {"sha256": "aa"})
        common.write_json(self.d / "status.json", {"code_sha256": "cc", "state": "running"})

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_identical_configuration_may_resume(self):
        common.check_resume_compat(self.d, dict(self.cfg), "cc", "aa")

    def test_any_difference_is_refused_with_a_useful_message(self):
        with self.assertRaisesRegex(common.CompatibilityError, "precision"):
            common.check_resume_compat(self.d, {**self.cfg, "precision": "bf16"}, "cc", "aa")
        with self.assertRaisesRegex(common.CompatibilityError, "limit"):
            common.check_resume_compat(self.d, {**self.cfg, "limit": 5}, "cc", "aa")
        with self.assertRaisesRegex(common.CompatibilityError, "checkpoint"):
            common.check_resume_compat(self.d, dict(self.cfg), "cc", "bb")
        with self.assertRaisesRegex(common.CompatibilityError, "code changed"):
            common.check_resume_compat(self.d, dict(self.cfg), "dd", "aa")

    def test_completed_or_empty_directories_are_refused(self):
        common.write_json(self.d / "manifest.json", {})
        with self.assertRaisesRegex(common.CompatibilityError, "already complete"):
            common.check_resume_compat(self.d, dict(self.cfg), "cc", "aa")
        with self.assertRaisesRegex(common.CompatibilityError, "no config.json"):
            common.check_resume_compat(self.d / "missing", dict(self.cfg), "cc", "aa")

    def test_request_cache_survives_a_torn_last_line_but_not_corruption_in_the_middle(self):
        p = self.d / "cache.jsonl"
        c = RequestCache(p); c.add({"key": "k1", "loglik": -1.0}); c.add({"key": "k2", "loglik": -2.0}); c.close()
        with open(p, "ab") as f:
            f.write(b'{"key": "k3", "logl')          # interrupted write
        c2 = RequestCache(p)
        self.assertEqual(set(c2.entries), {"k1", "k2"})
        c2.add({"key": "k3", "loglik": -3.0}); c2.close()
        c3 = RequestCache(p)
        self.assertEqual(set(c3.entries), {"k1", "k2", "k3"})
        c3.close()
        lines = p.read_text(encoding="utf-8").splitlines()
        lines[1] = "{broken"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            RequestCache(p)


def good_mc(i):
    return {"task": "t", "kind": "multiple_choice", "doc_id": i, "choice_loglik": [-1.0, -2.0], "gold_index": 0}


class ValidationTests(unittest.TestCase):
    def test_clean_set_passes(self):
        self.assertEqual(common.validate_per_sample([good_mc(i) for i in range(5)], "t", 5)["n"], 5)

    def test_missing_duplicate_and_unexpected_examples_detected(self):
        recs = [good_mc(i) for i in range(5)]
        with self.assertRaisesRegex(common.ValidationError, "missing"):
            common.validate_per_sample(recs[:4], "t", 5)
        with self.assertRaisesRegex(common.ValidationError, "duplicate"):
            common.validate_per_sample(recs + [good_mc(2)], "t", 5)
        with self.assertRaisesRegex(common.ValidationError, "unexpected"):
            common.validate_per_sample(recs + [good_mc(9)], "t", 5)

    def test_invalid_values_detected(self):
        for bad in (dict(good_mc(0), choice_loglik=[float("nan"), -1.0]), dict(good_mc(0), choice_loglik=[-1.0, 0.5]),
                    dict(good_mc(0), gold_index=2), dict(good_mc(0), gold_index=-1), dict(good_mc(0), choice_loglik=None)):
            with self.assertRaisesRegex(common.ValidationError, "invalid"):
                common.validate_per_sample([bad], "t", 1)
        with self.assertRaisesRegex(common.ValidationError, "invalid"):
            common.validate_per_sample([{"task": "l", "kind": "word_prediction", "doc_id": 0, "loglik": -1.0, "target": ""}], "l", 1)
        with self.assertRaisesRegex(common.ValidationError, "non-integer"):
            common.validate_per_sample([dict(good_mc(0), doc_id="0")], "t", 1)


if __name__ == "__main__":
    unittest.main()
