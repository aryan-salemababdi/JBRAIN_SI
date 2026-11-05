"""IBF algebra: exact recall of a bound key, exact erasure, update invariance, Rayleigh null calibration, and the binding
failure of bundled (unbound) codes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ibf import Codes, Field  # noqa: E402

C = Codes(1024)


def filled(n_e=8, attrs=("phone", "email", "city", "job"), mode="bind"):
    f = Field(C, mode)
    for i in range(n_e):
        for a in attrs:
            f.write(f"p{i}", a)
    return f


def test_exact_recall_of_bound_key():
    f = filled()
    ids, r, _ = f.read(f.query_code({"p3": 1.0}, {"email": 1.0}))
    assert ids[0] == "p3|email" and r[0] > Field.threshold(len(f.ids)) and r[1] < Field.threshold(len(f.ids))


def test_exact_erasure_and_update_invariance():
    f = filled()
    F0 = f.F.copy()
    assert f.write("p3", "email") is False and np.array_equal(f.F, F0)        # update: same identity -> field unchanged
    f.write("p9", "city")
    f.erase("p9", "city")
    assert np.allclose(f.F, F0, atol=1e-5)
    f.erase("p3", "email")
    ids, r, _ = f.read(f.query_code({"p3": 1.0}, {"email": 1.0}))
    assert "p3|email" not in ids and r[0] < Field.threshold(len(f.ids))


def test_rayleigh_null_false_alarm_rate():
    """Absent keys: the per-pointer null tail must follow exp(-r^2); the Bonferroni rule must keep the max below r* at <= alpha."""
    f = filled(n_e=16)
    M = len(f.ids)
    hits, tail, n = 0, [], 0
    for i in range(300):
        ids, r, _ = f.read(f.query_code({f"q{i}": 1.0}, {"phone": 1.0}))
        hits += r[0] >= Field.threshold(M, 0.01)
        tail += list(r)
    tail = np.array(tail)
    emp = float((tail >= 1.5).mean())
    assert abs(emp - np.exp(-1.5 ** 2)) < 0.03, emp
    assert hits / 300 <= 0.03, hits


def test_bundling_confuses_bindings():
    """Bundled (unbound) codes: a record sharing only the entity or only the attribute with the query resonates at about half the
    signal, far above the null threshold, so thresholds cannot separate it. With binding it stays at noise level."""
    for mode, expect_confusion in (("bundle", True), ("bind", False)):
        f = filled(mode=mode)
        thr = Field.threshold(len(f.ids))
        ids, r, _ = f.read(f.query_code({"p0": 1.0}, {"email": 1.0}))
        rr = dict(zip(ids, r))
        assert ids[0] == "p0|email"
        partial = [rr["p0|phone"], rr["p1|email"]]
        if expect_confusion:
            assert min(partial) > thr, (mode, partial, thr)
        else:
            assert max(partial) < thr, (mode, partial, thr)
