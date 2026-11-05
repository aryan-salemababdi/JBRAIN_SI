"""Re-check the MemBench-H split hygiene from the built files (independently of the generator's own assertions): no test-split
name or value in train/val, no held-out novel word or held-out template 5-gram in train/val text, and the stored file hashes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import gen_membench as G  # noqa: E402
from common import DATA, sha256_file  # noqa: E402
from membench import load_split  # noqa: E402


def test_split_hygiene_and_hashes():
    man = json.loads((DATA / "membench" / "manifest.json").read_text(encoding="utf-8"))
    for s, f in man["files"].items():
        assert sha256_file(DATA / "membench" / f"{s}.jsonl") == f["sha256"], s
    eps = {s: load_split(s) for s in G.SPLITS}
    leak = G.leakage_checks(eps)
    assert leak["ood_novel_words_in_train_val"] == [] and leak["ood_template_5grams_in_train_val"] == []
    for k, v in leak.items():
        if k.endswith(("_in_train", "_in_val")):
            assert v == 0, (k, v)
