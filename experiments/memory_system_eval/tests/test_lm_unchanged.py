"""The memory is fully external: using it (writing notes, asking questions) must leave J-BRAIN's logits bit-identical and its
checkpoint file unchanged; delivered answers are the stored payload ids verbatim."""
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "prototype"))
import jbrain_frozen as J  # noqa: E402
from common import CKPT, CKPT_SHA256, sha256_file  # noqa: E402


def test_logits_unchanged_and_payload_verbatim():
    from jbrain_memory import JBrainMemory
    d = Path(tempfile.mkdtemp())
    try:
        mem = JBrainMemory(d / "s")
        prompt = mem.enc.encode("The notes below are my personal notes.\nQ: Where does Imogen Varga live?\nA:")
        x = torch.tensor([prompt], device=mem.dev)
        with torch.no_grad():
            h0, _, _ = J.run_layers(mem.model, x)
            before = J.head(mem.model, h0).clone()
        mem.observe("Imogen Varga lives in Ljubljana.\nYou can reach Bastian Oyelaran at 212-555-0147.")
        a = mem.ask("Where does Imogen Varga live?")
        with torch.no_grad():
            h1, _, _ = J.run_layers(mem.model, x)
            after = J.head(mem.model, h1)
        assert torch.equal(before, after)
        assert a["answer"] == "Ljubljana" and a["payload_ids"] == mem.enc.encode(" Ljubljana")
        out = mem.deliver(prompt, a)
        assert out[len(prompt):len(prompt) + len(a["payload_ids"])] == a["payload_ids"]
        mem.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert sha256_file(CKPT) == CKPT_SHA256
