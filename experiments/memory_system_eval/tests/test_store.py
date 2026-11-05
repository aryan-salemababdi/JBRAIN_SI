"""Store semantics (criteria 3 and 9): the on-disk store and the in-memory store agree on every operation; payloads round-trip
exactly; insert / update / conflict / delete / revive are correct and persist across close and reopen; the IBF field equals the
sum of the live records' terms after deletions; deleted records leave the FTS index. Then every write/update/delete event of 100
MemBench test episodes (gold extraction) is replayed into a fresh disk store and the final state must equal the gold state."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import tokenizer  # noqa: E402
from diskstore import DiskStore  # noqa: E402
from extract import extract_episode  # noqa: E402
from membench import load_split  # noqa: E402
from store import Store, oracle_store  # noqa: E402
from test_extract import gender_oracle, gold_lines  # noqa: E402

ENC = tokenizer()


def ops_check(st):
    out = []
    st.add_entity("self", "me", kind="self")
    st.add_entity("e1", "Sara Kim", "Sara", "Kim", "f")
    out.append(st.write("e1", "phone", [1, 2, 3], 0, cued=False))        # insert
    out.append(st.write("e1", "phone", [1, 2, 3], 1, cued=False))        # duplicate
    out.append(st.write("e1", "phone", [4, 5], 2, cued=True))            # update
    out.append(st.write("e1", "phone", [6], 3, cued=False))              # conflict
    out.append(st.delete("e1", "phone", 4))                              # delete
    out.append(st.get("e1", "phone"))
    out.append(st.write("e1", "phone", [7, 8], 5, cued=False))           # revive
    out.append(st.get("e1", "phone"))
    out.append(st.delete("e1", "email", 6))                              # missing
    return out


def test_disk_and_memory_semantics_agree_and_persist():
    d = Path(tempfile.mkdtemp())
    try:
        mem, disk = Store(), DiskStore(d / "s")
        a = ops_check(mem)
        b = ops_check(disk)
        a[5] = None if a[5] is None else a[5].payload
        a[7] = a[7].payload
        assert a == b, (a, b)
        big = list(range(100000, 100050))
        disk.write("self", "wifi", big, 7, cued=False, doc_text="my wifi password maple orbit")
        disk.close()
        re = DiskStore(d / "s")
        assert re.get("self", "wifi") == big and re.get("e1", "phone") == [7, 8]
        assert re.bm25(["maple"], 5) and re.entities["e1"]["name"] == "Sara Kim"
        re.delete("self", "wifi", 8)
        assert re.bm25(["maple"], 5) == [] and re.get("self", "wifi") is None
        # IBF field == sum of live terms
        live = re.db.execute("SELECT id, eid, attr, bucket FROM records WHERE status=1").fetchall()
        F = np.zeros_like(np.asarray(re.field))
        for rid, e, at, bk in live:
            F[bk] += re._ibf_term(rid, e, at)
        assert np.allclose(F, np.asarray(re.field), atol=1e-4)
        re.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gold_event_replay_into_disk_store_matches_gold_state():
    d = Path(tempfile.mkdtemp())
    try:
        bad = 0
        for i, ep in enumerate(load_split("test_id", 50) + load_split("test_ood", 50)):
            mem = extract_episode(gold_lines(ep), ENC, gender_oracle(ep))
            ds = DiskStore(d / f"e{i}")
            for eid, v in mem.entities.items():
                ds.add_entity(eid, v["name"], v["first"], v["last"], v["gender"], v["kind"])
            for log in mem.log:                                    # replay the write log in order
                k = log["key"]
                r = mem.records.get(k)
                if log["op"] in ("insert", "revive", "update", "conflict", "duplicate"):
                    v = next(x for x in r.versions if x[1] == log["line"])
                    ds.write(k[0], k[1], v[0], log["line"], cued=v[2])
                elif log["op"] == "delete":
                    ds.delete(k[0], k[1], log["line"])
            gold = oracle_store(ep, ENC)
            g = {(gold.entities[e]["name"], a): (r.payload, r.status) for (e, a), r in gold.records.items()}
            got = {}
            for rid, e, a, st, pl in ds.db.execute("SELECT id, eid, attr, status, payload FROM records"):
                got[(ds.entities[e]["name"], a)] = (np.frombuffer(pl, dtype=np.uint32).tolist(), "live" if st else "deleted")
            bad += g != got
            ds.close()
        assert bad == 0, bad
    finally:
        shutil.rmtree(d, ignore_errors=True)
