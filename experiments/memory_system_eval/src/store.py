"""Record store semantics shared by every system (in-memory form used for the accuracy experiments).

Entities: id -> {name, first, last, gender, kind}. Records keyed by (entity id, attribute): versions (payload ids + source line),
status live / deleted, conflict flag. Updates append a version (latest wins); an uncued different value sets the conflict flag;
a delete tombstones the pair; re-assertion after a delete revives it. The on-disk SQLite form (src/diskstore.py) implements the same
semantics and is checked against this one in tests/test_store.py."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Record:
    eid: str
    attr: str
    versions: list = field(default_factory=list)          # [(payload_ids, line_index, cued)]
    status: str = "live"
    conflict: bool = False

    @property
    def payload(self):
        return self.versions[-1][0]

    @property
    def line(self):
        return self.versions[-1][1]


class Store:
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.records: dict[tuple, Record] = {}
        self.lines: list[str] = []                        # verbatim line ledger (text), indexed by line number
        self.log: list[dict] = []

    # entities
    def add_entity(self, eid, name, first=None, last=None, gender=None, kind="person"):
        if eid not in self.entities:
            self.entities[eid] = {"name": name, "first": first, "last": last, "gender": gender, "kind": kind}
        return eid

    def find_full(self, first, last):
        return [e for e, v in self.entities.items() if v["kind"] == "person" and v["first"] == first and v["last"] == last]

    def find_first(self, first):
        return [e for e, v in self.entities.items() if v["kind"] == "person" and v["first"] == first]

    def find_title(self, last, gender):
        return [e for e, v in self.entities.items() if v["kind"] == "person" and v["last"] == last and v["gender"] in (gender, None)]

    # records
    def write(self, eid, attr, payload, line, cued: bool):
        k = (eid, attr)
        r = self.records.get(k)
        if r is None:
            self.records[k] = Record(eid, attr, [(list(payload), line, cued)])
            self.log.append({"op": "insert", "key": k, "line": line})
            return "insert"
        if r.status == "deleted":
            r.versions.append((list(payload), line, cued))
            r.status = "live"
            self.log.append({"op": "revive", "key": k, "line": line})
            return "revive"
        if list(payload) == r.payload:
            self.log.append({"op": "duplicate", "key": k, "line": line})
            return "duplicate"
        if not cued:
            r.conflict = True
        r.versions.append((list(payload), line, cued))
        self.log.append({"op": "update" if cued else "conflict", "key": k, "line": line})
        return "update" if cued else "conflict"

    def delete(self, eid, attr, line):
        r = self.records.get((eid, attr))
        if r is None or r.status == "deleted":
            self.log.append({"op": "delete_missing", "key": (eid, attr), "line": line})
            return "missing"
        r.status = "deleted"
        self.log.append({"op": "delete", "key": (eid, attr), "line": line})
        return "deleted"

    def live(self):
        return {k: r for k, r in self.records.items() if r.status == "live"}

    def get(self, eid, attr):
        r = self.records.get((eid, attr))
        return r if (r is not None and r.status == "live") else None


def oracle_store(ep: dict, enc) -> Store:
    """Gold final state of an episode (oracle-write mode)."""
    st = Store()
    st.lines = [l["text"] for l in ep["lines"]]
    st.add_entity("self", "me", kind="self")
    for p in ep["persons"]:
        st.add_entity(p["eid"], p["full"], p["first"], p["last"], p["gender"])
    for r in ep["records"]:
        rec = Record(r["eid"], r["attr"], [(enc.encode(" " + v, disallowed_special=()), r["lines"][min(i, len(r["lines"]) - 1)], True)
                                           for i, v in enumerate(r["versions"])], r["status"], r["conflict"])
        st.records[(r["eid"], r["attr"])] = rec
    return st
