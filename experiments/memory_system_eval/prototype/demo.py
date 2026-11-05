"""Prototype demo / self-test / interactive shell.

  python prototype/demo.py --selftest            scripted notes -> questions, updates, deletions, absent questions; asserts exact answers
  python prototype/demo.py --interactive         type notes; lines starting with '?' are questions ('? turn || question' for a pronoun
                                                 question with a preceding dialogue turn); ':forget Name | attribute'; ':stats'; ':quit'
Options: --store DIR (default: a fresh temporary directory), --device cpu|cuda, --retriever hier|ibf"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jbrain_memory import JBrainMemory  # noqa: E402

NOTES = """Imogen Varga's cell is 415-555-0198.
You can reach Bastian Oyelaran at 212-555-0147.
Imogen Varga lives in Ljubljana.
I had coffee with Bastian Oyelaran this morning. He told me his email is lantern61@posteo.net.
My Wi-Fi password is copper-falcon-27.
Imogen Varga is allergic to sesame.
I should ask Bastian Oyelaran for his address.
The train was late again this morning."""

CHECKS = [  # (question, previous turn, expected answer or None for UNKNOWN)
    ("What is Imogen Varga's phone number?", None, "415-555-0198"),
    ("How can I reach Bastian Oyelaran by phone?", None, "212-555-0147"),
    ("Where does Imogen Varga live?", None, "Ljubljana"),
    ("What's his email?", "I'm about to call Bastian Oyelaran.", "lantern61@posteo.net"),
    ("What's my Wi-Fi password?", None, "copper-falcon-27"),
    ("What can't Imogen Varga eat?", None, "sesame"),
    ("What is Bastian Oyelaran's address?", None, None),          # mentioned, never stated
    ("What is Imogen Varga's work phone?", None, None),           # similar attribute
    ("What is Petra Novakova's email?", None, None),              # unknown person
]
UPDATES = ["Imogen Varga moved to Porto.", "Please forget my Wi-Fi password."]
CHECKS_AFTER = [("Where does Imogen Varga live?", None, "Porto"), ("What's my Wi-Fi password?", None, None)]


def run_checks(mem, checks):
    ok = 0
    for q, turn, exp in checks:
        t0 = time.perf_counter()
        a = mem.ask(q, turn)
        dt = (time.perf_counter() - t0) * 1000
        good = a["answer"] == exp
        ok += good
        print(f"  [{'ok' if good else 'XX'}] {q!r}{' (turn: ' + repr(turn) + ')' if turn else ''} -> {a['answer'] or 'UNKNOWN'}"
              f"  conf={a['confidence']} entity={a['entity']} attr={a['attribute']} ({dt:.0f} ms)")
        if a["evidence"]:
            print(f"        evidence: {a['evidence']!r}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--retriever", default="hier", choices=["hier", "ibf"])
    ap.add_argument("--profile", default="optimized", choices=["optimized", "preregistered"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    tmp = None
    store = args.store
    if store is None:
        tmp = tempfile.mkdtemp(prefix="jbrain_mem_")
        store = tmp
    try:
        t0 = time.perf_counter()
        mem = JBrainMemory(store, device=args.device, retriever=args.retriever, profile=args.profile)
        print(f"loaded J-BRAIN + heads on {mem.dev} in {time.perf_counter() - t0:.1f}s; profile {args.profile}; store at {store}")
        if args.interactive:
            while True:
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
                if line in (":quit", ":q"):
                    break
                if line == ":stats":
                    print(mem.stats())
                elif line.startswith(":forget"):
                    name, attr = [x.strip() for x in line[len(":forget"):].split("|")]
                    print(mem.forget(name, attr))
                elif line.startswith("?"):
                    body = line[1:].strip()
                    turn, q = (body.split("||", 1) if "||" in body else (None, body))
                    print(mem.ask(q.strip(), turn.strip() if turn else None))
                elif line:
                    for o in mem.observe(line):
                        print(o)
            mem.close()
            return
        print("\n== observe notes")
        for o in mem.observe(NOTES):
            print(f"  [{o['intent']}] {o['line']!r} -> {o['ops']}")
        print("\n== questions")
        n1 = run_checks(mem, CHECKS)
        print("\n== updates / deletions")
        for o in mem.observe("\n".join(UPDATES)):
            print(f"  [{o['intent']}] {o['line']!r} -> {o['ops']}")
        n2 = run_checks(mem, CHECKS_AFTER)
        a = mem.ask("Where does Imogen Varga live?")
        prompt = mem.enc.encode("Q: Where does Imogen Varga live?\nA:")
        print("\n== delivery into a J-BRAIN prompt (copy-constrained):", repr(mem.enc.decode(mem.deliver(prompt, a))))
        print("\n== stats", mem.stats())
        total = len(CHECKS) + len(CHECKS_AFTER)
        print(f"\nself-test: {n1 + n2}/{total} correct")
        mem.close()
        if args.selftest and n1 + n2 != total:
            sys.exit(1)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
