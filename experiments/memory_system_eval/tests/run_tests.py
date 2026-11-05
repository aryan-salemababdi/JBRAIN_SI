"""Minimal test runner (pytest is not installed in the offline environment): runs every test_* function in tests/test_*.py,
or in the modules named on the command line. Exit code 1 if any test fails."""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent
files = [here / f for f in sys.argv[1:]] if len(sys.argv) > 1 else sorted(here.glob("test_*.py"))
fails = 0
for f in files:
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS {f.name}::{name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {f.name}::{name}\n{traceback.format_exc()}", flush=True)
print(f"{fails} failed")
sys.exit(1 if fails else 0)
