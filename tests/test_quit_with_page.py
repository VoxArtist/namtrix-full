"""
Tests for the app quitting when its page is closed.

    python3 tests/test_quit_with_page.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))
import namtrix_bridge as b  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures += 1


clock = {"now": 1000.0}
b._time.monotonic = lambda: clock["now"]


def later(seconds):
    clock["now"] += seconds


check("no tab yet: the browser may still be opening", b.page_closed() is False)
b.do_tab({"id": "a"})
check("a tab is open", b.page_closed() is False)
later(60)
check("a tab that checked in a minute ago is still open", b.page_closed() is False)
later(b.TAB_SILENT_SECONDS)
check("a tab silent for three minutes is gone", b.page_closed() is True)

b.do_tab({"id": "a"})
b.do_tab_bye({"id": "a"})
check("just after goodbye, a reload may still check back in", b.page_closed() is False)
later(3)
b.do_tab({"id": "a"})
later(b.TAB_BYE_GRACE_SECONDS + 1)
check("a reload that checked back in keeps the app", b.page_closed() is False)

b.do_tab_bye({"id": "a"})
later(b.TAB_BYE_GRACE_SECONDS + 1)
check("a closed tab ends the app after the grace", b.page_closed() is True)

b.do_tab({"id": "a"})
b.do_tab({"id": "b"})
b.do_tab_bye({"id": "a"})
later(b.TAB_BYE_GRACE_SECONDS + 1)
check("another tab still open keeps the app", b.page_closed() is False)

try:
    b.do_tab({"id": ""})
    check("a tab needs an id", False)
except b.BridgeError:
    check("a tab needs an id", True)


class Running:
    state = "running"


b._jobs["train"] = Running()
check("training keeps the app alive", b._busy_reason() == "train")
b._jobs["train"] = None
check("nothing running", b._busy_reason() is None)

print()
print("all passed" if not failures else f"{failures} failed")
sys.exit(1 if failures else 0)
