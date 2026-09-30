"""
Tests for renaming a profile folder when its gear is renamed (/api/rename-profile).

    python3 tests/test_rename_profile.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))
import namtrix_bridge as b  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures += 1


def refused(body):
    try:
        b.do_rename_profile(body)
        return None
    except b.BridgeError as e:
        return str(e)


with tempfile.TemporaryDirectory() as tmp:
    chosen = Path(tmp) / "MARSHAL 1897x v2"
    old = chosen / "ENGL SAVAGE SPECIAL EDITION"
    (old / "recordings" / "amp").mkdir(parents=True)
    (old / "recordings" / "amp" / "amp_run_001.wav").write_bytes(b"RIFF")
    new = chosen / "Marshall 1987x Reissue_in Loudness_patch 1top-2 top"

    r = b.do_rename_profile({"from": str(old), "to": str(new)})
    check("renames the folder", r["gearDir"] == str(new) and new.is_dir() and not old.exists())
    check("the takes move with it", (new / "recordings" / "amp" / "amp_run_001.wav").is_file())
    check("the new folder may serve its takes",
          b._may_serve(str((new / "recordings" / "amp" / "amp_run_001.wav").resolve())))

    upper = chosen / new.name.upper()
    r = b.do_rename_profile({"from": str(new), "to": str(upper)})
    check("a change of case only is still a rename", [p.name for p in chosen.iterdir()] == [upper.name])

    other = chosen / "Someone else"
    other.mkdir()
    check("never onto an existing folder", refused({"from": str(upper), "to": str(other)}) is not None)
    check("never out of its folder", refused({"from": str(upper), "to": str(Path(tmp) / "moved")}) is not None)
    check("only a profile folder", refused({"from": str(other), "to": str(chosen / "x")}) is not None)
    check("a missing folder is reported", refused({"from": str(chosen / "gone"), "to": str(chosen / "y")}) is not None)

    class Busy:
        state = "running"
    b._jobs["train"] = Busy()
    try:
        msg = refused({"from": str(upper), "to": str(chosen / "z")})
        check("not while training reads from it", msg is not None and "train" in msg, msg)
    finally:
        b._jobs["train"] = None
    check("left alone when refused", upper.is_dir())

print()
print("all passed" if not failures else f"{failures} failed")
sys.exit(1 if failures else 0)
