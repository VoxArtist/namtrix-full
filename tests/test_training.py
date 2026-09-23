"""
Tests for bridge/training.py: the configs the trainer is handed, and reading
progress and results back off its output folder.

    python3 tests/test_training.py

Needs nothing installed: training.py imports no torch and no numpy.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))

import training  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures += 1


SIGNALS = {"v3": "/s/input.wav", "short_train": "/s/inputTrunc.wav", "short_val": "/s/validation.wav"}
RUNS = [
    {"y": "/r/run_001.wav", "yVal": "/r/run_001_val.wav", "params": {"Drive": 2.5, "Tone": 7}, "delay": 112},
    {"y": "/r/run_002.wav", "yVal": "/r/run_002_val.wav", "params": {"Drive": 9, "Tone": 0}, "delay": 110},
]

# --- standard signal: tail of each take validates, holdouts never appear ---
d = training.data_config([{**r, "yVal": None} for r in RUNS], "v3", SIGNALS)
check("v3: one train entry per run", len(d["train"]) == 2)
check("v3: one validation entry per run", len(d["validation"]) == 2)
check("v3: train uses the standard input", all(e["x_path"] == SIGNALS["v3"] for e in d["train"]))
check("v3: train window stops before the validation tail",
      all(e["start_seconds"] == 10.0 and e["stop_seconds"] == -9.0 for e in d["train"]))
check("v3: validation is the last 9 s of the same take",
      all(e["start_seconds"] == -9.0 and e["y_path"] == t["y_path"]
          for e, t in zip(d["validation"], d["train"])))
check("v3: each take carries its own measured delay",
      [e["delay"] for e in d["train"]] == [112, 110])
check("v3: params are floats", all(isinstance(v, float) for e in d["train"] for v in e["params"].values()))

# --- short signal: separate validation cut ---
d = training.data_config(RUNS, "short", SIGNALS)
check("short: train uses inputTrunc", all(e["x_path"] == SIGNALS["short_train"] for e in d["train"]))
check("short: validation uses validation.wav and the _val takes",
      all(e["x_path"] == SIGNALS["short_val"] and e["y_path"].endswith("_val.wav")
          for e in d["validation"]))
check("short: validation needs no pre-silence (the cut has none)",
      all(e["require_input_pre_silence"] is None for e in d["validation"]))

# --- refusals ---
try:
    training.data_config([], "v3", SIGNALS)
    check("no runs is refused", False)
except training.TrainingError:
    check("no runs is refused", True)
try:
    training.data_config([{**RUNS[0], "yVal": None}], "short", SIGNALS)
    check("short with no validation takes is refused", False)
except training.TrainingError:
    check("short with no validation takes is refused", True)

# --- model: one param per knob, in order, default in the middle ---
m = training.model_config([{"name": "Drive", "min": 0, "max": 10}, {"name": "Tone", "min": 1, "max": 5}])
params = m["net"]["config"]["params"]
check("model: params follow the knobs", [p["name"] for p in params] == ["Drive", "Tone"])
check("model: default is the midpoint", [p["default"] for p in params] == [5.0, 3.0])
check("model: HyperWaveNet", m["net"]["name"] == "HyperWaveNet")
check("learning: epochs applied", training.learning_config(400)["trainer"]["max_epochs"] == 400)
check("learning: template untouched", training._LEARNING["trainer"]["max_epochs"] == 400
      and training.learning_config(50)["trainer"]["max_epochs"] == 50
      and training._LEARNING["trainer"]["max_epochs"] == 400)
check("learning: not pinned to MPS", training.learning_config(1)["trainer"]["accelerator"] == "auto")

# --- names ---
check("safe name strips path characters", training._safe_name('Marshall/1987x: "Reissue".nam') == "Marshall_1987x_ _Reissue_")
check("empty name falls back", training._safe_name("  ") == "model")

# --- reading a run folder back ---
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    ck = run / "lightning_logs" / "version_0" / "checkpoints"
    ck.mkdir(parents=True)
    check("no checkpoints: 0 epochs done", training.epochs_done(run) == 0)
    check("no checkpoints: nothing to adopt", training.adopt_best_checkpoint(run) is False)
    names = ["epoch=0000_step=156_ESR=1.569e-02_MSE=1.1e-05",
             "epoch=0003_step=624_ESR=9.100e-03_MSE=1.0e-05",
             "epoch=0002_step=468_ESR=1.200e-02_MSE=1.0e-05"]
    for n in names:
        (ck / f"{n}.ckpt").write_text("x")
        (ck / f"{n}.nam").write_text(json.dumps({"which": n, "kind": "snapshot"}))
        (ck / f"{n}_parametric.nam").write_text(json.dumps({"which": n, "kind": "parametric"}))
    (ck / "checkpoint_epoch_epoch=0004.ckpt").write_text("x")
    check("epochs done read off checkpoint names", training.epochs_done(run) == 5)
    esr, best = training.best_checkpoint(run)
    check("best checkpoint is the lowest ESR", best.name.startswith("epoch=0003") and abs(esr - 0.0091) < 1e-9)
    check("adopting the best checkpoint exports both files", training.adopt_best_checkpoint(run) is True)
    got = json.loads((run / "model_parametric.nam").read_text())
    check("the adopted parametric model is the best one", got["which"] == names[1] and got["kind"] == "parametric")
    got = json.loads((run / "model.nam").read_text())
    check("the adopted snapshot is the best one", got["which"] == names[1] and got["kind"] == "snapshot")

    # metadata is written into the exported model
    model = run / "m_parametric.nam"
    model.write_text(json.dumps({"config": {"params": [{"name": "Presence"}]}, "metadata": {"date": 1}}))
    training.write_metadata(model, "Marshall 1987x Reissue", "mic1", True, ["Presence"])
    meta = json.loads(model.read_text())["metadata"]
    check("metadata: name carries gear, chain and kind", meta["name"] == "Marshall 1987x Reissue (mic1) parametric")
    check("metadata: make and model split from the gear name",
          meta["gear_make"] == "Marshall" and meta["gear_model"] == "1987x Reissue")
    check("metadata: a mic chain is amp_cab", meta["gear_type"] == "amp_cab")
    check("metadata: existing fields kept", meta["date"] == 1)

print()
print("all passed" if not failures else f"{failures} failed")
sys.exit(1 if failures else 0)
