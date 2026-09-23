"""
Score a trained parametric model against the holdout runs.

Run by the NAMTRIX bridge under the trainer's own Python, because it needs torch
and the trainer's model classes; the app itself carries neither.

    python validate_model.py request.json result.json

For each holdout run it plays the DI through the model at that run's knob
settings, lines the real take up by the chain's measured delay, and computes
the Error-to-Signal Ratio the page has always used:

    ESR = sum((y - y_hat)^2) / sum(y^2)

A take with every control at minimum is silent by construction; the ratio is
meaningless there (the model rightly predicts silence and the whole residual is
hiss), so it is scored on level instead, exactly as the page did when these
files were picked by hand.

The model scored is the checkpoint the trainer exported: the one with the best
validation ESR. Its prediction for each run is written next to it, so the
numbers can be checked by ear.
"""

import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch

SILENCE_DBFS = -45.0
SILENCE_MATCH_DB = 6.0


def read_wav(path):
    with wave.open(str(path)) as w:
        rate, width, channels, frames = (w.getframerate(), w.getsampwidth(),
                                         w.getnchannels(), w.getnframes())
        raw = w.readframes(frames)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / float(2 ** 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(2 ** 31)
    else:
        raise SystemExit(f"{path}: unsupported sample width")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav(path, samples, rate):
    ints = (np.clip(samples, -1.0, 1.0) * (2 ** 23 - 1)).astype("<i4")
    raw = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(rate)
        w.writeframes(raw)


def best_checkpoint(run_dir):
    best = None
    for ckpt in run_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt"):
        m = re.search(r"ESR=([0-9.eE+-]+)", ckpt.name)
        if m:
            esr = float(m.group(1).rstrip("."))
            if best is None or esr < best[0]:
                best = (esr, ckpt)
    if best is None:
        raise SystemExit(f"No scored checkpoint in {run_dir}")
    return best[1]


def load_model(run_dir):
    from nam.train import util as _util

    model_config = json.loads((run_dir / "config_model.json").read_text())
    cls = _util.resolve_lightning_module_class(model_config)
    model = cls.load_from_checkpoint(str(best_checkpoint(run_dir)),
                                     map_location=torch.device("cpu"),
                                     **cls.parse_config(model_config))
    model.eval()
    names = [p["name"] for p in model_config["net"]["config"]["params"]]
    return model, names


def score(y, y_hat):
    n = min(len(y), len(y_hat))
    y, y_hat = y[:n].astype(np.float64), y_hat[:n].astype(np.float64)
    err = float(np.sum((y - y_hat) ** 2))
    sig = float(np.sum(y ** 2))
    mod = float(np.sum(y_hat ** 2))
    to_db = lambda e: 10 * np.log10(e / n) if e > 0 else None  # noqa: E731
    target_db, model_db = to_db(sig), to_db(mod)
    silent = not sig > 0 or target_db < SILENCE_DBFS
    return {
        "silent": bool(silent),
        "targetDb": target_db,
        "modelDb": model_db,
        "score": None if silent else err / sig,
        "levelMatch": (model_db is None or model_db <= target_db + SILENCE_MATCH_DB)
        if silent else None,
    }


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    x, rate = read_wav(request["x"])
    x_t = torch.from_numpy(x)

    total = sum(len(c["holdout"]) for c in request["chains"])
    done = 0
    result = {"chains": []}
    for chain in request["chains"]:
        run_dir = Path(chain["runDir"])
        model, names = load_model(run_dir)
        runs = {}
        for h in chain["holdout"]:
            done += 1
            print(f"progress: {chain['name']} holdout {h['run']} ({done}/{total})",
                  flush=True)
            y_path = Path(h["y"])
            if not y_path.exists():
                runs[str(h["run"])] = {"missing": True}
                continue
            y, y_rate = read_wav(y_path)
            if y_rate != rate:
                raise SystemExit(f"{y_path.name} is {y_rate} Hz; the DI is {rate} Hz")
            params = torch.tensor([float(h["params"][n]) for n in names],
                                  dtype=torch.float32)
            with torch.no_grad():
                pred = model(x_t, params).cpu().numpy().flatten()
            # the real take lags the model by the chain's round trip
            d = int(h.get("delay") or 0)
            if d > 0:
                y_al, p_al = y[d:], pred[:len(pred) - d]
            elif d < 0:
                y_al, p_al = y[:len(y) + d], pred[-d:]
            else:
                y_al, p_al = y, pred
            runs[str(h["run"])] = score(y_al, p_al)
            write_wav(run_dir / "holdout_renders" / f"{y_path.stem}_model.wav", pred, rate)
        rated = [r["score"] for r in runs.values() if r.get("score") is not None]
        result["chains"].append({
            "name": chain["name"],
            "runDir": str(run_dir),
            "runs": runs,
            "average": float(np.mean(rated)) if rated else None,
            "worst": float(np.max(rated)) if rated else None,
        })
    out_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
