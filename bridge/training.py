"""
Training and validation, driven from the page instead of from Terminal.

The trainer itself is not bundled: it is PyTorch plus the parametric NAM fork,
a gigabyte of wheels that would dwarf everything else in the app. What the app
does is find an installed trainer, write the three config files from the
session the page already holds, run it, and report progress back. Nothing here
imports torch; the parts that need it run as a child process under the
trainer's own Python.

The dataset is built from this session's recordings. The old route - a shell
script with a fixed data.json - trained whatever that file pointed at, which
was one particular amp's 36 takes regardless of what had just been recorded.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class TrainingError(RuntimeError):
    """Something the user can act on; reported as a clean message."""


# NAMTRIX_SUPPORT_DIR exists for tests, so an install can be exercised without
# touching the real one.
SUPPORT_DIR = Path(os.environ.get("NAMTRIX_SUPPORT_DIR")
                   or Path.home() / "Library" / "Application Support" / "NAMTRIX")
CONFIG_FILE = SUPPORT_DIR / "config.json"
TRAINER_BIN = "nam-full-parametric"

# The trainer NAMTRIX installs for itself, when asked to.
MANAGED_DIR = SUPPORT_DIR / "trainer"
MANAGED_BIN = MANAGED_DIR / "venv" / "bin" / TRAINER_BIN
MANAGED_READY = MANAGED_DIR / "READY"
TRAINER_PYTHON = "3.12.13"
# The trainer's packaging reads its version from git, which a source archive
# does not have. This is the version the same commit reports from a checkout.
TRAINER_VERSION = "0.1.dev1"


# ----------------------------------------------------------------------------
# settings kept between launches
# ----------------------------------------------------------------------------

def load_settings() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:  # noqa: BLE001 - a missing or broken file means defaults
        return {}


def save_settings(settings: dict) -> None:
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2))
    os.replace(tmp, CONFIG_FILE)


# ----------------------------------------------------------------------------
# finding the trainer
# ----------------------------------------------------------------------------

def _python_beside(trainer: Path) -> Path | None:
    for name in ("python", "python3"):
        candidate = trainer.parent / name
        if candidate.exists():
            return candidate
    return None


def _usable(trainer: Path | None) -> bool:
    return bool(trainer) and trainer.is_file() and os.access(trainer, os.X_OK) \
        and _python_beside(trainer) is not None


def find_trainer() -> dict:
    """
    Where the parametric trainer is, if anywhere.

    A path remembered from an earlier "Locate" wins; then the environment; then
    PATH. There is no crawl of the disk: guessing at someone's folders is slow
    and finds the wrong copy as often as the right one.
    """
    candidates = []
    remembered = load_settings().get("trainer")
    if remembered:
        candidates.append(("remembered", Path(remembered)))
    # Only once the install finished: a half-installed environment has the
    # program in place long before it can run.
    if MANAGED_READY.exists():
        candidates.append(("installed", MANAGED_BIN))
    env = os.environ.get("NAMTRIX_TRAINER_BIN")
    if env:
        candidates.append(("environment", Path(env)))
    on_path = shutil.which(TRAINER_BIN)
    if on_path:
        candidates.append(("PATH", Path(on_path)))
    for source, path in candidates:
        path = path.expanduser()
        if _usable(path):
            return {"found": True, "path": str(path), "source": source,
                    "python": str(_python_beside(path))}
    return {"found": False, "path": None, "source": None, "python": None}


def remember_trainer(path: str) -> dict:
    trainer = Path(path).expanduser()
    if trainer.is_dir():
        # accept the environment folder or its bin/ as well as the program itself
        for inner in (trainer / TRAINER_BIN, trainer / "bin" / TRAINER_BIN):
            if inner.exists():
                trainer = inner
                break
    if not _usable(trainer):
        raise TrainingError(
            f"{trainer} is not the parametric trainer. Pick the program called "
            f"'{TRAINER_BIN}' inside the trainer's environment (its bin folder)."
        )
    settings = load_settings()
    settings["trainer"] = str(trainer)
    save_settings(settings)
    return find_trainer()


# ----------------------------------------------------------------------------
# config files
# ----------------------------------------------------------------------------

# The network that has trained well here: HyperWaveNet over the stock
# channels-8 WaveNet stack, the knobs acting through the hypernetwork.
_LAYER = {
    "input_size": 1,
    "condition_size": 1,
    "channels": 8,
    "kernel_sizes": [6] * 14 + [15, 15] + [6] * 7,
    "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13,
                  1, 3, 7, 17, 41, 101, 239],
    "activation": "LeakyReLU",
    "gated": False,
    "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
}

_LEARNING = {
    "torch_compile": {"enabled": False, "mode": "reduce-overhead"},
    "train_dataloader": {"batch_size": 16, "shuffle": True, "pin_memory": False,
                         "drop_last": True, "num_workers": 0},
    "val_dataloader": {"batch_size": 16, "pin_memory": False, "num_workers": 0},
    "trainer": {
        # "auto" is MPS on Apple silicon and CPU elsewhere, where a hardcoded
        # "mps" would refuse to start at all.
        "accelerator": "auto",
        "devices": 1,
        "precision": "32-true",
        "benchmark": False,
        "max_epochs": 400,
        "gradient_clip_val": 1.0,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    },
    "threshold_esr": None,
    "trainer_fit_kwargs": {},
}


def model_config(knobs: list[dict]) -> dict:
    params = []
    for k in knobs:
        lo, hi = float(k["min"]), float(k["max"])
        params.append({"name": k["name"], "min": lo, "max": hi,
                       "default": (lo + hi) / 2.0, "type": "continuous"})
    return {
        "net": {
            "name": "HyperWaveNet",
            "config": {
                "layers": [copy.deepcopy(_LAYER)],
                "head_scale": 0.01,
                "params": params,
                "hypernet": {"selector": {"exclude_suffixes": ["_conv.weight"]}},
            },
        },
        "loss": {"val_loss": "esr", "mrstft_weight": 0.0005},
        "optimizer": {"lr": 0.002, "weight_decay": 3.17e-07},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.994}},
    }


def learning_config(epochs: int) -> dict:
    cfg = copy.deepcopy(_LEARNING)
    cfg["trainer"]["max_epochs"] = int(epochs)
    return cfg


# Room left at the end of every validation clip for the network's receptive
# field (about 6,350 samples for this architecture), with some to spare.
_RECEPTIVE_MARGIN = 8192
_V3_VALIDATION_SECONDS = 9.0


def _wav_frames(path) -> tuple[int, int]:
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes(), w.getframerate()


def _validation_ny(runs: list[dict], clip_frames: int) -> int:
    """
    One window length for every validation clip.

    Each take's latency trims its clip by a different few samples, and the
    trainer refuses validation clips of unequal length ("Mismatch between ny of
    datasets"). Letting it size each clip itself failed a real 79-take session
    over seven samples; sizing them all to fit the take with the most latency
    cannot.
    """
    worst = max(abs(int(r.get("delay") or 0)) for r in runs)
    ny = clip_frames - worst - _RECEPTIVE_MARGIN
    if ny < 4096:
        raise TrainingError(
            f"The measured latency ({worst} samples) leaves too little of the "
            "validation clip to check the model against. Test the route again: "
            "a real interface is usually under 1,000 samples."
        )
    return ny


# Below this a take is silence, not a quiet setting. On a real 79-take session
# the silent takes (a volume, gain or lead at 0) sat near -86 dBFS and the
# quietest real ones near -57; -45 would have thrown fourteen real takes out of
# validation along with the eleven silent ones.
SILENT_DBFS = -60.0


def clip_rms_dbfs(path, last_seconds: float | None = None) -> float:
    """RMS level of a take (or its last few seconds), cheaply: every 16th sample."""
    import array
    import math
    import wave

    with wave.open(str(path)) as w:
        rate, width, channels, frames = (w.getframerate(), w.getsampwidth(),
                                         w.getnchannels(), w.getnframes())
        if last_seconds:
            w.setpos(max(0, frames - int(last_seconds * rate)))
            frames = min(frames, int(last_seconds * rate))
        raw = w.readframes(frames)
    stride = width * channels * 16
    total, count = 0.0, 0
    full = float(2 ** (8 * width - 1))
    for i in range(0, len(raw) - width + 1, stride):
        v = int.from_bytes(raw[i:i + width], "little", signed=(width > 1)) / full
        total += v * v
        count += 1
    rms = math.sqrt(total / count) if count else 0.0
    return 20 * math.log10(rms) if rms > 0 else float("-inf")


def data_config(runs: list[dict], preset: str, signals: dict, is_silent=None) -> dict:
    """
    One entry per recorded run.

    Validation during training is always unseen audio at a trained setting, never
    a holdout run: the checkpoint kept is the one that scores best on validation,
    so validating on the holdouts would quietly tune the model to them and leave
    the separate holdout check nothing honest to measure.

    Standard signal: train on 10 s to the last 9 s of each take, validate on that
    last 9 s. Short pair: train on inputTrunc.wav, validate on the separately
    recorded validation.wav take of the same run.

    A near-silent take stays in training - it teaches that those settings are
    quiet - but not in validation. The trainer averages its score per take, and
    one silent take's ratio is noise in the thousands (a real session scored
    5,925 after its first epoch), which would decide which checkpoint is kept.
    is_silent(path, last_seconds) says which; without it nothing is left out.
    """
    train, validation = [], []
    if not runs:
        raise TrainingError("No recorded runs to train on.")
    if preset == "short":
        val_ny = _validation_ny(runs, _wav_frames(signals["short_val"])[0]) \
            if Path(signals["short_val"]).exists() else None
    else:
        rate = _wav_frames(signals["v3"])[1] if Path(signals["v3"]).exists() else 48000
        val_ny = _validation_ny(runs, int(_V3_VALIDATION_SECONDS * rate))
    for run in runs:
        base = {"params": {k: float(v) for k, v in run["params"].items()},
                "delay": int(run.get("delay") or 0)}
        if preset == "short":
            train.append({**base, "x_path": signals["short_train"], "y_path": run["y"],
                          "start_seconds": 0.0, "stop_seconds": None, "ny": 8192})
            if run.get("yVal") and not (is_silent and is_silent(run["yVal"], None)):
                validation.append({**base, "x_path": signals["short_val"],
                                   "y_path": run["yVal"], "start_seconds": 0.0,
                                   "stop_seconds": None, "ny": val_ny,
                                   "require_input_pre_silence": None})
        else:
            train.append({**base, "x_path": signals["v3"], "y_path": run["y"],
                          "start_seconds": 10.0, "stop_seconds": -9.0, "ny": 8192})
            if is_silent and is_silent(run["y"], _V3_VALIDATION_SECONDS):
                continue
            validation.append({**base, "x_path": signals["v3"], "y_path": run["y"],
                               "start_seconds": -9.0, "stop_seconds": None, "ny": val_ny,
                               "require_input_pre_silence": None})
    if not train:
        raise TrainingError("No recorded runs to train on.")
    if not validation:
        raise TrainingError("No validation take has sound in it, so training has nothing "
                            "to check itself against. Check the recordings are not silent.")
    return {"type": "parametric", "common": {"delay": 0},
            "train": train, "validation": validation}


# ----------------------------------------------------------------------------
# metadata: the fields plugins and Tone3000 show, which the trainer leaves empty
# ----------------------------------------------------------------------------

def _gear_type(chain: str, knob_names: list[str]) -> str:
    c = (chain or "").lower()
    if any(w in c for w in ("cab", "mic", "full", "room")):
        return "amp_cab"
    names = " ".join(knob_names).lower()
    if any(w in names for w in ("presence", "master", "volume i", "bright")):
        return "amp"
    return "pedal" if knob_names else "amp"


def write_metadata(model_path: Path, gear: str, chain: str, parametric: bool,
                   knob_names: list[str]) -> None:
    doc = json.loads(model_path.read_text())
    parts = gear.split()
    name = gear or model_path.stem
    if chain:
        name += f" ({chain})"
    if parametric:
        name += " parametric"
    meta = dict(doc.get("metadata") or {})
    meta.update({
        "name": name,
        "modeled_by": os.environ.get("USER", "") or "unknown",
        "gear_make": parts[0] if parts else None,
        "gear_model": " ".join(parts[1:]) or None,
        "gear_type": _gear_type(chain, knob_names),
    })
    doc["metadata"] = meta
    model_path.write_text(json.dumps(doc))


# ----------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip())
    cleaned = re.sub(r"\.nam$", "", cleaned, flags=re.I).strip(" .")
    return cleaned or "model"


def _child_env() -> dict:
    cache = SUPPORT_DIR / "cache"
    env = dict(os.environ)
    for key, sub in (("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", "xdg"),
                     ("HF_HOME", "hf")):
        path = cache / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # PyInstaller leaves these pointing into the bundle; the trainer's Python
    # would then load our libraries instead of its own.
    for key in ("PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH"):
        env.pop(key, None)
    return env


_EPOCH = re.compile(r"Epoch (\d+)")
# The trainer's progress bar: "Epoch 3/399 ━━━━╸ 132/156 0:00:37 • 0:00:07 3.48it/s"
_PROGRESS = re.compile(r"Epoch (\d+)/(\d+)\D+?(\d+)/(\d+)")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[=>()][0-9A-B]?")
_ESR = re.compile(r"ESR=([0-9.eE+-]+)")
_CKPT_EPOCH = re.compile(r"epoch=(\d+)")


def epochs_done(run_dir: Path) -> int:
    """
    Completed epochs, read off the checkpoints the trainer writes each epoch.

    The console is no use for this: through a pipe the trainer's progress bar
    draws nothing until the run is over.
    """
    latest = -1
    for ckpt in run_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt"):
        m = _CKPT_EPOCH.search(ckpt.name)
        if m:
            latest = max(latest, int(m.group(1)))
    return latest + 1


def best_checkpoint(run_dir: Path) -> tuple[float | None, Path | None]:
    best = (None, None)
    for ckpt in run_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt"):
        m = _ESR.search(ckpt.name)
        if not m:
            continue
        try:
            esr = float(m.group(1).rstrip("."))
        except ValueError:
            continue
        if best[0] is None or esr < best[0]:
            best = (esr, ckpt)
    return best


def adopt_best_checkpoint(run_dir: Path) -> bool:
    """
    Export after a stop, from the best checkpoint.

    The trainer means to export the best model when interrupted, but the
    Lightning it runs on answers Ctrl-C with an immediate exit, so that code
    never runs. It has already written a ready .nam pair beside every scored
    checkpoint, though, so the best pair is copied into place instead.
    """
    _, ckpt = best_checkpoint(run_dir)
    if ckpt is None:
        return False
    copied = False
    for suffix in ("", "_parametric"):
        src = ckpt.with_name(f"{ckpt.stem}{suffix}.nam")
        if src.exists():
            shutil.copy2(src, run_dir / f"model{suffix}.nam")
            copied = copied or suffix == "_parametric"
    return copied


class TrainJob:
    """One training session: every requested chain, one after another."""

    def __init__(self, request: dict, out_dir: Path, trainer: dict, signals: dict):
        self.request = request
        self.out_dir = out_dir
        self.trainer = trainer
        self.signals = signals
        self.state = "running"
        self.error = None
        self.chains = [{"name": c["name"], "modelName": _safe_name(c["modelName"]),
                        "state": "waiting", "epoch": 0, "bestEsr": None, "silentLeftOut": 0,
                        "runDir": None, "files": [], "log": None}
                       for c in request["chains"]]
        self.current = 0
        self.epochs = int(request.get("epochs") or 400)
        self.started = time.time()
        self.finished = None
        self._proc = None
        self._stop = False
        self._tail: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="namtrix-train")

    def start(self):
        self._thread.start()

    def stop(self):
        """Stop early. The trainer still exports the best model it has so far."""
        self._stop = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:  # noqa: BLE001
                proc.send_signal(signal.SIGINT)

    def status(self) -> dict:
        chain = self.chains[self.current] if self.current < len(self.chains) else None
        if chain and chain["runDir"] and chain["state"] == "training":
            run_dir = Path(chain["runDir"])
            esr, _ = best_checkpoint(run_dir)
            if esr is not None:
                chain["bestEsr"] = esr
            chain["epoch"] = max(chain["epoch"], epochs_done(run_dir))
        done_chains = sum(1 for c in self.chains if c["state"] in ("done", "stopped"))
        within = 0.0
        if chain and chain["state"] == "training":
            within = (chain["epoch"] + (chain.get("step", 0) / chain["steps"]
                                        if chain.get("steps") else 0)) / self.epochs
        fraction = min(1.0, (done_chains + within) / max(1, len(self.chains)))
        if self.state == "done":
            fraction = 1.0
        return {
            "state": self.state,
            "error": self.error,
            "outDir": str(self.out_dir),
            "epochs": self.epochs,
            "current": self.current,
            "chains": self.chains,
            "elapsed": (self.finished or time.time()) - self.started,
            "fraction": fraction,
            "tail": self._tail[-12:],
        }

    # -- worker ------------------------------------------------------------

    def _run(self):
        try:
            for i, spec in enumerate(self.request["chains"]):
                if self._stop:
                    self.chains[i]["state"] = "skipped"
                    continue
                self.current = i
                self._train_chain(i, spec)
            self.state = "stopped" if self._stop else "done"
        except TrainingError as exc:
            self.state, self.error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.time()

    def _train_chain(self, i: int, spec: dict):
        info = self.chains[i]
        info["state"] = "preparing"
        base = self.out_dir / info["modelName"]
        config_dir = base / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        preset = self.request.get("preset") or "v3"
        # The DI goes beside the configs so the folder retrains on its own later,
        # without depending on where this app happened to live.
        local_signals = {}
        for key in (("short_train", "short_val") if preset == "short" else ("v3",)):
            src = Path(self.signals[key])
            dst = config_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            local_signals[key] = str(dst)

        knobs = self.request["knobs"]
        info["state"] = "checking"
        silent = lambda path, last: clip_rms_dbfs(path, last) < SILENT_DBFS  # noqa: E731
        data = data_config(spec["runs"], preset, local_signals, silent)
        info["silentLeftOut"] = len(spec["runs"]) - len(data["validation"])
        files = {
            "data": data,
            "model": model_config(knobs),
            "learning": learning_config(self.epochs),
        }
        for name, payload in files.items():
            (config_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))

        log_path = base / f"training-{_dt.datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        info["log"] = str(log_path)
        before = {p for p in base.iterdir() if p.is_dir()}
        cmd = ["/usr/bin/caffeinate", "-i", self.trainer["path"],
               str(config_dir / "data.json"), str(config_dir / "model.json"),
               str(config_dir / "learning.json"), str(base), "--no-plots"]
        info["state"] = "preparing"
        info["step"], info["steps"] = 0, 0
        # Through a pipe the trainer's progress bar draws nothing until the very
        # end, so it gets a pseudo-terminal instead and its live bar is read
        # back: that is the only progress it reports within an epoch.
        import fcntl
        import pty
        import struct
        import termios

        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))
        env = _child_env()
        env.update({"COLUMNS": "200", "LINES": "50", "TERM": "xterm-256color"})
        last_logged_epoch = -1
        with open(log_path, "w", encoding="utf-8") as log:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=slave,
                                          stderr=slave, env=env, start_new_session=True)
            os.close(slave)
            buf = ""
            while True:
                try:
                    chunk = os.read(master, 8192)
                except OSError:          # the terminal closes when the trainer exits
                    break
                if not chunk:
                    break
                buf += _ANSI.sub("", chunk.decode("utf-8", "replace"))
                # progress bars redraw with \r, so either ends a line
                parts = re.split(r"[\r\n]", buf)
                buf = parts.pop()
                for line in parts:
                    line = line.strip()
                    if not line:
                        continue
                    m = _PROGRESS.search(line)
                    if m:
                        epoch, last, step, steps = (int(g) for g in m.groups())
                        info["state"] = "training"
                        info["epoch"] = max(info["epoch"], epoch)
                        info["step"], info["steps"] = step, steps
                        # the log keeps one line per epoch, not every redraw
                        if step == steps and epoch != last_logged_epoch:
                            last_logged_epoch = epoch
                            log.write(line + "\n")
                            log.flush()
                        continue
                    if line.startswith(("Validation", "Sanity Checking")):
                        continue          # its own progress bar, redrawn constantly
                    log.write(line + "\n")
                    log.flush()
                    self._tail.append(line[-240:])
                    del self._tail[:-40]
                if info["runDir"] is None:
                    new = [p for p in base.iterdir() if p.is_dir() and p not in before
                           and p.name != "config"]
                    if new:
                        info["runDir"] = str(new[0])
            code = self._proc.wait()
            os.close(master)

        if info["runDir"] is None:
            new = [p for p in base.iterdir() if p.is_dir() and p not in before
                   and p.name != "config"]
            info["runDir"] = str(new[0]) if new else None
        run_dir = Path(info["runDir"]) if info["runDir"] else None
        exported = run_dir and (run_dir / "model_parametric.nam").exists()
        if not exported and self._stop and run_dir:
            exported = adopt_best_checkpoint(run_dir)
        if not exported and self._stop:
            # Stopped inside the first epoch: no checkpoint exists yet. That is
            # the user's choice, not a failure.
            info["state"] = "stopped"
            return
        if code != 0 and not exported:
            info["state"] = "failed"
            reason = next((line for line in reversed(self._tail)
                           if re.search(r"(Error|Exception)\b", line)), None) \
                or (self._tail[-1] if self._tail else f"exit code {code}")
            raise TrainingError(
                f"Training {info['name']} stopped: {reason}"
                f"\n\nThe full log is {log_path}."
            )
        if not exported:
            info["state"] = "failed"
            raise TrainingError(f"Training {info['name']} finished without exporting a model.")

        esr, _ = best_checkpoint(run_dir)
        info["bestEsr"] = esr
        info["epoch"] = max(info["epoch"], epochs_done(run_dir))
        knob_names = [k["name"] for k in knobs]
        gear = self.request.get("gear") or ""
        for suffix, parametric in (("_parametric", True), ("", False)):
            src = run_dir / f"model{suffix}.nam"
            if not src.exists():
                continue
            dst = run_dir / f"{info['modelName']}{suffix}.nam"
            if src != dst:
                src.rename(dst)
            try:
                write_metadata(dst, gear, info["name"], parametric, knob_names)
            except Exception:  # noqa: BLE001 - metadata is a nicety, the model is not
                pass
            info["files"].append(str(dst))
        info["state"] = "stopped" if self._stop else "done"


class InstallJob:
    """
    Install the trainer into NAMTRIX's own folder, with uv.

    uv fetches its own Python, then the pinned packages and the trainer itself
    from its repository, so nothing needs to be on the Mac beforehand and
    nothing outside this folder is touched. Deleting the folder removes it.
    """

    PHASES = ["Getting ready", "Downloading Python", "Downloading PyTorch and the trainer",
              "Checking the install"]

    def __init__(self, uv_source: Path, requirements: Path):
        self.uv_source = uv_source
        self.requirements = requirements
        self.state = "running"
        self.error = None
        self.phase = 0
        self.detail = ""
        self.downloads = []
        self.started = time.time()
        self.finished = None
        self._proc = None
        self._cancel = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="namtrix-install")

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancel = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                proc.terminate()

    def status(self) -> dict:
        return {
            "state": self.state, "error": self.error,
            "phase": self.phase, "phases": self.PHASES,
            "phaseLabel": self.PHASES[min(self.phase, len(self.PHASES) - 1)],
            "detail": self.detail, "downloads": self.downloads[-6:],
            "elapsed": (self.finished or time.time()) - self.started,
        }

    def _step(self, cmd, env):
        if self._cancel:
            raise TrainingError("Cancelled.")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      env=env, start_new_session=True)
        lines = []
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            lines.append(line)
            m = re.match(r"Downloading (\S+) \(([^)]+)\)", line)
            if m:
                self.downloads.append(f"{m.group(1)} ({m.group(2)})")
            self.detail = line[-160:]
        code = self._proc.wait()
        if self._cancel:
            raise TrainingError("Cancelled.")
        if code != 0:
            raise TrainingError("\n".join(lines[-8:]) or f"exit code {code}")
        return lines

    def _run(self):
        try:
            import platform

            if sys.platform != "darwin" or platform.machine() != "arm64":
                raise TrainingError("The automatic install is for Apple silicon Macs (M1 and later).")
            SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(SUPPORT_DIR).free
            if free < 4 * 1024 ** 3:
                raise TrainingError(
                    f"The install needs about 4 GB free while it runs (1.2 GB once done); "
                    f"this disk has {free / 1024 ** 3:.1f} GB."
                )

            # uv runs from our own folder, not from inside the app: a helper
            # program inside a downloaded app still carries the download's
            # quarantine flag, and macOS would stop it with a dialog of its own.
            bin_dir = SUPPORT_DIR / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            uv = bin_dir / "uv"
            shutil.copy2(self.uv_source, uv)
            uv.chmod(0o755)
            subprocess.run(["/usr/bin/xattr", "-d", "com.apple.quarantine", str(uv)],
                           capture_output=True)

            if MANAGED_DIR.exists():
                shutil.rmtree(MANAGED_DIR)
            MANAGED_DIR.mkdir(parents=True)
            env = _child_env()
            env.update({
                "UV_PYTHON_INSTALL_DIR": str(MANAGED_DIR / "python"),
                "UV_CACHE_DIR": str(MANAGED_DIR / "cache"),
                "UV_PYTHON_PREFERENCE": "only-managed",
                "UV_NO_CONFIG": "1",
                "NO_COLOR": "1",
                "SETUPTOOLS_SCM_PRETEND_VERSION": TRAINER_VERSION,
            })
            venv = MANAGED_DIR / "venv"

            self.phase = 1
            self._step([str(uv), "venv", str(venv), "--python", TRAINER_PYTHON], env)
            self.phase = 2
            self._step([str(uv), "pip", "install", "--python", str(venv / "bin" / "python"),
                        "-r", str(self.requirements)], env)
            self.phase = 3
            self._step([str(venv / "bin" / "python"), "-c",
                        "import torch, nam.train.parametric, nam.models.parametric; "
                        "print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"],
                       env)
            if not MANAGED_BIN.exists():
                raise TrainingError(f"The install finished without {TRAINER_BIN}.")
            # The download cache is as big again as the install and never read twice.
            shutil.rmtree(MANAGED_DIR / "cache", ignore_errors=True)
            MANAGED_READY.write_text(json.dumps({"python": TRAINER_PYTHON,
                                                 "installed": time.time()}))
            self.state = "done"
        except TrainingError as exc:
            self.state = "cancelled" if self._cancel else "failed"
            self.error = str(exc)
            self._cleanup()
        except Exception as exc:  # noqa: BLE001
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self._cleanup()
        finally:
            self.finished = time.time()

    def _cleanup(self):
        # Half an environment is worse than none: find_trainer would never see
        # it (no READY file), but it would sit there taking a gigabyte.
        shutil.rmtree(MANAGED_DIR, ignore_errors=True)


class ValidateJob:
    """Render every holdout run through the trained model and score it."""

    def __init__(self, request: dict, trainer: dict, script: Path, signals: dict):
        self.request = request
        self.trainer = trainer
        self.script = script
        self.signals = signals
        self.state = "running"
        self.error = None
        self.result = None
        self.progress = ""
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="namtrix-validate")

    def start(self):
        self._thread.start()

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "result": self.result,
                "progress": self.progress}

    def _run(self):
        work = SUPPORT_DIR / "validate"
        work.mkdir(parents=True, exist_ok=True)
        req_path, out_path = work / "request.json", work / "result.json"
        if out_path.exists():
            out_path.unlink()
        preset = self.request.get("preset") or "v3"
        payload = dict(self.request)
        payload["x"] = self.signals["short_train" if preset == "short" else "v3"]
        req_path.write_text(json.dumps(payload))
        try:
            proc = subprocess.Popen(
                [self.trainer["python"], str(self.script), str(req_path), str(out_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_child_env(),
            )
            lines = []
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    lines.append(line)
                    if line.startswith("progress:"):
                        self.progress = line[len("progress:"):].strip()
            code = proc.wait()
            if code != 0 or not out_path.exists():
                raise TrainingError("Validation failed:\n" + "\n".join(lines[-8:]))
            self.result = json.loads(out_path.read_text())
            self.state = "done"
        except TrainingError as exc:
            self.state, self.error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
