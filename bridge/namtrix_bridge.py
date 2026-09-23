"""
NAMTRIX capture bridge.

A browser cannot address physical channel 3 of an audio interface, so this small
local program does the audio and serves the page that drives it. Same origin,
one command, nothing to connect.

    python bridge/namtrix_bridge.py            # then open http://127.0.0.1:8765

It plays the reamp signal out one output channel and records every capture chain
at once, because the stream is opened at the device's full width anyway. Each
recording is measured for round-trip delay using the trainer's own blip preamble,
so the per-chain figures (our amp 47 / full 112) come back without hand work.

Saved takes keep the timebase of the signal file: the preamble this program adds
is stripped, the rig's delay is left in. That is what a DAW-recorded take looks
like, and what the trainer expects alongside an explicit `delay`. Pre-aligning
here would risk the correction being applied twice.

Needs only numpy and sounddevice. The delay measurement is vendored in
`latency.py` rather than imported from the trainer, so this runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time as _time
import traceback
import wave
import socketserver
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Importable whether this is run as a script, as a module, or from inside a
# PyInstaller bundle, none of which agree about what is on sys.path.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import training  # noqa: E402 - needs the sys.path line above

VERSION = "0.3.0"
PREAMBLE_PAD_SECONDS = 0.25   # silence after the preamble before the signal starts
OPEN_TIMEOUT_SECONDS = 10.0   # PortAudio can block indefinitely opening a dead device

# Set when a capture thread never returned: the device layer cannot be trusted again
# in this process, so say so rather than hanging every later request too.
_wedged = {"device": False}

# Takes written during this run, and the folders the user chose for recordings.
# /api/take serves a .wav only from one of these, so a crafted path cannot
# walk out into the rest of the disk.
_written_takes: set[str] = set()
_allowed_dirs: set[str] = set()
_written_lock = threading.Lock()

# The reamp signals ship inside the app: the person recording should not have to
# find, download or point at a DI file. Keys are what the page asks for.
SIGNAL_FILES = {
    "v3": "input.wav",               # NAM v3.0.0 standard input, 190 s
    "short_train": "inputTrunc.wav", # truncated v3 for parametric sessions, 38 s
    "short_val": "validation.wav",   # its separate 7 s validation cut
}

# Gain advice for the test route, in dBFS peak on the recorded return. Below the
# window the take wastes converter resolution; above it one hot run clips.
GAIN_LOW_DBFS = -18.0
GAIN_HIGH_DBFS = -3.0
LIVE_DBFS = -60.0

# Set by "cancel this run": the capture in progress stops and saves nothing.
_capture_cancel = threading.Event()

# Only one training or validation job at a time; the GPU is not shareable either.
_jobs = {"train": None, "validate": None, "install": None}


class BridgeError(RuntimeError):
    """Something the user can act on; reported as a clean message, not a stack."""


def _json_safe(value):
    """
    Recursively replace inf/-inf/nan with None.

    dbfs() and peak_dbfs() return float('-inf') for true digital silence - an
    unpatched input, or a channel with nothing connected - which is a normal,
    expected reading, not a fault. Python's json.dumps serialises it as the
    bare token -Infinity, which is valid in Python's own relaxed reader but not
    in the JSON spec the browser's JSON.parse enforces: it throws, the fetch
    that otherwise succeeded looks like a failure, and the page reports
    "Bridge returned 200" instead of the silence reading it was sent. Replacing
    it with null keeps the response valid JSON and lets the page's own
    isFinite() checks report "silent" the way they were always meant to.
    """
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _signals_dir() -> Path:
    return _bundled_root() / "signals"


def signal_path(key: str) -> Path:
    if key not in SIGNAL_FILES:
        raise BridgeError(f"Unknown reamp signal '{key}'.")
    path = _signals_dir() / SIGNAL_FILES[key]
    if not path.is_file():
        raise BridgeError(f"The app is missing its reamp signal {path.name}. Reinstall it.")
    return path


def _allow_dir(path) -> None:
    with _written_lock:
        _allowed_dirs.add(str(Path(path).expanduser().resolve()))


def _may_serve(resolved: str) -> bool:
    if not resolved.lower().endswith(".wav"):
        return False
    with _written_lock:
        if resolved in _written_takes:
            return True
        return any(resolved.startswith(d.rstrip("/") + "/") for d in _allowed_dirs)


# ----------------------------------------------------------------------------
# native dialogs: a web page cannot learn a folder's real path, so the app asks
# ----------------------------------------------------------------------------

def _applescript_string(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _osascript_choose(kind: str, prompt: str, default: str | None = None) -> str | None:
    """A macOS choose-folder/choose-file dialog. None when cancelled."""
    # For automated tests only: a native dialog cannot be clicked by a script.
    canned = os.environ.get("NAMTRIX_DIALOG_ANSWER")
    if canned is not None:
        return canned or None
    where = ""
    if default and Path(default).expanduser().exists():
        where = f" default location (POSIX file {_applescript_string(str(Path(default).expanduser()))})"
    script = (
        "activate\n"
        "try\n"
        f"  return POSIX path of (choose {kind} with prompt {_applescript_string(prompt)}{where})\n"
        "on error number -128\n"
        '  return ""\n'
        "end try"
    )
    proc = subprocess.run(["/usr/bin/osascript", "-e", script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise BridgeError(f"The dialog could not open: {proc.stderr.strip()}")
    chosen = proc.stdout.strip()
    return chosen.rstrip("/") or None if chosen else None


def do_choose_folder(body):
    chosen = _osascript_choose("folder", body.get("prompt") or "Choose a folder",
                               body.get("default"))
    if not chosen:
        return {"ok": True, "cancelled": True, "path": None}
    _allow_dir(chosen)
    return {"ok": True, "cancelled": False, "path": chosen}


def do_allow_folder(body):
    path = Path(body.get("path") or "").expanduser()
    if not str(body.get("path") or "").strip() or not path.is_dir():
        return {"ok": True, "allowed": False}
    _allow_dir(path)
    return {"ok": True, "allowed": True}


def do_reveal(body):
    path = Path(body.get("path") or "").expanduser()
    if not path.exists():
        raise BridgeError(f"{path} no longer exists.")
    subprocess.run(["/usr/bin/open", "-R", str(path)], check=False)
    return {"ok": True}


# ----------------------------------------------------------------------------
# user gear presets, kept outside the browser so a new port does not lose them
# ----------------------------------------------------------------------------

PRESETS_FILE = training.SUPPORT_DIR / "presets.json"


def load_presets():
    try:
        data = json.loads(PRESETS_FILE.read_text())
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def do_save_presets(body):
    presets = body.get("presets")
    if not isinstance(presets, list):
        raise BridgeError("Expected a list of presets.")
    training.SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PRESETS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(presets, indent=2))
    os.replace(tmp, PRESETS_FILE)
    return {"ok": True, "presets": presets}


# ----------------------------------------------------------------------------
# training and validation
# ----------------------------------------------------------------------------

def _signal_paths() -> dict:
    return {k: str(signal_path(k)) for k in SIGNAL_FILES}


def _job_busy():
    for kind, job in _jobs.items():
        if job is not None and job.state == "running":
            return kind
    return None


def do_trainer_locate(body):
    chosen = _osascript_choose(
        "file", "Locate the parametric trainer (nam-full-parametric, in its "
        "environment's bin folder)")
    if not chosen:
        return {"ok": True, "cancelled": True, "trainer": trainer_status()}
    try:
        training.remember_trainer(chosen)
        return {"ok": True, "cancelled": False, "trainer": trainer_status()}
    except training.TrainingError as exc:
        raise BridgeError(str(exc)) from exc


def _bundled_file(name: str) -> Path | None:
    """A file shipped beside the bridge: in the checkout, or inside the app."""
    for base in (Path(__file__).resolve().parent, _bundled_root()):
        if (base / name).exists():
            return base / name
    return None


def trainer_status() -> dict:
    job = _jobs["install"]
    return {**training.find_trainer(),
            "install": job.status() if job else None,
            "canInstall": _uv_source() is not None}


def _uv_source() -> Path | None:
    shipped = _bundled_file("uv")
    if shipped:
        return shipped
    env = os.environ.get("NAMTRIX_UV")
    if env and Path(env).exists():
        return Path(env)
    import shutil as _shutil

    found = _shutil.which("uv")
    return Path(found) if found else None


def do_trainer_install(body):
    job = _jobs["install"]
    if job is not None and job.state == "running":
        return {"ok": True, "trainer": trainer_status()}
    uv = _uv_source()
    requirements = _bundled_file("trainer-requirements.txt")
    if uv is None or requirements is None:
        raise BridgeError("This copy of NAMTRIX cannot install the trainer (its installer "
                          "is missing). Reinstall the app, or locate an existing trainer.")
    job = training.InstallJob(uv, requirements)
    _jobs["install"] = job
    job.start()
    return {"ok": True, "trainer": trainer_status()}


def do_trainer_install_cancel(body):
    job = _jobs["install"]
    if job is not None and job.state == "running":
        job.cancel()
    return {"ok": True}


def do_train(body):
    busy = _job_busy()
    if busy:
        raise BridgeError(f"A {busy} job is already running.")
    trainer = training.find_trainer()
    if not trainer["found"]:
        raise BridgeError("The parametric trainer was not found. Locate it first.")
    missing = [r[k] for c in body["chains"] for r in c["runs"] for k in ("y", "yVal")
               if r.get(k) and not Path(r[k]).expanduser().is_file()]
    if missing:
        names = ", ".join(Path(m).name for m in missing[:4])
        raise BridgeError(
            f"{len(missing)} recording{'s are' if len(missing) != 1 else ' is'} missing "
            f"from the recordings folder ({names}{'...' if len(missing) > 4 else ''})."
        )
    chosen = _osascript_choose("folder", "Choose where to save the trained model",
                               body.get("defaultDir"))
    if not chosen:
        return {"ok": True, "cancelled": True}
    job = training.TrainJob(body, Path(chosen), trainer, _signal_paths())
    _jobs["train"] = job
    job.start()
    return {"ok": True, "cancelled": False, "status": job.status()}


def do_train_stop(body):
    job = _jobs["train"]
    if job is None or job.state != "running":
        return {"ok": True, "stopped": False}
    job.stop()
    return {"ok": True, "stopped": True}


def do_validate(body):
    busy = _job_busy()
    if busy:
        raise BridgeError(f"A {busy} job is already running.")
    trainer = training.find_trainer()
    if not trainer["found"]:
        raise BridgeError("Validation runs the model through the trainer, which was not "
                          "found. Locate it on the training step first.")
    for chain in body["chains"]:
        if not Path(chain["runDir"]).is_dir():
            raise BridgeError(f"The trained model folder for {chain['name']} is gone: "
                              f"{chain['runDir']}")
    job = training.ValidateJob(body, trainer, _bundled_file("validate_model.py"), _signal_paths())
    _jobs["validate"] = job
    job.start()
    return {"ok": True, "status": job.status()}


# ----------------------------------------------------------------------------
# lazy imports: the page must still load and explain itself if audio is broken
# ----------------------------------------------------------------------------

_audio_state = {"ready": False, "error": None, "starting": False}
_audio_lock = threading.Lock()


def _audio():
    """Import the audio stack once, remembering why it failed if it did."""
    if _audio_state["ready"]:
        return _audio_state["mods"]
    if _audio_state["error"]:
        raise BridgeError(_audio_state["error"])
    with _audio_lock:
        if _audio_state["ready"]:
            return _audio_state["mods"]
        if _audio_state["error"]:
            raise BridgeError(_audio_state["error"])
        return _import_audio()


def _import_audio():
    try:
        import numpy as np
        import sounddevice as sd

        from latency import BlipPreamble, measure_delay

        sd.query_devices()          # forces PortAudio to initialise now, not later
        _audio_state["mods"] = {
            "np": np, "sd": sd,
            "BlipPreamble": BlipPreamble, "measure_delay": measure_delay,
        }
        _audio_state["ready"] = True
        return _audio_state["mods"]
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
        _audio_state["error"] = (
            f"Audio stack unavailable: {type(exc).__name__}: {exc}. "
            "The packaged app carries its own copy of numpy and sounddevice; if you "
            "are running the script directly, install them first."
        )
        raise BridgeError(_audio_state["error"]) from exc


# ----------------------------------------------------------------------------
# wav io
# ----------------------------------------------------------------------------

def read_wav_mono(path: Path):
    np = _audio()["np"]
    with wave.open(str(path)) as w:
        rate, width, channels, frames = (
            w.getframerate(), w.getsampwidth(), w.getnchannels(), w.getnframes()
        )
        raw = w.readframes(frames)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        as_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (as_bytes[:, 0] | (as_bytes[:, 1] << 8) | (as_bytes[:, 2] << 16))
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / float(2 ** 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(2 ** 31)
    else:
        raise BridgeError(f"{path.name}: unsupported sample width {width * 8}-bit")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav_24(path: Path, samples, rate: int):
    np = _audio()["np"]
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * (2 ** 23 - 1)).astype(np.int32)
    raw = bytearray()
    for v in ints.tolist():
        raw += int(v).to_bytes(3, "little", signed=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(rate)
        w.writeframes(bytes(raw))


def dbfs(samples) -> float:
    np = _audio()["np"]
    if not len(samples):
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def peak_dbfs(samples) -> float:
    np = _audio()["np"]
    if not len(samples):
        return float("-inf")
    peak = float(np.max(np.abs(samples)))
    return 20.0 * np.log10(peak) if peak > 0 else float("-inf")


# ----------------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------------

def list_devices():
    sd = _audio()["sd"]
    try:
        sd._terminate()
        sd._initialize()
    except Exception:  # noqa: BLE001 - refresh is best-effort
        pass
    host_apis = sd.query_hostapis()
    out = []
    for index, d in enumerate(sd.query_devices()):
        out.append({
            "index": index,
            "name": d["name"],
            "hostApi": host_apis[d["hostapi"]]["name"] if d["hostapi"] < len(host_apis) else "",
            "maxInput": d["max_input_channels"],
            "maxOutput": d["max_output_channels"],
            "defaultSampleRate": d["default_samplerate"],
        })
    return out


# ----------------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------------

def stream_settings(body, sample_rate):
    """
    The buffer the page asked for, as PortAudio wants it.

    A DAW-style buffer size in samples. Asked for alone, PortAudio's "high"
    latency preset opens buffers of about 20,000 samples here, which puts a
    quarter of a second between playing and recording: harmless to training,
    which measures and removes it, but it cuts that much off the end of every
    take. The round trip measures three buffers: 767 samples at 256.
    """
    size = body.get("blockSize")
    if size:
        size = int(size)
        if not 16 <= size <= 8192:
            raise BridgeError(f"Buffer size {size} is out of range (16 to 8192).")
        return {"blocksize": size, "latency": size / float(sample_rate)}
    return {"blocksize": int(body.get("blocksize") or 0),
            "latency": body.get("latency") or "high"}


def play_and_record(playback, *, output_device, input_device, output_channel,
                    sample_rate, blocksize=0, latency="high"):
    """
    Play a mono signal on one output channel and return every input channel.

    The stream is opened at the device's full width and the signal placed at the
    literal channel index, the way a DAW does: a narrow stream would be routed by
    CoreAudio to the default pair instead of the physical output asked for.
    """
    mods = _audio()
    sd, np = mods["sd"], mods["np"]

    out_info = sd.query_devices(output_device)
    in_info = sd.query_devices(input_device)
    out_channels = int(out_info["max_output_channels"])
    in_channels = int(in_info["max_input_channels"])
    if out_channels < 1:
        raise BridgeError(f"{out_info['name']} has no output channels.")
    if in_channels < 1:
        raise BridgeError(f"{in_info['name']} has no input channels.")
    if not 1 <= output_channel <= out_channels:
        raise BridgeError(
            f"Output channel {output_channel} is out of range for "
            f"{out_info['name']} ({out_channels} outputs)."
        )

    frame = np.zeros((len(playback), out_channels), dtype=np.float32)
    frame[:, output_channel - 1] = playback

    # PortAudio can block inside Pa_OpenStream itself - a virtual device with
    # nothing driving it does exactly that - so the open cannot be supervised from
    # this thread. Run the whole call in a worker and give up on it if it never
    # returns. The worker may stay stuck; Python cannot kill a thread, so the
    # device is marked wedged and later captures fail fast instead of piling up.
    if _wedged["device"]:
        raise BridgeError(
            "A previous capture left the audio device unresponsive. Restart the "
            "bridge, and check that the interface is connected and free."
        )

    expected = len(playback) / float(sample_rate)
    budget = OPEN_TIMEOUT_SECONDS + expected + max(5.0, 0.5 * expected)
    box = {}

    def worker():
        try:
            rec = sd.playrec(
                frame,
                samplerate=sample_rate,
                device=(input_device, output_device),
                channels=in_channels,
                dtype="float32",
                blocksize=blocksize,
                latency=latency,
                blocking=True,
            )
            box["recording"] = rec
            box["status"] = sd.get_status()
        except BaseException as exc:  # noqa: BLE001 - reported through the box
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True, name="namtrix-capture")
    thread.start()
    thread.join(budget)
    if thread.is_alive():
        _wedged["device"] = True
        try:
            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        raise BridgeError(
            f"The audio device did not complete a {expected:.1f}s capture within "
            f"{budget:.0f}s. It is most likely a virtual device with nothing driving "
            "it, or an interface held by another application. Nothing was saved; "
            "restart the bridge before trying again."
        )
    if "error" in box:
        exc = box["error"]
        raise BridgeError(f"Capture failed: {type(exc).__name__}: {exc}")

    recording = box["recording"]
    status = box.get("status")
    dropped = bool(status.input_overflow or status.output_underflow) if status else False
    return recording, dropped, in_channels


def build_playback(signal, preamble):
    """Signal with the detection preamble in front, and where the signal starts."""
    np = _audio()["np"]
    pad = np.zeros(int(PREAMBLE_PAD_SECONDS * preamble.sample_rate), dtype=np.float32)
    pre = preamble.render().astype(np.float32)
    return np.concatenate([pre, pad, signal.astype(np.float32)]), len(pre) + len(pad)


def do_capture_cancel(body):
    """
    Stop the take that is playing. The capture thread sees the flag when the
    stream returns and writes nothing, so a cancelled take never reaches disk.
    """
    _capture_cancel.set()
    try:
        _audio()["sd"].stop()
    except Exception:  # noqa: BLE001 - nothing playing is fine
        pass
    return {"ok": True}


def do_delete_takes(body):
    """Delete named takes from the recordings folder: the files of a cancelled run."""
    out_dir = Path(body.get("outDir") or "").expanduser()
    deleted = []
    for name in body.get("files") or []:
        name = str(name)
        if "/" in name or name.startswith(".") or not name.lower().endswith(".wav"):
            raise BridgeError(f"Not a take name: {name}")
        target = (out_dir / name).resolve()
        if target.is_file() and _may_serve(str(target)):
            target.unlink()
            with _written_lock:
                _written_takes.discard(str(target))
            deleted.append(name)
    return {"ok": True, "deleted": deleted}


def do_capture(body):
    _capture_cancel.clear()
    mods = _audio()
    np, BlipPreamble, measure_delay = mods["np"], mods["BlipPreamble"], mods["measure_delay"]

    signal, signal_rate = read_wav_mono(signal_path(body.get("signal") or "v3"))
    sample_rate = int(body.get("sampleRate") or signal_rate)
    if sample_rate != signal_rate:
        raise BridgeError(
            f"Signal file is {signal_rate} Hz but the capture is set to {sample_rate} Hz. "
            "Match them rather than resampling mid-session."
        )

    chains = body["chains"]                      # [{name, channel, file}]
    out_dir = Path(body["outDir"]).expanduser()
    overwrite = bool(body.get("overwrite"))
    targets = {}
    for chain in chains:
        target = out_dir / chain["file"]
        if target.exists() and not overwrite:
            raise BridgeError(
                f"{target.name} already exists. Re-recording this run would overwrite a "
                "take you have; delete it first or confirm the overwrite."
            )
        targets[chain["name"]] = target
    # The other files this run will write, checked now so a clash on the second
    # half of a take is caught before the first half has been played.
    for name in body.get("alsoCheck") or []:
        other = out_dir / name
        if other.exists() and not overwrite:
            raise BridgeError(
                f"{other.name} already exists. Re-recording this run would overwrite a "
                "take you have; delete it first or confirm the overwrite."
            )

    preamble = BlipPreamble(sample_rate=sample_rate)
    playback, signal_start = build_playback(signal, preamble)

    recording, dropped, in_channels = play_and_record(
        playback,
        output_device=int(body["outputDevice"]),
        input_device=int(body["inputDevice"]),
        output_channel=int(body["outputChannel"]),
        sample_rate=sample_rate,
        **stream_settings(body, sample_rate),
    )
    if _capture_cancel.is_set():
        raise BridgeError("Recording cancelled; nothing was saved.")
    if dropped:
        raise BridgeError(
            "The audio stream dropped samples, so this take is not trustworthy. "
            "Nothing was saved. Raise the buffer size (Output, section 2) or close "
            "other audio apps, then record it again."
        )

    results = []
    for chain in chains:
        channel = int(chain["channel"])
        if not 1 <= channel <= in_channels:
            raise BridgeError(
                f"Chain '{chain['name']}' wants input {channel}, but the device has "
                f"{in_channels} inputs."
            )
        captured = recording[:, channel - 1].copy()

        latency_result = measure_delay(captured, preamble)
        delay, delay_note = latency_result.delay, None
        if delay is None:
            delay_note = (
                "No response to the timing blips. Check this chain is actually "
                "patched and passing signal."
            )
        elif latency_result.disagreement_too_high:
            # The two blips read the chain a second apart; when they disagree the
            # average blends two arrival times and means nothing. Say so rather
            # than writing a confident number into the chain settings.
            blips = ", ".join(str(d) for d in latency_result.blip_delays)
            delay_note = (
                f"The two timing blips disagree ({blips} samples), so the delay "
                "cannot be trusted. The stream most likely glitched; re-record."
            )
            delay = None

        # keep the signal file's timebase: drop our preamble, leave the rig's delay in
        body_audio = captured[signal_start:]
        if len(body_audio) < len(signal):
            body_audio = np.concatenate(
                [body_audio, np.zeros(len(signal) - len(body_audio), dtype=np.float32)]
            )
        else:
            body_audio = body_audio[:len(signal)]

        target = targets[chain["name"]]
        # Written beside the old take and swapped in, so re-recording a run replaces
        # it in one step: there is never a moment with neither take on disk.
        partial = target.with_name(f".{target.name}.partial")
        write_wav_24(partial, body_audio, sample_rate)
        os.replace(partial, target)
        with _written_lock:
            _written_takes.add(str(target.resolve()))
        results.append({
            "chain": chain["name"],
            "channel": channel,
            "file": str(target),
            "delay": None if delay is None else int(delay),
            "delayDetected": delay is not None,
            "delayNote": delay_note,
            "rms": dbfs(body_audio),
            "peak": peak_dbfs(body_audio),
            "clipped": bool(np.max(np.abs(body_audio)) >= 0.999) if len(body_audio) else False,
        })

    _allow_dir(out_dir)
    return {
        "ok": True,
        "sampleRate": sample_rate,
        "seconds": len(signal) / sample_rate,
        "results": results,
    }


def _loudest_excerpt(signal, rate, seconds=1.5):
    """
    The part of the reamp signal that will hit the converters hardest: a window
    around its biggest peak and another around its loudest stretch. Levels read
    from these are the levels the real runs will produce, which a sine at some
    arbitrary amplitude cannot promise.
    """
    np = _audio()["np"]
    n = int(seconds * rate)
    if len(signal) <= 2 * n:
        return signal.astype(np.float32)
    peak_at = int(np.argmax(np.abs(signal)))
    energy = np.cumsum(np.square(signal.astype(np.float64)))
    loud_at = int(np.argmax(energy[n:] - energy[:-n]))
    pieces = []
    for start in sorted({max(0, min(len(signal) - n, peak_at - n // 2)), loud_at}):
        piece = signal[start:start + n].astype(np.float32).copy()
        fade = int(0.01 * rate)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        piece[:fade] *= ramp
        piece[-fade:] *= ramp[::-1]
        pieces.append(piece)
    return np.concatenate(pieces)


def gain_advice(peak):
    if peak is None or peak == float("-inf") or peak < LIVE_DBFS:
        return "none"
    if peak >= GAIN_HIGH_DBFS:
        return "decrease"
    if peak < GAIN_LOW_DBFS:
        return "increase"
    return "ok"


def do_test_route(body):
    """
    Play timing blips and the loudest moments of the reamp signal out of the
    chosen output, then report for each input what came back: its level, what
    to do with the gain, and - for the inputs a chain uses - the round-trip
    delay, measured the same way a capture measures it.
    """
    mods = _audio()
    np, BlipPreamble, measure_delay = mods["np"], mods["BlipPreamble"], mods["measure_delay"]
    signal, sample_rate = read_wav_mono(signal_path(body.get("signal") or "v3"))
    excerpt = _loudest_excerpt(signal, sample_rate)
    preamble = BlipPreamble(sample_rate=sample_rate)
    playback, signal_start = build_playback(excerpt, preamble)

    recording, dropped, in_channels = play_and_record(
        playback,
        output_device=int(body["outputDevice"]),
        input_device=int(body["inputDevice"]),
        output_channel=int(body["outputChannel"]),
        sample_rate=sample_rate,
        **stream_settings(body, sample_rate),
    )
    wanted = {int(c) for c in (body.get("channels") or [])}
    inputs = []
    for c in range(in_channels):
        channel = c + 1
        captured = recording[:, c].copy()
        heard = captured[signal_start:]
        peak = peak_dbfs(heard)
        entry = {"channel": channel, "rms": dbfs(heard), "peak": peak,
                 "advice": gain_advice(peak), "delay": None, "delayNote": None}
        if channel in wanted:
            result = measure_delay(captured, preamble)
            if result.delay is None:
                entry["delayNote"] = "No response to the timing blips."
            elif result.disagreement_too_high:
                entry["delayNote"] = "The two timing blips disagree; test again."
            else:
                entry["delay"] = int(result.delay)
        inputs.append(entry)
    if dropped:
        raise BridgeError("The audio stream dropped samples during the test. Raise the "
                          "buffer size (Output, section 2) or close other audio apps.")
    return {
        "ok": True,
        "dropped": dropped,
        "sampleRate": sample_rate,
        "target": {"low": GAIN_LOW_DBFS, "high": GAIN_HIGH_DBFS},
        "inputs": inputs,
    }


# ----------------------------------------------------------------------------
# http
# ----------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    root = Path(".")
    _lock = threading.Lock()          # one capture at a time; the device is not shareable

    def translate_path(self, path):
        rel = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        target = (self.root / rel).resolve() if rel else (self.root / "index.html").resolve()
        if target.is_dir():
            target = target / "index.html"
        try:
            target.relative_to(self.root.resolve())
        except ValueError:
            return str(self.root / "index.html")
        return str(target)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _json(self, payload, status=200):
        data = json.dumps(_json_safe(payload)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/health"):
            # Never block here. Importing numpy and opening PortAudio takes the
            # better part of twenty seconds from a bundled app, and a page that
            # asked during that window would be told there is no audio and would
            # quietly turn itself into Lite for the rest of the session.
            if _audio_state["ready"]:
                return self._json({"ok": True, "version": VERSION, "audio": True,
                                   "starting": False, "error": None})
            if _audio_state["error"]:
                return self._json({"ok": True, "version": VERSION, "audio": False,
                                   "starting": False, "error": _audio_state["error"]})
            return self._json({"ok": True, "version": VERSION, "audio": False,
                               "starting": True, "error": None})
        if self.path.startswith("/api/devices"):
            try:
                return self._json({"ok": True, "devices": list_devices()})
            except BridgeError as exc:
                return self._json({"ok": False, "error": str(exc)}, 503)
        if self.path.startswith("/api/take"):
            return self._serve_take()
        if self.path.startswith("/api/signals"):
            return self._json({"ok": True, "signals": {
                k: (_signals_dir() / f).is_file() for k, f in SIGNAL_FILES.items()}})
        if self.path.startswith("/api/presets"):
            return self._json({"ok": True, "presets": load_presets()})
        if self.path.startswith("/api/trainer"):
            return self._json({"ok": True, "trainer": trainer_status()})
        if self.path.startswith("/api/train/status"):
            job = _jobs["train"]
            return self._json({"ok": True, "status": job.status() if job else None})
        if self.path.startswith("/api/validate/status"):
            job = _jobs["validate"]
            return self._json({"ok": True, "status": job.status() if job else None})
        if self.path.startswith("/api/quit"):
            self._json({"ok": True})
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return
        return super().do_GET()

    def _serve_take(self):
        """
        Hand a recorded take back to the page so it can analyse it.

        The page does the knob check in the browser - one implementation for Lite
        and Full both - and a browser cannot open a path off the filesystem. It
        can fetch one from us, since we are the same origin.

        Only files this bridge wrote in this session are served. The set is
        closed, so a crafted path cannot walk out of it.
        """
        from urllib.parse import parse_qs, urlparse

        wanted = (parse_qs(urlparse(self.path).query).get("file") or [""])[0]
        resolved = str(Path(wanted).expanduser().resolve()) if wanted else ""
        if not resolved or not _may_serve(resolved) or not Path(resolved).is_file():
            return self._json(
                {"ok": False, "error": "Not a take this bridge recorded."}, 404
            )
        data = Path(resolved).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    AUDIO_ROUTES = {"/api/capture": do_capture, "/api/test-route": do_test_route}
    OTHER_ROUTES = {
        "/api/choose-folder": do_choose_folder,
        "/api/allow-folder": do_allow_folder,
        "/api/capture/cancel": do_capture_cancel,
        "/api/delete-takes": do_delete_takes,
        "/api/reveal": do_reveal,
        "/api/presets": do_save_presets,
        "/api/trainer/locate": do_trainer_locate,
        "/api/trainer/install": do_trainer_install,
        "/api/trainer/install/cancel": do_trainer_install_cancel,
        "/api/train/stop": do_train_stop,
        "/api/train": do_train,
        "/api/validate": do_validate,
    }

    def do_POST(self):
        # A JSON content type cannot be sent cross-origin without a preflight,
        # which this server never approves - so another web page open in the
        # same browser cannot drive the interface or the trainer.
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json({"ok": False, "error": "Expected JSON."}, 415)
        path = self.path.split("?", 1)[0]
        audio = path in self.AUDIO_ROUTES
        handler = self.AUDIO_ROUTES.get(path) or self.OTHER_ROUTES.get(path)
        if handler is None:
            return self._json({"ok": False, "error": "No such endpoint"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:  # noqa: BLE001
            return self._json({"ok": False, "error": f"Bad request: {exc}"}, 400)

        if audio and not self._lock.acquire(blocking=False):
            return self._json(
                {"ok": False, "error": "A capture is already running."}, 409
            )
        try:
            return self._json(handler(body))
        except (BridgeError, training.TrainingError) as exc:
            return self._json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
            if audio:
                self._lock.release()


class Server(ThreadingHTTPServer):
    """
    A server that binds immediately.

    ``http.server`` looks up the fully-qualified name of the address it binds
    to, purely to fill in a field used by CGI scripts this has none of. Inside
    the packaged app that lookup finds no resolver willing to answer for
    127.0.0.1 and blocks until DNS gives up - a measured 35 seconds, every
    launch, before the window can open. Outside the bundle the same call returns
    at once, which is what made it look like the audio devices were slow.
    """

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def _bundled_root() -> Path:
    """
    Where index.html lives. Inside a PyInstaller app that is the unpacked bundle;
    running from a checkout it is the folder above this file.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _first_free_port(host: str, start: int, tries: int = 20) -> int:
    """
    A port nobody else is on. A second copy of the app, or anything else already
    holding 8765, should not turn into a crash the user has to interpret.
    """
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    raise BridgeError(
        f"No free port between {start} and {start + tries - 1}. Quit whatever is "
        "using them, or pass --port."
    )


def main():
    ap = argparse.ArgumentParser(description="NAMTRIX capture bridge")
    ap.add_argument("--root", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true",
                    help="Do not open a browser window on start.")
    a = ap.parse_args()

    mimetypes.add_type("application/javascript", ".js")
    Handler.root = Path(a.root).resolve() if a.root else _bundled_root()

    explicit_port = a.port is not None
    port = a.port if explicit_port else _first_free_port(a.host, 8765)

    print(f"NAMTRIX bridge {VERSION}")
    print(f"  serving {Handler.root}")

    server = Server((a.host, port), Handler)

    # Warm the audio stack behind the server rather than in front of it, so the
    # window opens straight away and fills in when the devices are ready.
    def warm():
        try:
            _audio()
            print("  audio ready")
        except BridgeError as exc:
            print(f"  WARNING: {exc}")
            print("  The page will still load; capture stays switched off.")

    _audio_state["starting"] = True
    threading.Thread(target=warm, daemon=True, name="namtrix-audio-warmup").start()
    url = f"http://{a.host}:{port}"
    print(f"\n  open  {url}\n  stop  Ctrl-C\n")

    # Quitting from the Dock sends SIGTERM. Without this the app would vanish
    # from the Dock and keep the port, and the next launch would land on 8766.
    import signal

    def _stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except ValueError:
            pass

    if not a.no_browser:
        # Only once the socket is listening, so the browser cannot beat us to it.
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("\nstopped")


if __name__ == "__main__":
    main()
