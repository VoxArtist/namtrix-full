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

VERSION = "0.2.0"
PREAMBLE_PAD_SECONDS = 0.25   # silence after the preamble before the signal starts
OPEN_TIMEOUT_SECONDS = 10.0   # PortAudio can block indefinitely opening a dead device

# Set when a capture thread never returned: the device layer cannot be trusted again
# in this process, so say so rather than hanging every later request too.
_wedged = {"device": False}

# Takes written during this run, which are the only files /api/take will serve.
_written_takes: set[str] = set()
_written_lock = threading.Lock()


class BridgeError(RuntimeError):
    """Something the user can act on; reported as a clean message, not a stack."""


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


def do_capture(body):
    mods = _audio()
    np, BlipPreamble, measure_delay = mods["np"], mods["BlipPreamble"], mods["measure_delay"]

    signal_path = Path(body["signalPath"]).expanduser()
    if not signal_path.is_file():
        raise BridgeError(f"Signal file not found: {signal_path}")
    signal, signal_rate = read_wav_mono(signal_path)
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

    preamble = BlipPreamble(sample_rate=sample_rate)
    playback, signal_start = build_playback(signal, preamble)

    recording, dropped, in_channels = play_and_record(
        playback,
        output_device=int(body["outputDevice"]),
        input_device=int(body["inputDevice"]),
        output_channel=int(body["outputChannel"]),
        sample_rate=sample_rate,
        blocksize=int(body.get("blocksize") or 0),
        latency=body.get("latency") or "high",
    )
    if dropped:
        raise BridgeError(
            "The audio stream dropped samples, so this take is not trustworthy. "
            "Nothing was saved. Close other audio apps or raise the buffer size."
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
        write_wav_24(target, body_audio, sample_rate)
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

    return {
        "ok": True,
        "sampleRate": sample_rate,
        "seconds": len(signal) / sample_rate,
        "results": results,
    }


def do_test_route(body):
    """Play a short tone and report what each input heard, before a real run."""
    mods = _audio()
    np = mods["np"]
    sample_rate = int(body.get("sampleRate") or 48000)
    seconds = float(body.get("seconds") or 1.5)
    freq = float(body.get("frequency") or 440.0)
    amplitude = float(body.get("amplitude") or 0.2)

    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = min(int(0.01 * sample_rate), len(tone) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]

    recording, dropped, in_channels = play_and_record(
        tone,
        output_device=int(body["outputDevice"]),
        input_device=int(body["inputDevice"]),
        output_channel=int(body["outputChannel"]),
        sample_rate=sample_rate,
        latency=body.get("latency") or "high",
    )
    return {
        "ok": True,
        "dropped": dropped,
        "inputs": [
            {"channel": c + 1, "rms": dbfs(recording[:, c]), "peak": peak_dbfs(recording[:, c])}
            for c in range(in_channels)
        ],
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
        data = json.dumps(payload).encode()
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
        with _written_lock:
            allowed = resolved in _written_takes
        if not allowed or not Path(resolved).is_file():
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

    def do_POST(self):
        routes = {"/api/capture": do_capture, "/api/test-route": do_test_route}
        route = next((r for r in routes if self.path.startswith(r)), None)
        if route is None:
            return self._json({"ok": False, "error": "No such endpoint"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:  # noqa: BLE001
            return self._json({"ok": False, "error": f"Bad request: {exc}"}, 400)

        if not self._lock.acquire(blocking=False):
            return self._json(
                {"ok": False, "error": "A capture is already running."}, 409
            )
        try:
            return self._json(routes[route](body))
        except BridgeError as exc:
            return self._json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
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
