"""
The vendored delay measurement must agree with the trainer's, exactly.

The number it returns goes into the trainer's `data.json`, so "close enough" is
not a category here. Run this against a checkout of neural-amp-modeler-parametric
to compare the two directly:

    NAMTRIX_TRAINER=/path/to/neural-amp-modeler-parametric python3 tests/test_latency.py

Without one, the self-consistency checks still run.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))
import latency as mine  # noqa: E402

SR = 48000
rng = np.random.default_rng(7)
failures = []


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def synth(delay, *, noise=1e-5, ring=False, amp=0.9, body_seconds=3.0):
    """A recording of our own preamble, delayed, optionally bandlimited."""
    pre = mine.BlipPreamble(sample_rate=SR, amplitude=amp)
    played = pre.render()
    body = 0.3 * rng.standard_normal(int(body_seconds * SR)).astype(np.float32)
    playback = np.concatenate([played, np.zeros(int(0.25 * SR), np.float32), body])
    rec = np.zeros(len(playback) + delay + SR)
    if ring:
        taps = np.arange(-40, 41)
        kernel = np.sinc(taps * 0.88) * np.hanning(len(taps))
        shaped = np.convolve(playback.astype(np.float64), kernel, "full")[40:40 + len(playback)]
        rec[delay:delay + len(shaped)] = shaped
    else:
        rec[delay:delay + len(playback)] = playback
    rec += noise * rng.standard_normal(len(rec))
    return rec, pre


print("round trips of a known length")
for true_delay in (3, 112, 8_900, 9_400, 12_685, 30_000):
    rec, pre = synth(true_delay)
    got = mine.measure_delay(rec, pre)
    # The safety factor is deliberate: it is always better to under-estimate.
    check(f"delay {true_delay}", got.delay == true_delay - 1,
          f"-> {got.delay} (blips {got.blip_delays})")

print("\na bandlimited arrival reads early, and must")
rec, pre = synth(5_000, ring=True)
got = mine.measure_delay(rec, pre)
check("sinc pre-ringing lands before the peak", got.delay is not None and got.delay < 5_000,
      f"-> {got.delay} for a peak at 5000")

print("\nrefusals")
quiet = 1e-6 * rng.standard_normal(int(6 * SR))
got = mine.measure_delay(quiet, mine.BlipPreamble(sample_rate=SR))
check("no response at all is reported, not guessed", got.delay is None and not got.detected)

rec, pre = synth(500)
rec[pre.blip_locations[1] + 500] = 0.0
rec[pre.blip_locations[1] + 560] = 0.9          # one blip arrives 60 samples late
got = mine.measure_delay(rec, pre)
check("blips that disagree are flagged", got.disagreement_too_high,
      f"blips {got.blip_delays}")
check("...and such a result is not 'ok'", not got.ok)

short = np.zeros(1000)
try:
    mine.measure_delay(short, mine.BlipPreamble(sample_rate=SR))
    check("a recording too short to scan is refused", False)
except ValueError:
    check("a recording too short to scan is refused", True)

trainer = os.environ.get("NAMTRIX_TRAINER")
if trainer and Path(trainer).is_dir():
    print("\nparity with the trainer's own implementation")
    sys.path.insert(0, trainer)
    from nam.capture.latency import BlipPreamble as TheirPreamble
    from nam.capture.latency import measure_delay as their_measure

    for true_delay in (112, 5_000, 12_685):
        rec, pre = synth(true_delay)
        theirs = their_measure(rec, TheirPreamble(sample_rate=SR).as_played())
        ours = mine.measure_delay(rec, pre)
        check(f"delay {true_delay} matches the trainer",
              theirs.delay == ours.delay, f"{theirs.delay} vs {ours.delay}")
else:
    print("\n  (set NAMTRIX_TRAINER to also compare against the trainer's own code)")

print(f"\n{'all passed' if not failures else str(len(failures)) + ' failed'}")
sys.exit(1 if failures else 0)
