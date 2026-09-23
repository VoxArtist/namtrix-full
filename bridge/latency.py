"""
Round-trip delay measurement, on numpy alone.

This is the parametric trainer's own blip calibration, reimplemented here so the
bridge can be handed to someone who has no trainer checkout and no torch. It is a
deliberate copy, not an improvement: the delay it returns goes into the trainer's
`data.json`, so it has to be the number the trainer would have arrived at. Same
preamble geometry, same thresholds, same safety factor, same disagreement test.

What it measures is the *first arrival of energy*, not the loudest point. Those are
not the same thing and the difference is large. A converter round trip is
bandlimited, so a played impulse comes back as a sinc: symmetric ringing spreading
both ways from the peak. On a real loopback here, the peak landed 35 samples after
the first sample above the noise floor. Triggering on that first sample and then
stepping back one more is not sloppiness - it is the only safe direction. Align to
the peak instead and the target sits ahead of the input, and a causal network
cannot fit what it has not yet been given. Over-correcting by 8 samples once cost
us a whole training run: ESR 0.166 where 0.012 was available.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional

import numpy as _np

# Preamble layout in seconds. Mirrors the v2 NAM input file's proportions so the
# calibration sees the geometry it was written for.
_BLIP_SECTION_SECONDS = 2.0
_BLIP_TIMES_SECONDS = (0.5, 1.5)
_GAP_SECONDS = 0.25
DEFAULT_BLIP_AMPLITUDE = 0.9

# The fine calibrator only looks from _LOOKAHEAD before to _LOOKBACK after each
# expected blip. A buffered virtual route can sit further out than that - our own
# audio-bridge loopback measured 12,685 samples - so a coarse pass runs first.
_LOOKAHEAD = 1_000
_LOOKBACK = 10_000

_ABS_THRESHOLD = 0.0003
_REL_THRESHOLD = 0.001
_SAFETY_FACTOR = 1
_MAX_DISAGREEMENT = 20


@_dataclass(frozen=True)
class BlipPreamble:
    """What to play in front of the signal so the return can be timed."""

    sample_rate: int
    amplitude: float = DEFAULT_BLIP_AMPLITUDE

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive; got {self.sample_rate}")
        if not 0.0 < self.amplitude <= 1.0:
            raise ValueError(f"amplitude must be within (0, 1]; got {self.amplitude}")

    @property
    def blip_locations(self) -> tuple[int, ...]:
        return tuple(int(t * self.sample_rate) for t in _BLIP_TIMES_SECONDS)

    @property
    def n_samples(self) -> int:
        """Whole preamble including its trailing gap; the signal starts here."""
        return int((_BLIP_SECTION_SECONDS + _GAP_SECONDS) * self.sample_rate)

    @property
    def noise_interval(self) -> tuple[int, int]:
        """
        A window that is silent in any preamble worth the name: the middle of the
        run-up to the first blip. Its peak sets the trigger threshold.
        """
        first = self.blip_locations[0]
        return (first // 2, first * 3 // 4)

    def render(self) -> _np.ndarray:
        playback = _np.zeros(self.n_samples, dtype=_np.float32)
        for location in self.blip_locations:
            playback[location] = self.amplitude
        return playback


@_dataclass(frozen=True)
class LatencyResult:
    delay: _Optional[int]
    detected: bool
    disagreement_too_high: bool
    safety_factor: int
    # What each blip said before averaging. Kept so a capture that trips the
    # disagreement check can show its numbers: one blip arriving late is a
    # specific fault (a stream glitch during the preamble), and seeing the pair
    # is what makes it recognisable.
    blip_delays: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return self.detected and not self.disagreement_too_high


def _coarse_bulk_delay(recording: _np.ndarray, preamble: BlipPreamble) -> int:
    """
    Find a large fixed latency that would otherwise sit outside the fine
    calibrator's window. Returns 0 for normal low-latency routes, where the fine
    pass handles it alone and this must stay out of the way.

    The search is confined to the silent blip section - never the signal region,
    which is as loud as the blips - and only as far as still leaves the last blip
    inside the fine window.
    """
    locations = preamble.blip_locations
    max_shift = min(preamble.n_samples - locations[-1], len(recording) - locations[-1])
    if max_shift <= _LOOKBACK:
        return 0

    score = _np.zeros(max_shift, dtype=_np.float64)
    for location in locations:
        score += _np.abs(recording[location : location + max_shift])

    shift = int(_np.argmax(score))
    if shift <= _LOOKBACK - _LOOKAHEAD:
        return 0  # the fine pass can already see it; don't interfere
    peak = float(score[shift])
    baseline = float(_np.median(score))
    # Demand that the winner clearly stands out. Otherwise it is noise, and
    # reporting "not detected" is better than reporting a number made of hiss.
    if peak < 0.02 * len(locations) or peak < 5.0 * (baseline + 1e-9):
        return 0
    return shift


def measure_delay(recording: _np.ndarray, preamble: BlipPreamble) -> LatencyResult:
    """
    Round-trip delay in samples, from a recording that starts when playback did.
    """
    recording = _np.asarray(recording, dtype=_np.float64).squeeze()
    if recording.ndim != 1:
        raise ValueError(f"Expected a single-channel recording; got {recording.shape}")
    minimum = preamble.blip_locations[-1] + _LOOKBACK
    if len(recording) < minimum:
        raise ValueError(
            f"Recording too short to scan for blips: {len(recording)} < {minimum}"
        )

    coarse = _coarse_bulk_delay(recording, preamble)
    if coarse > 0:
        scan = _np.zeros_like(recording)
        scan[: len(recording) - coarse] = recording[coarse:]
    else:
        scan = recording

    section = scan[: preamble.n_samples]
    noise_lo, noise_hi = preamble.noise_interval
    background = float(_np.max(_np.abs(section[noise_lo:noise_hi]))) if noise_hi > noise_lo else 0.0
    threshold = max(background + _ABS_THRESHOLD, (1.0 + _REL_THRESHOLD) * background)

    windows = []
    for location in preamble.blip_locations:
        start, stop = location - _LOOKAHEAD, location + _LOOKBACK
        if start < 0 or stop > len(section):
            raise ValueError("Preamble does not fit inside the recording")
        windows.append(section[start:stop])

    def first_crossing(window) -> _Optional[int]:
        """Delay of the first sample to trip the trigger, relative to its blip."""
        triggered = _np.flatnonzero(_np.abs(window) > threshold)
        return None if not len(triggered) else int(triggered[0]) - _LOOKAHEAD

    # Each blip alone, which is what the disagreement check compares. The
    # recommended figure still comes from the averaged windows - averaging is the
    # noise-robust estimate - but averaging *first* leaves a single number, and the
    # range of one number is always zero, so the check could never fire.
    per_blip = [d for d in map(first_crossing, windows) if d is not None]
    averaged = first_crossing(_np.mean(_np.stack(windows), axis=0))

    if averaged is None:
        return LatencyResult(None, False, False, _SAFETY_FACTOR, ())

    disagreement = len(per_blip) > 1 and (max(per_blip) - min(per_blip)) >= _MAX_DISAGREEMENT
    return LatencyResult(
        delay=averaged - _SAFETY_FACTOR + coarse,
        detected=True,
        disagreement_too_high=disagreement,
        safety_factor=_SAFETY_FACTOR,
        blip_delays=tuple(d + coarse for d in per_blip),
    )
