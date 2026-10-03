# NAMTRIX Profiler — changes from the ENGL / Marshall evaluation (2026-10-03)

**Status: all items below are implemented in Profiler 0.7.23 / Lite v0.09**, except where noted.
Pending: the channel rule for 6–7 knobs (waits on the Marshall 16-channel result).

Evidence: every model scored on its session's holdouts (settings never trained on). Error follows the
distance from a setting to the nearest training run (rank correlation +0.78, both amps):
<1.2 steps/knob → ESR ~0.03 · 1.2–1.6 → ~0.06 · 1.6–2.2 → ~0.27 · >2.2 (edges) → blow-ups.

## 1. Run-count guidance (Matrix step)
Show recommended runs for the number of knobs; warn below "acceptable"; above 7 knobs suggest fixing knobs.

| knobs | good (≤1.4) | acceptable (≤1.6) | corner runs | holdouts |
|---|---|---|---|---|
| 2 | 6 | 6 | 8 | 8 |
| 3 | 16 | 10 | 8 | 8 |
| 4 | 36 | 22 | 8 | 8 |
| 5 | 82 | 42 | 16 | 10 |
| 6 | 168 | 80 | 16 | 12 |
| 7 | 368 | 160 | 16 | 15 |
| 8 | 784 | 304 | 16 | 20 |
| 9 | 1664 | 592 | 32 | 25 |
| 10 | 3584 | 1056 | 32 | 30 |

Rule of thumb: acceptable ≈ 80 × 1.93^(knobs−6). Keep random LHS for the interior (it beats maximin on median coverage).

## 2. Corners in training
Add a two-level fractional factorial (resolution IV) to the **training** set: 8 runs ≤4 knobs, 16 for 5–8, 32 for 9+.
Holdouts keep their own corners (all-max, alternating); drop any training corner that equals a holdout.

## 3. Volume-type knobs start at 2–3
`avoidZero` knobs (gain/lead/master/volume) currently floor at 1, which is still near-silent. Floor master/volume at 2–3.

## 4. Choose checkpoints by unseen settings
Hold back ~10 loud, spread-out training runs as validation-only (their _val cuts); train on the rest.
Report the **holdout median (non-edge)** as the headline score; show validation only as secondary.

## 5. Network defaults by knob count (provisional)
- Knob mapping: larger (hidden 16) — done in 0.7.22.
- Channels: ≤5 knobs 8; ≥8 knobs 16; 6–7 pending the Marshall 16ch result. 16ch ≈ 2× training time and ≈ 2× plugin CPU.
- lr 0.001 with 16 channels (16ch at 0.002 diverged).
- Epochs ≈ 565,000 / (13 × training runs), clamped 100–400; lr decay per epoch = 0.09^(1/epochs).

## 6. Safety check before export
Render corners and a coarse knob grid with a loud signal; flag settings whose output exceeds 0 dBFS or blows up.

## 7. Smaller items
- Lite Export: `chains.csv` and the ESR script still assume one delay per chain — use per-take delays.
- Recording screen label: "Run 119 · 45 of 110" (ID and position).
- Preset menu must not rename the profile folder once takes exist.
