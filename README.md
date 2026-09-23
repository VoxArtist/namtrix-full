# NAMTRIX Full

Everything [NAMTRIX Lite](https://github.com/VoxArtist/namtrix) does — Latin Hypercube run
sheets, snapped knob values, holdout sets, ESR validation — **plus recording**. It plays the
reamp signal, captures every chain at once through a multi-channel interface, and measures
each chain's round-trip delay from the take itself.

## Why there is a local program

A browser cannot address physical channel 3 of an interface. So a small local program (the
"bridge") owns the audio and serves the page that drives it: same origin, one command,
nothing to connect.

Without the bridge running, `index.html` behaves exactly like Lite — the capture card never
appears. Nothing is lost, you just record in a DAW as before.

## Running it

```bash
python bridge/namtrix_bridge.py
```

Then open **http://127.0.0.1:8765**.

Use the trainer's Python (the one with `sounddevice`, `numpy`, `soundfile`) — on this
machine that is the `nam-venv` from the parametric trainer setup. `start.command` does this
for you; double-click it.

**macOS will ask for microphone permission the first time**, because recording counts as
mic access. Approve Terminal in System Settings → Privacy & Security → Microphone. Skipping
it yields silent recordings rather than an error, which is a confusing way to lose an hour.

## A run

1. Pick the output device and the channel feeding the reamp box.
2. Pick the input device, and which input carries each capture chain.
3. **Test route** plays a short tone and shows the level on every input, so you can confirm
   the routing before committing to fifty runs.
4. **Record run** plays the signal, records every chain in one pass, writes one file per
   chain using your naming pattern, measures each chain's delay and writes it into the
   chain settings, ticks the run off and shows the next knob positions full screen.

## What it saves

One file per chain per run, in the signal file's own timebase: the detection preamble the
bridge adds is stripped, **the rig's delay is left in**. That is what a DAW-recorded take
looks like, and what the trainer expects alongside an explicit `delay`. Pre-aligning here
would risk the correction being applied twice — which cost us a full training run once,
ESR 0.166 instead of 0.012, from an 8-sample over-correction.

## Refusals

The bridge would rather fail than hand you a bad take:

- **A device that blocks on open** (a virtual bridge with nothing driving it) is abandoned
  after a timeout with a clear message. PortAudio can hang inside `Pa_OpenStream` itself, so
  the whole call is supervised from another thread; once a device wedges, later captures
  fail immediately instead of piling up.
- **Dropped samples** void the take. Nothing is written.
- **An existing file** is never overwritten without being asked.
- **Sample-rate mismatch** between the signal file and the capture stops the run rather than
  resampling behind your back.

## Notes carried over from Lite

- **Delay direction is asymmetric.** Under-estimating is harmless; over-estimating shifts
  the target ahead of the input and a causal network cannot fit it.
- **Each chain has its own delay** — a miked cab lags a load-box feed by the extra air path.
- **ESR has a floor** set by the amp's own hiss.
- **Silent runs are scored on level, not ESR.** Every control at minimum makes the amp
  silent, so the model rightly predicts silence and the residual is unpredictable hiss — a
  ratio near 1.0 however good the model is. Those rows are checked for "is the model silent
  too" and kept out of the ESR average. Training is unaffected: the trainer's loss is MSE,
  not ESR, so a silent target contributes almost no gradient.
