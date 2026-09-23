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

## Installing it

Download **NAMTRIX-Full-macOS.zip** from the
[latest release](https://github.com/VoxArtist/namtrix-full/releases), unzip it, and drag
**NAMTRIX Full.app** to Applications. Nothing else to install — the app carries its own
Python, numpy and PortAudio.

Double-click it and your browser opens on the tool. Quit it from the Dock when you are done.

Two things to expect the first time:

- **Right-click → Open, not a double-click.** The app is signed, but signing it in a way
  that satisfies Gatekeeper outright needs a paid Apple Developer account. Right-click →
  Open once and macOS remembers.
- **macOS asks for microphone permission**, because it counts any audio input as a
  microphone. Say yes. Saying no yields *silent recordings rather than an error*, which is
  a confusing way to lose an hour.

### From source

```bash
python3 -m pip install numpy sounddevice
python3 bridge/namtrix_bridge.py
```

Then open **http://127.0.0.1:8765**. Nothing else is needed: the delay measurement is
vendored in `bridge/latency.py`, so no trainer checkout and no torch.

### Building the app

`build/build_app.sh` makes its own environment, builds the icon, bundles, signs and zips.

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

## Knob check

After each take, the page reads the recording back and compares it with every run so far.
Every run plays the same signal, so takes differ only by where the knobs were — which makes
a run checkable against the others. If one disagrees with what the rest predict, the run
sheet is a likelier culprit than the amp, and you hear about it **while the amp is still set
that way**, when re-recording costs one take instead of a training run.

It reports how big an error it can actually see, per knob, and that number is worth reading
before the findings are. On our own 1987X it lands between 6 and 17 knob units — the amp was
driven into saturation, so its controls move the sound less than one take differs from the
next, and nothing can recover what the amp never expressed. On gear whose knobs do more, it
sees about one unit. Either way it says which it is instead of implying the flattering one.

A suggestion is the closest single-knob explanation, not a verdict. An amp can do things no
knob accounts for — our run 27 had Volume I at zero, which switches off the High Treble
channel entirely, and the check reached for Presence because that is the nearest thing in
its vocabulary. Look at the run before changing anything.

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
