# NAMTRIX Full

Everything [NAMTRIX Lite](https://github.com/VoxArtist/namtrix) does — Latin Hypercube run
sheets, snapped knob values, holdout sets — **plus the rest of the job, without a terminal or
a DAW**: it plays the reamp signal (both signals ship inside the app), records every chain at
once through a multi-channel interface, measures each chain's latency, checks the takes for
knob mistakes, trains the model from this session's recordings, and scores it against the
holdout runs.

## Why there is a local program

A browser cannot address physical channel 3 of an interface. So a small local program (the
"bridge") owns the audio and serves the page that drives it: same origin, one command,
nothing to connect.

Opened as a plain file without the bridge, the page says capture is unavailable; use Lite for
the DAW-based workflow.

## Installing it

Download **NAMTRIX-Full-macOS.zip** from the
[latest release](https://github.com/VoxArtist/namtrix-full/releases), unzip it, and drag
**NAMTRIX Full.app** to Applications. Nothing else to install — the app carries its own
Python, numpy and PortAudio.

Double-click it and your browser opens on the tool. Quit it from the Dock when you are done.

Two things to expect the first time:

- **Right-click → Open, not a double-click.** The app is signed, but signing it in a way
  that satisfies Gatekeeper outright needs a paid Apple Developer account. Right-click →
  Open once and macOS remembers. (A plain double-click is refused outright, silently, so
  this step is not optional. The zip carries the same note beside the app.)

  **If right-click → Open still refuses**, some macOS versions need the longer route:
  1. Try to open the app once (either way) — it will refuse, and that refusal is what
     unlocks the next step.
  2. **System Settings → Privacy & Security**, scroll down to the Security section near
     the bottom.
  3. You'll see *"NAMTRIX Full" was blocked to protect your Mac* with an **Open Anyway**
     button next to it. Click it.
  4. Confirm with your password or Touch ID if asked.
  5. Open the app again — one more confirmation dialog, then it runs from then on.

  This isn't a NAMTRIX-specific problem — every unsigned or ad-hoc-signed app hits it,
  and it's the same $99/year Apple Developer account that would remove it entirely.
- **macOS asks for microphone permission**, because it counts any audio input as a
  microphone. Say yes. Saying no yields *silent recordings rather than an error*, which is
  a confusing way to lose an hour.

If you extract the zip from a terminal, use `ditto -x -k` rather than `unzip`: `unzip` does
not preserve everything the signature covers, and macOS then refuses to run the app with
"a sealed resource is missing or invalid". Finder's own double-click does the right thing.

### From source

```bash
python3 -m pip install numpy sounddevice
python3 bridge/namtrix_bridge.py
```

Then open **http://127.0.0.1:8765**. Recording needs nothing else: the delay measurement is
vendored in `bridge/latency.py`, so no trainer checkout and no torch.

The reamp signals in `signals/` are NAM's standard v3.0.0 `input.wav` and the parametric
trainer's `inputTrunc.wav` / `validation.wav` cut of it (training-inputs-v1 release), both
from MIT-licensed projects.

### Building the app

`build/build_app.sh` makes its own environment, builds the icon, bundles, signs and zips.

## A session

1. **Parameter space.** Pick a gear preset or enter the controls. *Save as preset* adds your
   gear to the list under *My presets* (kept by the app, so it survives a restart). *Skip 0*
   keeps a control off 0 in the run sheet, since a silent take costs a full run and teaches
   the model nothing; it starts ticked for any control named Gain, Lead, Master, Volume, Level or Output.
2. **OED matrix.** Choose run counts and the reamp signal — *Long* (NAM's standard 190 s
   input) or *Short* (38 s + a 7 s validation cut) — and the time estimate follows.
3. **Reamp & record**, in order:
   1. *Capture chains* — the device alone, or an amp and cab with how many mics (each mic is
      a chain), optionally plus the head's direct output.
   2. *Output* — where the reamp signal goes, and the *buffer size* (64–2048 samples, 256 by
      default), as in a DAW. The delay between playing and recording is three buffers: 767
      samples (16 ms) at 256. Raise it if a take reports dropped samples.
   3. *Inputs* — one per chain.
   4. *Test the route* — plays timing blips and the loudest moments of the reamp signal
      through the chosen output. Each chain gets its peak level, a gain verdict (*increase*,
      *decrease* or *good*, against a −18 to −3 dBFS target) and its latency, which is saved
      to the chain and used for training. Test with the gear at its loudest run-sheet setting.
   5. *File names* — with examples for a training and a holdout (verification) run.
   6. *Recordings folder* — chosen in a normal macOS folder dialog.
   7. *Record* — each run shows its knob settings full screen; dial them, press Record. A
      take that clips, or comes back silent, is not ticked off.
   8. *Check the recordings* — see below.
4. **Train.** Pick epochs (400 by default) and press Start. A folder dialog asks where the
   model goes; the app writes the dataset from this session's takes — each with its knob
   settings and its own measured latency — and runs the trainer, showing progress, best ESR
   so far and time left. The model name defaults to the gear name.
5. **Validate.** One button. Every holdout run is played through the trained model at its
   knob settings and compared with the real recording; the average ESR comes back with what
   it means.

## What it saves

One file per chain per run, in the signal file's own timebase: the detection preamble the
bridge adds is stripped, **the rig's delay is left in**. That is what a DAW-recorded take
looks like, and what the trainer expects alongside an explicit `delay`. Pre-aligning here
would risk the correction being applied twice — which cost us a full training run once,
ESR 0.166 instead of 0.012, from an 8-sample over-correction.

## Knob check

Once every run is recorded, the app offers to check the takes. It reads each one back from
the recordings folder and compares it with all the others. Every run plays the same signal,
so takes differ only by where the knobs were — which makes a run checkable against the rest.
Runs that disagree with what the others predict are listed; **Re-record these runs** queues
just those, at their own settings, and each new take replaces the old file. When the queue is
done the check runs again, until it comes back clean or you choose **Ignore**.

It reports how big an error it can actually see, per knob, and that number is worth reading
before the findings are. On our own 1987X it lands between 6 and 17 knob units — the amp was
driven into saturation, so its controls move the sound less than one take differs from the
next, and nothing can recover what the amp never expressed. On gear whose knobs do more, it
sees about one unit. Either way it says which it is instead of implying the flattering one.

A suggestion is the closest single-knob explanation, not a verdict. An amp can do things no
knob accounts for — our run 27 had Volume I at zero, which switches off the High Treble
channel entirely, and the check reached for Presence because that is the nearest thing in
its vocabulary. Look at the run before changing anything.

## The trainer

Training and validation run the
[parametric NAM trainer](https://github.com/phillipmself/neural-amp-modeler-parametric)
(PyTorch plus the trainer, about 1 GB). It is not inside the app — every update would
re-download that gigabyte — but the app installs it for you: from the OED Matrix step on, if
it is missing, a card offers **Download and install the trainer**. It installs in the
background into `~/Library/Application Support/NAMTRIX/trainer` while you carry on with the
matrix and the recordings; the header shows its progress. Nothing outside that folder is
touched, and deleting the folder removes it.

Under the hood the app carries [uv](https://github.com/astral-sh/uv), which fetches its own
Python 3.12 and the exact package set in `bridge/trainer-requirements.txt` — the one that has
trained real models here — with the trainer pinned to a fixed commit. Apple silicon only,
like the app itself. Anyone who already has a trainer can point the app at it instead
(*I already have one — locate it*); recording and the knob check need none of this.

Training runs in the background with the Mac kept awake; leave the app open. *Stop early*
still exports the best model so far. Each chain gets its own folder with the configs, a copy
of the DI, the log and the trainer's timestamped run folder holding
`<name>_parametric.nam` (for the NAM Parametric Plugin) and `<name>.nam`.

Validation scores the checkpoint the trainer exported and writes each holdout prediction to
`holdout_renders/` in the run folder, so the numbers can be checked by ear.

## Refusals

The bridge would rather fail than hand you a bad take:

- **A device that blocks on open** (a virtual bridge with nothing driving it) is abandoned
  after a timeout with a clear message. PortAudio can hang inside `Pa_OpenStream` itself, so
  the whole call is supervised from another thread; once a device wedges, later captures
  fail immediately instead of piling up.
- **Dropped samples** void the take. Nothing is written.
- **An existing file** is never overwritten without being asked — and every file a run will
  write is checked before anything plays, so a clash cannot waste half a take.
- **A re-recorded take** is written beside the old one and swapped in, so there is never a
  moment with neither on disk.

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
