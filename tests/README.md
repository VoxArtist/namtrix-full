# Knob-check tests

`knobcheck.js` here is the same engine that is inlined into `index.html`, kept extractable
so it can be tested outside a browser.

```bash
node tests/test_knobcheck.js
```

The suite builds synthetic amps whose true response is known, corrupts one label, and
checks that the corrupted run is flagged, that the right knob is named, and — the part that
matters most — that correctly-labelled runs are left alone.

`extract.sh` re-derives this copy from `index.html`, so the two cannot drift apart
unnoticed.
