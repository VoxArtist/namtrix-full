#!/bin/zsh
# Re-derive tests/knobcheck.js from the copy inlined in index.html, so a change
# made in the page cannot silently diverge from what the tests cover.
set -e
cd "${0:A:h}/.."
python3 - <<'PY'
from pathlib import Path
page = Path("index.html").read_text()
start = page.index("/* ========================= KNOB CHECK =========================")
end = page.index("/* ------------------------- knob check: the UI side")
body = page[start:end].rstrip() + "\n\nif(typeof module !== \"undefined\")\n  module.exports = {kcFeatures, kcAnalyse, kcNormalise, kcFit, kcChooseShape, KC_FEATURES, kcFFT};\n"
Path("tests/knobcheck.js").write_text(body)
print(f"extracted {len(body):,} chars from index.html")
PY
node --check tests/knobcheck.js && echo "syntax ok"
