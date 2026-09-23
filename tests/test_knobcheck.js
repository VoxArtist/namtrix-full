const {kcAnalyse, kcFeatures, KC_FEATURES} = require("./knobcheck.js");
let pass = 0, fail = 0;
const ok  = (c,m)=>{ c ? (pass++, console.log("  ok   "+m)) : (fail++, console.log("  FAIL "+m)); };

const KNOBS = [
  {name:"Drive",  min:0, max:10, step:0.5},
  {name:"Tone",   min:0, max:10, step:0.5},
  {name:"Level",  min:0, max:10, step:0.5},
];

// A stand-in amp: each feature a smooth, monotone function of the knobs, plus
// take-to-take noise of the size a real rig has (we measured ~0.3 dB).
// A stand-in amp. Each band responds smoothly to the knobs, the way a tone
// stack does, plus take-to-take noise of the size a real rig has.
const NB = KC_FEATURES.filter(f=>f.band).length;
function respond(v, rng, noise){
  const [drive, tone, level] = v;
  const n = () => (noise||0) * (rng()*2-1);
  const out = {
    silent:false,
    rms:       -24 + 1.30*level + 0.22*drive + n(),
    crest:      14 -  0.75*drive + n(),
    dynamics:   18 -  1.10*drive + n(),
  };
  // Tone tilts the spectrum about a pivot; Drive adds a little top. One knob
  // therefore moves several bands together, which is what the check keys on.
  for(let i=0;i<NB;i++){
    const centre = (i - (NB-1)/2)/((NB-1)/2);          // -1 low .. +1 high
    out[`b${i}`] = -10 + 3.0*Math.cos(i*0.9)
                 + 0.85*tone*centre
                 + 0.20*drive*Math.max(0, centre)
                 + n();
  }
  return out;
}

function mulberry(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0;
  let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t;
  return ((t^t>>>14)>>>0)/4294967296; }; }

// A space-filling set, the way the matrix generator makes one.
function makeRuns(n, seed, noise){
  const rng = mulberry(seed), runs = [];
  const perm = k => { const a=[...Array(n).keys()]; for(let i=n-1;i>0;i--){const j=Math.floor(rng()*(i+1)); [a[i],a[j]]=[a[j],a[i]];} return a; };
  const cols = KNOBS.map(()=>perm());
  for(let i=0;i<n;i++){
    const values = KNOBS.map((k,ki)=>{
      const u = (cols[ki][i] + rng())/n;
      const raw = k.min + u*(k.max-k.min);
      return Math.round(raw/k.step)*k.step;
    });
    runs.push({run:i+1, values, f:respond(values, rng, noise)});
  }
  return runs;
}

console.log("\n1. A clean set raises no alarms");
{
  let falseAlarms = 0, total = 0;
  for(let s=1; s<=12; s++){
    const runs = makeRuns(40, s, 0.3);
    const r = kcAnalyse(runs, KNOBS);
    total += runs.length;
    falseAlarms += r.findings.filter(f=>f.status!=="sparse").length;
  }
  console.log(`   ${falseAlarms} flags across ${total} good runs (12 sets)`);
  ok(falseAlarms === 0, "no false alarms on 480 correctly-labelled runs");
}

console.log("\n2. A mis-set knob is caught, and named");
{
  const cases = [
    {ki:0, name:"Drive", wrongBy: 4.0},
    {ki:1, name:"Tone",  wrongBy: 5.0},
    {ki:2, name:"Level", wrongBy: 3.0},
    {ki:1, name:"Tone",  wrongBy:-4.5},
    {ki:2, name:"Level", wrongBy:-3.5},
  ];
  let caught=0, named=0, valued=0, n=0, skipped=0, hedged=0;
  for(let s=20; s<32; s++){
    for(const c of cases){
      const runs = makeRuns(40, s, 0.3);
      const victim = runs[17];
      const actual = victim.values.slice();
      const k = KNOBS[c.ki];
      actual[c.ki] = Math.max(k.min, Math.min(k.max, actual[c.ki] + c.wrongBy));
      // The declared position may already sit near an end stop, in which case
      // clamping turns a big error into a small one. That is a different test -
      // test 3 owns it - so it does not belong in this count.
      if(Math.abs(actual[c.ki] - victim.values[c.ki]) < 2.0){ skipped++; continue; }
      n++;
      victim.f = respond(actual, mulberry(s*97), 0.3);
      const r = kcAnalyse(runs, KNOBS);
      const found = r.findings.find(f=>f.run===victim.run && f.status!=="sparse");
      if(found) caught++;
      if(found && found.status==="uncertain") hedged++;
      if(found && found.suggestion && found.suggestion.knob === c.name) named++;
      if(found && found.suggestion && Math.abs(found.suggestion.value - actual[c.ki]) <= 1.0) valued++;
    }
  }
  console.log(`   caught ${caught}/${n}   right knob named ${named}/${n}   value within 1.0 ${valued}/${n}`
              + `   (${skipped} skipped: clamping left them under 2 knob units`
              + `${hedged ? `; ${hedged} hedged as extrapolation` : ""})`);
  ok(caught === n, "catches every mis-set knob of 2 units or more");
  ok(named === caught, "names the right knob every time it catches one");
  ok(valued >= 0.9*caught, "and gets the value within one knob unit");
}

console.log("\n2b. Sensitivity: how big an error does it take?");
{
  const sizes = [0.5, 1, 1.5, 2, 3, 4];
  const rows = [];
  for(const size of sizes){
    let caught = 0, n = 0;
    for(let s=60; s<84; s++){
      for(const ki of [0,1,2]){
        const runs = makeRuns(40, s, 0.3);
        const v = runs[(s*7+ki) % 40];
        const k = KNOBS[ki];
        const dir = (s % 2) ? 1 : -1;
        const actual = v.values.slice();
        const want = actual[ki] + dir*size;
        if(want < k.min || want > k.max) continue;   // no clamping, no confound
        actual[ki] = want;
        n++;
        v.f = respond(actual, mulberry(s*131+ki), 0.3);
        const r = kcAnalyse(runs, KNOBS);
        if(r.findings.some(f=>f.run===v.run && f.status!=="sparse")) caught++;
      }
    }
    rows.push({size, rate: n ? caught/n : 0, n});
  }
  rows.forEach(r=>console.log(`   ${String(r.size).padStart(4)} knob units off:  ${(100*r.rate).toFixed(0).padStart(3)}%  (n=${r.n})`));
  const at2 = rows.find(r=>r.size===2).rate, at05 = rows.find(r=>r.size===0.5).rate;
  ok(at2 >= 0.95, "2 units off is caught essentially always");
  ok(at05 <= 0.15, "half a detent stays under the threshold");
  ok(rows.every((r,i)=> i===0 || r.rate >= rows[i-1].rate - 0.05), "detection rises with the size of the error");
}

console.log("\n3. A small error is NOT flagged (it should not cry wolf)");
{
  let flagged = 0;
  for(let s=40; s<52; s++){
    const runs = makeRuns(40, s, 0.3);
    const v = runs[11];
    const actual = v.values.slice();
    actual[1] = Math.min(KNOBS[1].max, actual[1] + 0.5);   // one detent out
    v.f = respond(actual, mulberry(s*31), 0.3);
    if(kcAnalyse(runs, KNOBS).findings.some(f=>f.run===v.run && f.status!=="sparse")) flagged++;
  }
  console.log(`   ${flagged}/12 single-detent errors flagged`);
  ok(flagged <= 2, "a one-detent slip stays below the alarm threshold");
}

console.log("\n4. Gating");
{
  const few = makeRuns(8, 5, 0.3);
  const r = kcAnalyse(few, KNOBS);
  ok(r.ready === false, "refuses to judge too few runs");
  ok(/analysed runs/.test(r.reason) && /\d+/.test(r.reason), "says why, with a number");

  const silentAll = makeRuns(14, 5, 0.3).map(r=>({...r, f:{...r.f, silent:true}}));
  ok(kcAnalyse(silentAll, KNOBS).ready === false, "silent runs do not count towards the population");
}

console.log("\n5. Extrapolation is reported as uncertain, not as a verdict");
{
  // Every other run keeps Level between 0 and 4; this one sits at 10. Nothing
  // in the set says what the amp does up there, so the model can only
  // extrapolate - and an extrapolation must not convict.
  const rng = mulberry(9);
  const runs = [];
  for(let i=0;i<30;i++){
    const v = [Math.round(rng()*20)/2, Math.round(rng()*20)/2, Math.round(rng()*8)/2];
    runs.push({run:i+1, values:v, f:respond(v, rng, 0.3)});
  }
  const odd = [5, 5, 10];
  const actual = [5, 5, 10];
  runs.push({run:99, values:odd, f:respond(actual, rng, 0.3)});
  const r = kcAnalyse(runs, KNOBS);
  const f = r.findings.find(x=>x.run===99);
  ok(!f || f.status === "uncertain",
     f ? `the extrapolated run is marked "${f.status}"` : "the extrapolated run is not convicted");
  if(f) ok(/extrapolate/.test(f.note||""), "and says why it cannot be trusted");
  else pass++, console.log("  ok   (not flagged at all, which is also acceptable)");

  // Meanwhile a run in the middle of the pack is judged normally.
  const mid = makeRuns(40, 5, 0.3);
  const v = mid[20]; const bad = v.values.slice();
  bad[1] = Math.max(0, Math.min(10, bad[1] + 4));
  if(Math.abs(bad[1]-v.values[1]) >= 2){
    v.f = respond(bad, mulberry(77), 0.3);
    const got = kcAnalyse(mid, KNOBS).findings.find(x=>x.run===v.run);
    ok(!!got && got.status !== "uncertain" && got.status !== "sparse",
       "a well-surrounded run is still judged outright");
  } else { pass++; console.log("  ok   (clamped, skipped)"); }
}

console.log("\n6. Feature extraction on real waveforms");
{
  const SR = 48000, N = SR*2;
  const mk = (fn)=>{ const a = new Float32Array(N); for(let i=0;i<N;i++) a[i]=fn(i/SR,i); return a; };
  const bandOf = (f, hz) => {
    const idx = KC_FEATURES.findIndex(x=>x.band && x.lo <= hz && hz < x.hi);
    return idx >= 0 ? f[KC_FEATURES[idx].key] : null;
  };

  const quiet = kcFeatures(mk(t=>1e-6*Math.sin(2*Math.PI*440*t)), SR);
  ok(quiet.silent === true, "a take below -45 dBFS is reported silent");

  const loud = kcFeatures(mk(t=>0.5*Math.sin(2*Math.PI*440*t)), SR);
  const soft = kcFeatures(mk(t=>0.05*Math.sin(2*Math.PI*440*t)), SR);
  ok(Math.abs((loud.rms - soft.rms) - 20) < 0.1,
     `level difference measured as ${(loud.rms-soft.rms).toFixed(2)} dB (20 expected)`);

  // A tone must land in its own band and nowhere else.
  const lo = kcFeatures(mk(t=>0.3*Math.sin(2*Math.PI*150*t)), SR);
  const hi = kcFeatures(mk(t=>0.3*Math.sin(2*Math.PI*4000*t)), SR);
  ok(bandOf(lo,150) > bandOf(lo,4000) + 20, "a 150 Hz tone shows up in the 120-200 Hz band");
  ok(bandOf(hi,4000) > bandOf(hi,150) + 20, "a 4 kHz tone shows up in the 3.15-5 kHz band");

  const sine = kcFeatures(mk(t=>0.3*Math.sin(2*Math.PI*440*t)), SR);
  const clicks = kcFeatures(mk((t,i)=> (i % 4800 === 0) ? 0.9 : 0.0), SR);
  ok(clicks.crest > sine.crest + 10, "crest factor separates peaky from steady material");

  const dyn  = kcFeatures(mk((t)=> 0.4*Math.sin(2*Math.PI*440*t) * (t < 1 ? 1 : 0.05)), SR);
  const flat = kcFeatures(mk((t)=> 0.4*Math.sin(2*Math.PI*440*t)), SR);
  ok(dyn.dynamics > flat.dynamics + 10,
     `varying material reads wider than steady (${dyn.dynamics.toFixed(1)} vs ${flat.dynamics.toFixed(1)} dB)`);

  // Level must not leak into the band figures: they are relative by design, so
  // a gain trim on the interface cannot look like a tone control.
  const band = f => KC_FEATURES.filter(x=>x.band).map(x=>f[x.key]);
  const a = band(loud), b = band(soft);
  ok(a.every((v,i)=>Math.abs(v-b[i]) < 0.2), "band levels are unmoved by a 20 dB gain change");
}

console.log("\n7. A nonlinear amp, and six knobs");
{
  // Real controls are not linear: they saturate at the top, do almost nothing at
  // the bottom, and interact. Local linear regression should still track it,
  // because it only ever has to be right nearby.
  const K6 = [
    {name:"Gain",     min:0, max:10, step:0.5},
    {name:"Bass",     min:0, max:10, step:0.5},
    {name:"Mid",      min:0, max:10, step:0.5},
    {name:"Treble",   min:0, max:10, step:0.5},
    {name:"Presence", min:0, max:10, step:0.5},
    {name:"Master",   min:0, max:10, step:0.5},
  ];
  const sat = x => 10*Math.tanh(x/5);        // knobs crowd up at the top
  const respond6 = (v, rng, noise) => {
    const [g,b,m,tr,pr,ma] = v.map(sat);
    const n = () => (noise||0)*(rng()*2-1);
    const out = {silent:false,
      rms:      -26 + 1.25*ma + 0.30*g + 0.02*g*ma + n(),
      crest:     15 - 0.80*g + 0.01*g*g + n(),
      dynamics:  19 - 1.15*g + n(),
    };
    // Each tone control owns a region of the spectrum, the way a real stack does.
    for(let i=0;i<NB;i++){
      const c = (i - (NB-1)/2)/((NB-1)/2);
      const bell = (at, w) => Math.exp(-0.5*Math.pow((c-at)/w, 2));
      out[`b${i}`] = -10 + 2.5*Math.cos(i*0.8)
        + 0.55*b*bell(-0.85,0.45) + 0.50*m*bell(-0.1,0.40)
        + 0.60*tr*bell(0.55,0.45) + 0.40*pr*bell(0.95,0.35)
        + 0.10*g*Math.max(0,c) + n();
    }
    return out;
  };
  const make6 = (n, seed, noise) => {
    const rng = mulberry(seed), runs = [];
    const perm = () => { const a=[...Array(n).keys()];
      for(let i=n-1;i>0;i--){const j=Math.floor(rng()*(i+1));[a[i],a[j]]=[a[j],a[i]];} return a; };
    const cols = K6.map(()=>perm());
    for(let i=0;i<n;i++){
      const values = K6.map((k,ki)=>{ const u=(cols[ki][i]+rng())/n;
        return Math.round((k.min+u*(k.max-k.min))/k.step)*k.step; });
      runs.push({run:i+1, values, f:respond6(values, rng, noise)});
    }
    return runs;
  };

  let falseAlarms = 0, total = 0;
  for(let s=100; s<108; s++){
    const runs = make6(60, s, 0.3);
    const r = kcAnalyse(runs, K6);
    total += runs.length;
    falseAlarms += r.findings.filter(f=>f.status!=="sparse").length;
  }
  console.log(`   ${falseAlarms} flags across ${total} good runs, 6 knobs, saturating response`);
  ok(falseAlarms <= 2, "curvature and six knobs do not manufacture alarms");

  let caught = 0, named = 0, n = 0;
  for(let s=110; s<134; s++){
    const ki = s % 6;
    const runs = make6(60, s, 0.3);
    const v = runs[(s*3) % 60];
    const want = v.values[ki] + ((s%2)?1:-1)*3.5;
    if(want < 0 || want > 10) continue;
    n++;
    const actual = v.values.slice(); actual[ki] = want;
    v.f = respond6(actual, mulberry(s*17), 0.3);
    const f = kcAnalyse(runs, K6).findings.find(x=>x.run===v.run && x.status!=="sparse");
    if(f) caught++;
    if(f && f.suggestion && f.suggestion.knob === K6[ki].name) named++;
  }
  console.log(`   caught ${caught}/${n}, right knob named ${named}/${n}`);
  ok(caught >= 0.85*n, "still catches a 3.5-unit error with six interacting knobs");
  ok(named >= 0.7*n, "and still usually names the right one");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
