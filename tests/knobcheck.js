/* ========================= KNOB CHECK =========================
 * Did a knob actually sit where the run sheet says it did?
 *
 * Every run plays the same input signal, so two takes differ only by where the
 * knobs were. That is what makes this possible without a reference model, an
 * alignment step, or a trained network: the takes are directly comparable.
 *
 * Knob settings predict sound, and they do it smoothly. So a run's sound should
 * be predictable from the other runs. When a run disagrees with what every
 * other run says it should be, the run sheet is a likelier culprit than the amp.
 *
 * Three things had to be right for this to work on real captures rather than
 * only on tidy synthetic ones:
 *
 *   1. Narrow bands, not broad ones. "Everything above 2 kHz" tracked this
 *      amp's tone stack with an R^2 of 0.05. Eleven third-octave-ish bands
 *      track it at 0.7 to 0.85.
 *   2. The model's complexity chosen from the data. With 35 runs and 6 knobs a
 *      quadratic term overfits, and its leave-one-out error is 40% worse than a
 *      plain linear fit on the feature that matters most.
 *   3. A matched filter, not a per-feature threshold. One knob moves several
 *      bands together, by a fixed pattern. Testing that whole pattern at once is
 *      what turns an undetectable error into a detectable one - on this amp it
 *      is the difference between needing an 11-unit error and a 5-unit one.
 * ============================================================= */

// Band edges across the guitar range, roughly third-octave where tone controls
// act. Relative levels, so the interface's gain trim cannot masquerade as a knob.
const KC_BAND_EDGES = [60,120,200,320,500,800,1250,2000,3150,5000,8000,12000];

// `floor` is the smallest spread that will ever be believed. Without it an
// unusually consistent set of runs divides by nearly nothing and every run
// becomes a fault.
const KC_FEATURES = [
  {key:"rms",      label:"level",         unit:"dB", floor:0.30},
  {key:"crest",    label:"crest factor",  unit:"dB", floor:0.30},
  {key:"dynamics", label:"dynamic range", unit:"dB", floor:0.40},
];
for(let i=0;i<KC_BAND_EDGES.length-1;i++){
  const lo = KC_BAND_EDGES[i], hi = KC_BAND_EDGES[i+1];
  const name = hi >= 1000 ? `${(lo/1000).toFixed(lo<1000?1:0)}-${(hi/1000).toFixed(0)} kHz`
                          : `${lo}-${hi} Hz`;
  KC_FEATURES.push({key:`b${i}`, label:name, unit:"dB", floor:0.35, band:true, lo, hi});
}

const KC_MIN_RUNS   = 12;   // below this there is no population to compare against
const KC_FLAG_Z     = 4.0;  // a false alarm costs a re-record, so be reluctant
const KC_EXPLAIN_Z  = 2.5;  // what must be left over for an answer to count
const KC_ODD_Z      = 5.5;  // "unlike any other run", with no knob to blame
const KC_SILENT_DBFS = -45; // matches the ESR validator's own silence threshold
// Leverage past this multiple of average means the model extrapolated to reach
// this run, and a guess is not grounds for making someone re-record.
const KC_MAX_LEVERAGE_RATIO = 4.0;
// Statistics alone will happily flag a knob a third of a detent out. Nobody can
// set a pointer knob that accurately, and the matrix snaps to detents anyway, so
// an error has to be this big before it is worth interrupting someone over:
// either a couple of real detents, or a tenth of the knob's travel.
const KC_MIN_STEPS = 1.5;
const KC_MIN_TRAVEL = 0.08;

/* ---------- FFT: iterative radix-2, in place ---------- */
function kcFFT(re, im){
  const n = re.length;
  for(let i=1, j=0; i<n; i++){
    let bit = n >> 1;
    for(; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if(i < j){ const tr=re[i]; re[i]=re[j]; re[j]=tr; const ti=im[i]; im[i]=im[j]; im[j]=ti; }
  }
  for(let len=2; len<=n; len<<=1){
    const ang = -2*Math.PI/len, wr = Math.cos(ang), wi = Math.sin(ang);
    for(let i=0; i<n; i+=len){
      let cr = 1, ci = 0;
      for(let k=0; k<len/2; k++){
        const ur=re[i+k], ui=im[i+k];
        const vr=re[i+k+len/2]*cr - im[i+k+len/2]*ci;
        const vi=re[i+k+len/2]*ci + im[i+k+len/2]*cr;
        re[i+k]=ur+vr; im[i+k]=ui+vi;
        re[i+k+len/2]=ur-vr; im[i+k+len/2]=ui-vi;
        const ncr = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = ncr;
      }
    }
  }
}

const KC_FFT_SIZE = 4096;
const KC_MAX_BLOCKS = 240;   // enough to average down, cheap enough to stay instant

/* ---------- what one take measures as ---------- */
function kcFeatures(samples, sampleRate){
  const n = samples.length;
  if(!n) return null;

  let sumSq = 0, peak = 0;
  for(let i=0;i<n;i++){ const v=samples[i]; sumSq += v*v; const a=v<0?-v:v; if(a>peak) peak=a; }
  const rmsLin = Math.sqrt(sumSq/n);
  const toDb = v => v>0 ? 20*Math.log10(v) : -Infinity;
  const rms = toDb(rmsLin);

  const blank = () => {
    const o = {silent:true, rms, crest:0, dynamics:0};
    for(let i=0;i<KC_BAND_EDGES.length-1;i++) o[`b${i}`] = 0;
    return o;
  };
  // Every control at minimum makes the amp silent. That is correct behaviour,
  // not a fault, and silence has no spectrum to compare.
  if(!(rmsLin > 0) || rms < KC_SILENT_DBFS) return blank();

  // Welch-style average over blocks spread across the whole take, so a loud
  // opening cannot stand in for the rest of it.
  const size = KC_FFT_SIZE, half = size>>1;
  const nBlocks = Math.max(1, Math.min(KC_MAX_BLOCKS, Math.floor(n/size)));
  const stride = Math.max(size, Math.floor((n - size)/Math.max(1,nBlocks-1)) || size);
  const win = new Float64Array(size);
  for(let i=0;i<size;i++) win[i] = 0.5 - 0.5*Math.cos(2*Math.PI*i/(size-1));
  const power = new Float64Array(half);
  const re = new Float64Array(size), im = new Float64Array(size);
  let used = 0;
  for(let b=0; b<nBlocks; b++){
    const start = b*stride;
    if(start + size > n) break;
    let blockE = 0;
    for(let i=0;i<size;i++){ const v = samples[start+i]; blockE += v*v; }
    if(blockE <= 0) continue;                    // a silent block carries no colour
    for(let i=0;i<size;i++){ re[i] = samples[start+i]*win[i]; im[i] = 0; }
    kcFFT(re, im);
    for(let k=0;k<half;k++) power[k] += re[k]*re[k] + im[k]*im[k];
    used++;
  }
  if(!used) return blank();

  const binHz = sampleRate/size;
  const nBands = KC_BAND_EDGES.length-1;
  const acc = new Float64Array(nBands);
  let total = 0;
  for(let k=1;k<half;k++){
    const f = k*binHz;
    if(f < 25 || f > 16000) continue;            // rumble and the anti-alias filter
    total += power[k];
    for(let i=0;i<nBands;i++){
      if(f >= KC_BAND_EDGES[i] && f < KC_BAND_EDGES[i+1]){ acc[i] += power[k]; break; }
    }
  }
  if(!(total > 0)) return blank();

  // Dynamic range: the spread of short-block levels. A saturating amp squashes
  // it, which is what makes this read Drive rather than Volume.
  const blockLen = Math.max(1, Math.round(0.1*sampleRate));
  const levels = [];
  for(let s=0; s+blockLen<=n; s+=blockLen){
    let e=0; for(let i=0;i<blockLen;i++){ const v=samples[s+i]; e+=v*v; }
    const r = Math.sqrt(e/blockLen);
    if(r > 0) levels.push(20*Math.log10(r));
  }
  levels.sort((a,b)=>a-b);
  const pct = q => levels.length
    ? levels[Math.min(levels.length-1, Math.max(0, Math.round(q*(levels.length-1))))] : 0;

  const out = {
    silent: false,
    rms,
    crest: toDb(peak) - rms,
    dynamics: levels.length >= 8 ? pct(0.95) - pct(0.25) : 0,
  };
  // Floor relative to the take's own energy, never absolute: an absolute one
  // would bite on a quiet take and not on a loud one, which is exactly the
  // level dependence these bands exist to avoid.
  const bandFloor = 1e-12*total;
  for(let i=0;i<nBands;i++) out[`b${i}`] = 10*Math.log10(Math.max(acc[i],bandFloor)/total);
  return out;
}

/* ---------- knob space ---------- */
function kcNormalise(values, knobs){
  return values.map((v,i)=>{
    const k = knobs[i];
    const span = (k.max - k.min) || 1;
    return (Number(v) - k.min)/span;
  });
}

function kcMedian(xs){
  if(!xs.length) return 0;
  const s = xs.slice().sort((a,b)=>a-b), m = s.length>>1;
  return s.length%2 ? s[m] : 0.5*(s[m-1]+s[m]);
}
const kcMad = (xs) => { const m = kcMedian(xs); return kcMedian(xs.map(v=>Math.abs(v-m))); };

/* ---------- small dense linear algebra ---------- */
// Gauss-Jordan with partial pivoting, several right-hand sides at once. The
// systems here are at most 13x13, so nothing cleverer is warranted.
function kcSolve(A, rhsList){
  const n = A.length, m = rhsList.length;
  const M = A.map((row,i)=>{ const r = Array.from(row); for(const b of rhsList) r.push(b[i]); return r; });
  for(let c=0;c<n;c++){
    let piv = c;
    for(let r=c+1;r<n;r++) if(Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if(Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for(let r=0;r<n;r++){
      if(r===c) continue;
      const factor = M[r][c]/M[c][c];
      if(!factor) continue;
      for(let k=c;k<n+m;k++) M[r][k] -= factor*M[c][k];
    }
  }
  const out = [];
  for(let j=0;j<m;j++) out.push(M.map((row,i)=>row[n+j]/M[i][i]));
  return out;
}

/*
 * The response model: one smooth curve per knob, added together.
 *
 *   linear     feature = b0 + sum_j b_j*u_j
 *   quadratic  feature = b0 + sum_j (b_j*u_j + c_j*u_j^2),    u_j = knob_j - 0.5
 *
 * Which one is used is decided by the data, not in advance, because the right
 * answer changes with how many runs there are. On 35 runs and 6 knobs the
 * quadratic term overfits badly. On a three-knob pedal with 40 runs it earns
 * its place. Leave-one-out error picks between them.
 *
 * Either way it assumes knobs act mostly independently. They do not act
 * *entirely* independently - a tone stack interacts - but the leftover lands in
 * the residuals, which widens the spread and makes the check more cautious
 * rather than wrong.
 */
const KC_ROWS = {
  linear:    x => { const r=[1]; for(const v of x) r.push(v-0.5); return r; },
  quadratic: x => { const r=[1]; for(const v of x){ const u=v-0.5; r.push(u,u*u); } return r; },
};

function kcFit(points, knobCount, shape){
  const rowFn = KC_ROWS[shape];
  const cols = rowFn(points[0].x).length;
  if(points.length < cols + 2) return null;

  const A = []; for(let i=0;i<cols;i++) A.push(new Float64Array(cols));
  const rhs = KC_FEATURES.map(()=>new Float64Array(cols));
  for(const pt of points){
    const row = rowFn(pt.x);
    for(let a=0;a<cols;a++){
      for(let b=0;b<cols;b++) A[a][b] += row[a]*row[b];
      KC_FEATURES.forEach((f,fi)=>{ rhs[fi][a] += row[a]*pt.f[f.key]; });
    }
  }
  // Ridge on the shape terms only, never the intercept: a knob that never moved
  // should flatten to "no effect" rather than pick up a wild coefficient.
  let trace = 0; for(let j=1;j<cols;j++) trace += A[j][j];
  for(let j=1;j<cols;j++) A[j][j] += 1e-4*(trace/Math.max(1,cols-1)) + 1e-9;

  const solved = kcSolve(A, rhs);
  if(!solved) return null;
  const coef = {};
  KC_FEATURES.forEach((f,fi)=>{ coef[f.key] = solved[fi]; });

  const eye = [];
  for(let i=0;i<cols;i++){ const e = new Float64Array(cols); e[i] = 1; eye.push(e); }
  const inv = kcSolve(A, eye);

  return {
    shape, cols, n: points.length,
    predict(x){
      const row = rowFn(x), out = {};
      for(const f of KC_FEATURES){
        let v = 0; const c = coef[f.key];
        for(let i=0;i<row.length;i++) v += row[i]*c[i];
        out[f.key] = v;
      }
      return out;
    },
    // d(feature)/d(normalised knob j): what turns "this run is wrong" into
    // "this knob is wrong, and by this much".
    gradient(x){
      const out = {};
      for(const f of KC_FEATURES){
        const c = coef[f.key], g = [];
        for(let j=0;j<knobCount;j++){
          g.push(shape === "linear" ? c[1+j] : c[1+2*j] + 2*c[2+2*j]*(x[j]-0.5));
        }
        out[f.key] = g;
      }
      return out;
    },
    // How far outside the fitted data a point sits, in the regression's own
    // terms. Around cols/n for a typical run; several times that means the
    // model is guessing.
    leverage(x){
      if(!inv) return Infinity;
      const row = rowFn(x);
      let h = 0;
      for(let a=0;a<cols;a++){
        let acc = 0;
        for(let b=0;b<cols;b++) acc += inv[b][a]*row[b];
        h += row[a]*acc;
      }
      return h;
    },
  };
}

// Fit every shape the data can support and keep the one that predicts runs it
// has not seen best. More parameters always fit the runs you have better, which
// is exactly why that is not the thing to measure.
function kcChooseShape(pts, K){
  let best = null;
  for(const shape of ["linear","quadratic"]){
    // A model with nearly as many coefficients as runs will fit anything, and
    // its slopes go wild - which makes the check report a sensitivity it does
    // not have. Three runs per coefficient before a shape is even considered.
    const cols = KC_ROWS[shape](pts[0].x).length;
    if(pts.length < 3*cols) continue;
    const errs = [];
    for(const p of pts){
      const fit = kcFit(pts.filter(q=>q!==p), K, shape);
      if(!fit) { errs.length = 0; break; }
      const pred = fit.predict(p.x);
      for(const f of KC_FEATURES) errs.push(Math.abs(p.f[f.key] - pred[f.key])/f.floor);
    }
    if(!errs.length) continue;
    const score = kcMedian(errs);
    if(!best || score < best.score) best = {shape, score};
  }
  return best && best.shape;
}

/**
 * Check every analysed run against what the other runs say it should sound like.
 *
 * `runs`  [{run, values:[...], f:{features}}]
 * `knobs` [{name, min, max, step}]
 */
function kcAnalyse(runs, knobs, opts){
  opts = opts || {};
  const flagZ    = opts.flagZ    || KC_FLAG_Z;
  const explainZ = opts.explainZ || KC_EXPLAIN_Z;
  const oddZ     = opts.oddZ     || KC_ODD_Z;
  const usable = runs.filter(r => r.f && !r.f.silent);
  const silentCount = runs.filter(r => r.f && r.f.silent).length;
  const K = knobs.length;
  // Enough runs to pin down a linear response with room to spare. Fewer than
  // this and the residual spread is mostly the model's own uncertainty, which
  // would make the check confidently wrong about its own resolution.
  const need = Math.max(KC_MIN_RUNS, 3*(K + 1));

  if(usable.length < need){
    return {ready:false, analysed:usable.length, silentCount, findings:[], needed:need,
      reason:`Needs ${need} analysed runs before it can tell a mis-set knob from ordinary `
           + `variation; ${usable.length} so far.`};
  }

  const pts = usable.map(r => ({run:r.run, x:kcNormalise(r.values, knobs), f:r.f, values:r.values}));
  const shape = kcChooseShape(pts, K);
  if(!shape){
    return {ready:false, analysed:usable.length, silentCount, findings:[], needed:need,
      reason:`With ${K} knobs this needs at least ${3*(K+1)} analysed runs to model the amp `
           + `without fitting the noise; ${usable.length} so far.`};
  }

  // Leave one out, always. A run fitted partly to itself would drag the model
  // towards its own mistake and then look innocent - which is precisely the run
  // this exists to catch.
  const preds = {}, grads = {}, levs = {};
  for(const p of pts){
    const fit = kcFit(pts.filter(q => q !== p), K, shape);
    if(!fit) continue;
    preds[p.run] = fit.predict(p.x);
    grads[p.run] = fit.gradient(p.x);
    levs[p.run]  = fit.leverage(p.x)/(fit.cols/fit.n);   // 1.0 is an average run
  }
  const judged = pts.filter(p => preds[p.run]);
  if(judged.length < need){
    return {ready:false, analysed:usable.length, silentCount, findings:[], needed:need,
      reason:"The runs recorded so far do not vary enough to fit a model."};
  }

  // Robust spread per feature. A mean and a standard deviation would be dragged
  // about by the very outliers this exists to find; a median absolute deviation
  // is not.
  const scale = {};
  for(const f of KC_FEATURES){
    const rs = judged.map(p => p.f[f.key] - preds[p.run][f.key]);
    scale[f.key] = Math.max(1.4826*kcMad(rs), f.floor);
  }
  const zOf = p => KC_FEATURES.map(f => (p.f[f.key] - preds[p.run][f.key])/scale[f.key]);

  // The matched filter, whitened.
  //
  // One knob moves a whole set of bands together, by a pattern the model already
  // knows, so testing for the entire pattern at once beats asking whether any
  // single band looks wrong. But bands do not vary independently: a take that is
  // generally darker is darker across every high band at once. Left alone, that
  // ordinary drift looks exactly like a tone control moving, and it costs a
  // factor of two and a half - a Treble error that needs 3.6 knob units to show
  // up stays buried until 8.5.
  //
  // So the residuals are decorrelated first and the knob's pattern tested in
  // that space. Directions where takes drift about on their own are discounted;
  // directions peculiar to one knob are not.
  const F = KC_FEATURES.length;
  const zs = {};
  for(const p of judged) zs[p.run] = zOf(p);

  // Covariance of the residuals, over the tidier three quarters of the runs.
  // Trimming stops the very outliers this is hunting from widening the noise
  // model that is meant to reveal them.
  const byNorm = judged.slice().sort((a,b)=>
    zs[a.run].reduce((s,v)=>s+v*v,0) - zs[b.run].reduce((s,v)=>s+v*v,0));
  const clean = byNorm.slice(0, Math.max(K+2, Math.ceil(0.75*byNorm.length)));
  const cov = [];
  for(let a=0;a<F;a++) cov.push(new Float64Array(F));
  for(const p of clean){
    const z = zs[p.run];
    for(let a=0;a<F;a++) for(let b=0;b<F;b++) cov[a][b] += z[a]*z[b];
  }
  for(let a=0;a<F;a++) for(let b=0;b<F;b++) cov[a][b] /= clean.length;

  // Shrink towards the identity. Estimating a full covariance from a few dozen
  // runs is hopeless on its own; shrinkage makes it usable, and leans harder on
  // the identity the fewer runs there are.
  const alpha = Math.min(0.9, Math.max(0.15, (F + 2)/(clean.length + F)));
  for(let a=0;a<F;a++){
    for(let b=0;b<F;b++) cov[a][b] *= (1 - alpha);
    cov[a][a] += alpha;
  }
  const eyeF = [];
  for(let i=0;i<F;i++){ const e = new Float64Array(F); e[i] = 1; eyeF.push(e); }
  const covInv = kcSolve(cov, eyeF);
  const applyInv = (v) => {
    if(!covInv) return Array.from(v);
    const out = new Array(F).fill(0);
    for(let a=0;a<F;a++){ let acc = 0; for(let b=0;b<F;b++) acc += covInv[b][a]*v[b]; out[a] = acc; }
    return out;
  };

  const proj = {}, gnorm = {}, maha = {};
  for(const p of judged){
    const z = zs[p.run], wz = applyInv(z);
    maha[p.run] = Math.sqrt(Math.max(z.reduce((a,v,i)=>a + v*wz[i], 0), 0));
    const g = [], t = [];
    for(let j=0;j<K;j++){
      const gj = KC_FEATURES.map(f => grads[p.run][f.key][j]/scale[f.key]);
      const wg = applyInv(gj);
      const norm = Math.sqrt(Math.max(gj.reduce((a,v,i)=>a + v*wg[i], 0), 0));
      g.push({gj, wg, norm});
      t.push(norm > 1e-6 ? z.reduce((a,v,i)=>a + v*wg[i], 0)/norm : 0);
    }
    gnorm[p.run] = g;
    proj[p.run] = t;
  }

  // Standardise each knob's statistic against its own spread across the runs.
  // Whitening should leave these near unit variance, but measuring rather than
  // assuming costs nothing and covers whatever the shrinkage did not.
  const projScale = [], projMed = [];
  for(let j=0;j<K;j++){
    const col = judged.map(p => proj[p.run][j]);
    projMed.push(kcMedian(col));
    projScale.push(Math.max(1.4826*kcMad(col), 0.7));
  }

  // How unlike the others a run is overall, whitened, for faults with no knob
  // shape at all.
  const mahas = judged.map(p => maha[p.run]);
  const mahaMed = kcMedian(mahas);
  const mahaScale = Math.max(1.4826*kcMad(mahas), 0.5);

  // How big an error would have to be before it could be seen. Worth reporting
  // whatever the findings are: on a saturated amp whose knobs barely move the
  // sound, the honest answer is "more than this tool can see", and that is a
  // far better thing to say than nothing.
  const limits = knobs.map((k,j)=>{
    const norms = judged.map(p => gnorm[p.run][j].norm).filter(v=>v>1e-6);
    const typical = norms.length ? kcMedian(norms) : 0;
    const span = (k.max - k.min) || 1;
    const statistical = typical > 1e-6 ? flagZ*projScale[j]/typical*span : Infinity;
    return {
      knob: k.name,
      units: Math.max(statistical, Math.max(KC_MIN_STEPS*(k.step || 0.5), KC_MIN_TRAVEL*span)),
      limitedByNoise: statistical > Math.max(KC_MIN_STEPS*(k.step || 0.5), KC_MIN_TRAVEL*span),
    };
  });

  // Statistically clear is not the same as worth acting on: this is the second
  // test, and it is a physical one.
  const meaningful = (j, delta) => {
    const k = knobs[j], span = (k.max - k.min) || 1;
    return Math.abs(delta*span) >= Math.max(KC_MIN_STEPS*(k.step || 0.5), KC_MIN_TRAVEL*span);
  };

  const findings = [];
  for(const p of judged){
    const z = zs[p.run];
    const t = proj[p.run].map((v,j)=>(v - projMed[j])/projScale[j]);

    let bestKnob = 0;
    for(let j=1;j<K;j++) if(Math.abs(t[j]) > Math.abs(t[bestKnob])) bestKnob = j;
    const knobStat = t[bestKnob];

    // How far this knob would have had to move, before deciding whether to care.
    const moveFor = (j) => {
      const {wg, norm} = gnorm[p.run][j];
      return norm > 1e-6 ? z.reduce((a,v,i)=>a + v*wg[i], 0)/(norm*norm) : 0;
    };
    const knobShaped = Math.abs(knobStat) > flagZ && meaningful(bestKnob, moveFor(bestKnob));

    let worstF = 0, worstZ = 0;
    z.forEach((v,i)=>{ if(Math.abs(v) > Math.abs(worstZ)){ worstZ = v; worstF = i; } });

    // Unlike every other run, but not in the shape of any one knob: a patch lead
    // half out, a channel that switched off, an amp doing something
    // discontinuous at one end of its travel.
    const justOdd = Math.abs(worstZ) > oddZ
                 || (maha[p.run] - mahaMed)/mahaScale > oddZ;
    if(!knobShaped && !justOdd) continue;

    const overExtended = levs[p.run] > KC_MAX_LEVERAGE_RATIO;
    let suggestion = null;
    if(knobShaped){
      // Least-squares move along this knob's gradient, snapped to a real detent.
      const {gj, wg, norm} = gnorm[p.run][bestKnob];
      const delta = moveFor(bestKnob);
      const k = knobs[bestKnob], span = (k.max - k.min) || 1, step = k.step || 0.5;
      let value = p.values[bestKnob] + delta*span;
      value = Math.max(k.min, Math.min(k.max, Math.round(value/step)*step));
      const applied = (value - p.values[bestKnob])/span;
      if(Math.abs(applied) > 1e-9){
        let left = 0;
        z.forEach((v,i)=>{ const r = v - applied*gj[i]; if(Math.abs(r) > Math.abs(left)) left = r; });
        if(Math.abs(left) <= Math.max(explainZ, Math.abs(worstZ)))
          suggestion = {knob:k.name, knobIndex:bestKnob, value, was:p.values[bestKnob]};
      }
    }

    findings.push({
      run: p.run,
      status: overExtended ? "uncertain" : suggestion ? "explained" : "odd",
      z: knobShaped ? knobStat : worstZ,
      knobStat, worstZ,
      feature: KC_FEATURES[worstF].key,
      featureLabel: KC_FEATURES[worstF].label,
      unit: KC_FEATURES[worstF].unit,
      measured: p.f[KC_FEATURES[worstF].key],
      expected: preds[p.run][KC_FEATURES[worstF].key],
      leverage: levs[p.run],
      distance: maha[p.run],
      suggestion,
      note: overExtended
        ? "These knob settings sit outside the spread of every other run, so the model "
        + "had to extrapolate to predict them. Treat this as a hint, not a verdict."
        : undefined,
    });
  }

  findings.sort((a,b) => Math.abs(b.z||0) - Math.abs(a.z||0));
  return {ready:true, analysed:usable.length, silentCount, findings, scale, shape, limits};
}

if(typeof module !== "undefined")
  module.exports = {kcFeatures, kcAnalyse, kcNormalise, kcFit, kcChooseShape, KC_FEATURES, kcFFT};
