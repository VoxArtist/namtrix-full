// Tests for the per-chain session delay used by training (index.html).
//     node tests/test_delays.js
// The functions are read out of the page itself, so the test covers the shipped code.
const fs = require("fs"), path = require("path");
const page = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const code = page.slice(page.indexOf("const DELAY_GROUP_GAP"), page.indexOf("function matrixFits(matrix){"));
let failures = 0;
const check = (name, ok, detail) => { console.log((ok ? "ok   " : "FAIL ") + name + (ok || detail === undefined ? "" : `  (${detail})`)); if (!ok) failures++; };

function make(takeDelays, done, takeBlocks = {}) {
  const state = { takeDelays, takeBlocks, runDone: {}, holdoutDone: {} };
  for (const k of done) { const [kind, run] = k.split(":"); (kind === "holdout" ? state.holdoutDone : state.runDone)[run] = true; }
  const isDone = (kind, run) => !!(kind === "holdout" ? state.holdoutDone : state.runDone)[run];
  return new Function("state", "isDone", code + "; return {delayGroups, takeDelay, measuredDelay};")(state, isDone);
}
const amp = { name: "amp", delay: 870 };

// readings that drift with the amp's settings: every take trains at the same, early delay
{
  const td = {}, done = [];
  [12391, 12378, 12403, 12384, 12395, 12380, 12399, 12386, 12390, 12382].forEach((d, i) => { td[`train:${i+1}:amp`] = d; done.push(`train:${i+1}`); });
  const f = make(td, done);
  const used = done.map(k => f.takeDelay(...k.split(":").map((x, i) => i ? +x : x), amp));
  check("one delay for every take of a chain", new Set(used).size === 1, used);
  check("it is the early end of the readings", used[0] === 12378 || used[0] === 12380, used[0]);
  check("never later than any reading", used[0] <= Math.min(...Object.values(td)));
}
// a misfire hundreds of samples late does not move it, and that take gets the group's delay
{
  const td = {}, done = [];
  for (let i = 1; i <= 20; i++) { td[`train:${i}:amp`] = 12440 + (i % 10); done.push(`train:${i}`); }
  td["train:21:amp"] = 13386; done.push("train:21");
  const f = make(td, done);
  check("a late misfire trains at the group's delay", f.takeDelay("train", 21, amp) === f.takeDelay("train", 1, amp), f.takeDelay("train", 21, amp));
  check("the misfire does not pull the delay up", f.takeDelay("train", 1, amp) <= 12442);
}
// two buffer sizes in one session stay apart
{
  const td = {}, done = [];
  for (let i = 1; i <= 10; i++) { td[`train:${i}:amp`] = 12380 + i; done.push(`train:${i}`); }
  for (let i = 11; i <= 20; i++) { td[`train:${i}:amp`] = 866 + (i % 5); done.push(`train:${i}`); }
  const f = make(td, done);
  check("two buffer sizes form two groups", f.delayGroups("amp").length === 2);
  check("an old take keeps the old group's delay", f.takeDelay("train", 3, amp) > 12000);
  check("a new take gets the new group's delay", f.takeDelay("train", 15, amp) < 1000);
  check("a take with no reading joins the group nearest the chain's current delay", (() => { const g = make(td, [...done, "train:21"]); return g.takeDelay("train", 21, amp) < 1000; })());
}
// readings of runs not recorded (or since replaced) are ignored
{
  const f = make({ "train:1:amp": 100, "train:2:amp": 50 }, ["train:1"]);
  check("only recorded takes count", f.takeDelay("train", 1, amp) === 100);
}
// the separate validation cut's reading is not a take of its own
{
  const f = make({ "train:1:amp": 100, "train:1:amp:val": 20 }, ["train:1"]);
  check("validation-cut readings are left out", f.delayGroups("amp").length === 1 && f.delayGroups("amp")[0].lo === 100);
}
// takes that know their buffer size group by it, and a take with no reading uses its group's delay
{
  const td = {}, done = [], blocks = {};
  for (let i = 1; i <= 10; i++) { td[`train:${i}:amp`] = 12380 + i; done.push(`train:${i}`); blocks[`train:${i}`] = "high"; }
  for (let i = 11; i <= 20; i++) { td[`train:${i}:amp`] = 866 + (i % 5); done.push(`train:${i}`); blocks[`train:${i}`] = 256; }
  done.push("train:21"); blocks["train:21"] = 256;          // quiet take: clicks unreadable, no reading at all
  done.push("train:22"); blocks["train:22"] = "high";
  const stale = { name: "amp", delay: 12390 };               // chain's last reading from the other buffer size
  const f = make(td, done, blocks);
  check("groups follow the recorded buffer size", f.delayGroups("amp").length === 2 && f.delayGroups("amp").every(g => g.block != null));
  check("a 256 take with no reading gets the 256 delay, whatever the chain last read", f.takeDelay("train", 21, stale) < 1000, f.takeDelay("train", 21, stale));
  check("an old take with no reading gets the old delay", f.takeDelay("train", 22, amp) > 12000, f.takeDelay("train", 22, amp));
  check("every 256 take trains at one delay", new Set([11, 15, 20, 21].map(r => f.takeDelay("train", r, stale))).size === 1);
}
// a single late misfire inside a buffer-size group changes nothing
{
  const td = {}, done = [], blocks = {};
  for (let i = 1; i <= 12; i++) { td[`train:${i}:amp`] = 870 + (i % 3); done.push(`train:${i}`); blocks[`train:${i}`] = 256; }
  td["train:13:amp"] = 1810; done.push("train:13"); blocks["train:13"] = 256;
  const f = make(td, done, blocks);
  check("a misfire in a keyed group trains at the group delay", f.takeDelay("train", 13, amp) === f.takeDelay("train", 1, amp), f.takeDelay("train", 13, amp));
}
console.log(failures ? `\n${failures} failed` : "\nall passed");
process.exit(failures ? 1 : 0);
