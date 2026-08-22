/* =====================================================================
UI + number spiral.  The algorithms themselves live in three
self-contained modules, so that each can be read on its own:
    algorithm-a.js    exact one-touch streams        (algorithm.md §3)
    algorithm-b.js    wheeled streams                (algorithm.md §4)
    algorithm-c.js    min-factor exponent spine      (§4C, min_factor.md)
with primegen-core.js holding the wheel tables, the min-heap and the
reference sieve.  Every module exports the same four things:
    meta            id / title / spec reference / blurb
    stream(opts)    the unbounded generator
    run(N, opts)    bounded driver → { primes, stats, wheel }
    summary(res,N)  algorithm-specific report lines

The spiral no longer paints a prime dot mask.  Every cell carries the
*factorisation field* of n — hue = ω(n) (or Ω(n)), lightness = sopfr(n)/n
— which is computed, pyramided and rendered by spiral-field.js.  What
stays here is the layout (square Ulam walk / axial hex walk, both in
closed form as one `cellToN`), the pan/zoom state and the DOM.
===================================================================== */
import { assertExact, sieveRef } from './primegen-core.js';
import * as AlgA from './algorithm-a.js';
import * as AlgB from './algorithm-b.js';
import * as AlgC from './algorithm-c.js';
import {
  computeFactorFields,
  buildFieldPyramid,
  renderFieldTo,
  pickLevel,
  drawFieldLegend,
  describeCell,
  FieldPalette,
  FIELD_MAX_CELLS,
  packRGBA,
} from './spiral-field.js';

const ALGOS = { A: AlgA, B: AlgB, C: AlgC };
const currentAlgo = () => ALGOS[$('algo').value] || AlgB;

/* =====================================================================
Number spiral — factorisation field
---------------------------------------------------------------------
Two regimes, both area-correct:

  cellPx ≥ 1        one cell covers ≥ 1 device pixel → the exact field
                    is sampled per pixel through the inverse layout
  cellPx < 1        a device pixel covers many cells → the pre-computed
                    pyramid of *sums* (Σ1, Σω, ΣΩ, Σ sopfr/n) is sampled
                    and averaged, RGB conversion happening last

Averaging sums (never packed pixels) is what keeps hue and brightness
honest when a texel covers thousands of integers; nearest-neighbour
downscaling of an RGB image would drop primes and wash out the hue.
Every path touches only the visible sub-rectangle, so pan/zoom costs
O(viewport pixels) independent of the grid size.
===================================================================== */
const MAX_SIDE = 4096; // L² = 2^24 cells = FIELD_MAX_CELLS
const MAX_RING = (MAX_SIDE - 1) >> 1; // hex: axial box side 2R+1 ≤ MAX_SIDE
const BG_RGB = [0x0b, 0x0f, 0x15];
const BG_PX = packRGBA(BG_RGB[0], BG_RGB[1], BG_RGB[2], 255);
/* hex lattice, pointy-top, cell width 1:
        X(q,r) = q + r/2 + R + 0.5      (columns are sheared by half a cell)
        Y(q,r) = (r + R + 0.5)·√3/2     (row pitch = √3/2, hex area = √3/2)
      so one hex occupies exactly the same area as one square cell of width 1. */
const HEX_K = Math.sqrt(3) / 2; // row pitch

const spiralCanvas = () => document.getElementById('spiralCanvas');
const pal = new FieldPalette();
let fields = null; // { N, omega, Omega, lum, omegaMax }
let pyr = null; // { base, levels } — null ⇒ nothing drawn yet
let cellToN = () => -1; // grid cell → integer, −1 outside the spiral
let spiralL = 0; // square: side L · hex: axial box side 2R+1
let spiralR = 0; // hex mode: ring count
let spiralView = { scale: 1, tx: 0, ty: 0, min: 0.01 };
let lastSpiral = null; // { primes, N } from the latest generator run
let renderPending = false;
let exactImg = null; // reused ImageData for the draw paths
let spiralMode = 'square'; // 'square' (Ulam) | 'hex'
let spiralCells = 0; // integers enumerated by the walk (L² or 1+3R(R+1))
let spiralExtW = 0; // grid bounding box in cell units (fit + clipping)
let spiralExtH = 0;
let spiralLabel = ''; // HUD prefix
let spiralOrigin = 1; // integer sitting on the centre cell (0…3)
let spiralLast = 0; // largest integer placed on the grid
let gridW = 0; // grid bounding box in cells (all three layouts)
let gridH = 0;
let sliceRowStart = null; // linear slice: Σ widths of the rows above
let sliceRowW = null; // linear slice: ⌊1 + δ·k⌋ per row
let sliceAlign = 'center'; // 'left' | 'center' | 'right'
let sliceDelta = 2; // row-width growth per row

/* =====================================================================
layout — closed-form inverse of the two walks
---------------------------------------------------------------------
Both return the 1-based position along the walk, so the integer on the
cell is `index + n₀ − 1`.  Closed form (rather than a walk that fills an
array) is what lets buildFieldPyramid stream over the grid without any
per-cell storage of its own.
===================================================================== */
/* square: centre at (L>>1, L>>1), first step +x, then −y (up) */
function squareIndex(x, y, L) {
  const c = L >> 1;
  const dx = x - c,
    dy = y - c;
  const r = Math.max(Math.abs(dx), Math.abs(dy));
  if (r === 0) return 1;
  const E = (2 * r + 1) * (2 * r + 1); // last integer of ring r
  if (dy === r) return E - (r - dx);
  if (dx === -r) return E - 2 * r - (r - dy);
  if (dy === -r) return E - 6 * r + (r - dx);
  if (dx === r) return E - 6 * r - (dy + r);
  return -1;
}
/* hex: ring k holds 6k cells, base = 1 + 3k(k−1) is its last predecessor */
function hexIndex(q, r, R) {
  const k = Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r));
  if (k === 0) return 1;
  if (k > R) return -1;
  const base = 1 + 3 * (k - 1) * k;
  if (q === 1 && r === k - 1) return base + 1;
  if (q + r === k && q >= 2 && q <= k) return base + q;
  if (q === k && r <= -1 && r >= -k) return base + k - r;
  if (r === -k && q >= 0 && q <= k - 1) return base + 3 * k - q;
  if (q + r === -k && q <= -1 && q >= -k) return base + 3 * k - q;
  if (q === -k && r >= 1 && r <= k) return base + 4 * k + r;
  if (r === k && q >= -k + 1 && q <= 0) return base + 6 * k + q;
  return -1;
}
/* =====================================================================
linear slice — the third layout
---------------------------------------------------------------------
Row k (top = 0) holds  w_k = ⌊1 + δ·k⌋  cells, i.e. the literal
recurrence "next row = ⌊δ + previous⌋" written in closed form so the
flooring never accumulates.  After R rows the walk has covered
     T(R) = Σ_{k<R} ⌊1 + δ·k⌋ ≈ R + δ·R(R−1)/2,
so δ selects the quadratic projection: δ = 2 gives the odd rows
1, 3, 5, … and T(R) = R² exactly (a right triangle / square when
centred), δ = 1 gives the triangular numbers, δ = 0 a single column.
Rows are aligned left, centred or right inside the bounding box; the
cells outside the slice return −1 and are painted as background.
===================================================================== */
function sliceRowWidth(k, delta) {
  return Math.max(1, Math.floor(1 + delta * k));
}
function buildSlice(budget, delta) {
  const starts = [],
    widths = [];
  let total = 0,
    maxW = 0;
  for (let k = 0; k < MAX_SIDE; k++) {
    const rw = sliceRowWidth(k, delta);
    if (rw > MAX_SIDE) break; // row wider than the largest grid we allow
    if (total + rw > budget) break; // would run past N
    const nw = rw > maxW ? rw : maxW;
    if (nw * (k + 1) > FIELD_MAX_CELLS) break; // bounding box budget
    starts.push(total);
    widths.push(rw);
    total += rw;
    maxW = nw;
  }
  const rows = widths.length;
  if (rows === 0) return null;
  return {
    starts: Int32Array.from(starts),
    widths: Int32Array.from(widths),
    rows,
    maxW,
    cells: total,
  };
}
/* the layout closure handed to buildFieldPyramid / renderFieldTo */
function makeCellToN() {
  const first = spiralOrigin,
    cells = spiralCells;
  if (spiralMode === 'hex') {
    const R = spiralR;
    return (x, y) => {
      const i = hexIndex(x - R, y - R, R);
      return i >= 1 && i <= cells ? i + first - 1 : -1;
    };
  }
  if (spiralMode === 'linear') {
    const starts = sliceRowStart,
      widths = sliceRowW,
      rows = widths ? widths.length : 0,
      W = gridW,
      align = sliceAlign;
    return (x, y) => {
      if (y < 0 || y >= rows) return -1;
      const rw = widths[y];
      const x0 = align === 'right' ? W - rw : align === 'center' ? (W - rw) >> 1 : 0;
      const dx = x - x0;
      if (dx < 0 || dx >= rw) return -1;
      const i = starts[y] + dx + 1; // 1-based position along the walk
      return i <= cells ? i + first - 1 : -1;
    };
  }
  const L = spiralL;
  return (x, y) => {
    const i = squareIndex(x, y, L);
    return i >= 1 && i <= cells ? i + first - 1 : -1;
  };
}
/* =====================================================================
linear slice — column statistics
---------------------------------------------------------------------
A column of the slice is *not* an arithmetic progression: row k starts
at Σ_{j<k} ⌊1 + δ·j⌋ and is then shifted by the alignment, so the
integers sitting in one column x follow a quadratic
       n(k) = (δ/2)·k² + β·k + γ ,
       β = 1 − δ/2 (left) · 1 (centre) · 1 + δ/2 (right),
       γ = n₀ + x  ·  n₀ + x − (W−1)/2  ·  n₀ + x − W + 1 .
When the floors are harmless (δ integral, parity of W − w_k constant in
the centred case) the polynomial is *exact*: it is then recovered from
three consecutive samples and verified against every cell of the
column.  Otherwise the closed form above is reported as an
approximation.  Aggregating ω, Ω and sopfr over the column is what
makes the vertical striping of the field — the residue structure mod
the row width — readable rather than merely visible.
===================================================================== */
let sliceColCache = { key: null, text: '' };
const trimNum = (v) =>
  !Number.isFinite(v)
    ? '—'
    : Math.abs(v - Math.round(v)) < 1e-9
      ? String(Math.round(v))
      : String(Math.round(v * 1e4) / 1e4);
function fmtPoly(a, b, c) {
  const term = (v, sym) => {
    if (Math.abs(v) < 1e-12) return '';
    const mag = Math.abs(v);
    const body = sym && Math.abs(mag - 1) < 1e-12 ? sym : trimNum(mag) + sym;
    return (v < 0 ? ' − ' : ' + ') + body;
  };
  const s = (term(a, 'k²') + term(b, 'k') + term(c, '')).replace(/^ \+ /, '').replace(/^ − /, '−');
  return `n(k) = ${s || '0'}`;
}
/* x0 of row width rw under the current alignment — same rule as makeCellToN */
function sliceRowX0(rw) {
  return sliceAlign === 'right' ? gridW - rw : sliceAlign === 'center' ? (gridW - rw) >> 1 : 0;
}
function columnQuadratic(ks, ns, x) {
  if (ks.length >= 3 && ks[1] === ks[0] + 1 && ks[2] === ks[0] + 2) {
    const k0 = ks[0];
    const a = (ns[2] - 2 * ns[1] + ns[0]) / 2;
    const b = ns[1] - ns[0] - a * (2 * k0 + 1);
    const c = ns[0] - a * k0 * k0 - b * k0;
    let ok = true;
    for (let i = 0; i < ks.length; i++) {
      const k = ks[i];
      if (a * k * k + b * k + c !== ns[i]) {
        ok = false;
        break;
      }
    }
    if (ok) return { a, b, c, exact: true };
  }
  /* closed form with the floors dropped */
  const d = sliceDelta,
    W = gridW,
    a = d / 2;
  if (sliceAlign === 'right') return { a, b: 1 + d / 2, c: spiralOrigin + x - W + 1, exact: false };
  if (sliceAlign === 'center') return { a, b: 1, c: spiralOrigin + x - (W - 1) / 2, exact: false };
  return { a, b: 1 - d / 2, c: spiralOrigin + x, exact: false };
}
/* the column block of the tooltip — cached, so pointermove stays free */
function describeColumn(x) {
  if (spiralMode !== 'linear' || !sliceRowW || !sliceRowStart || !fields) return '';
  if (x < 0 || x >= gridW) return '';
  const key = `${x}|${spiralLabel}`;
  if (sliceColCache.key === key) return sliceColCache.text;
  const rows = sliceRowW.length,
    ks = [],
    ns = [];
  for (let k = 0; k < rows; k++) {
    const rw = sliceRowW[k];
    const dx = x - sliceRowX0(rw);
    if (dx < 0 || dx >= rw) continue;
    const i = sliceRowStart[k] + dx + 1; // 1-based position along the walk
    if (i > spiralCells) break;
    ks.push(k);
    ns.push(i + spiralOrigin - 1);
  }
  let text = '';
  if (ks.length) {
    const q = columnQuadratic(ks, ns, x);
    const omA = fields.omega,
      OmA = fields.Omega,
      raA = fields.lum;
    let cnt = 0,
      pr = 0;
    let omMin = Infinity,
      omMax = -Infinity,
      omSum = 0;
    let OmMin = Infinity,
      OmMax = -Infinity,
      OmSum = 0;
    let spMin = Infinity,
      spMax = -Infinity,
      spSum = 0;
    let raMin = Infinity,
      raMax = -Infinity,
      raSum = 0;
    for (let i = 0; i < ns.length; i++) {
      const n = ns[i];
      if (n < 2 || n >= omA.length) continue; // ω, Ω undefined at 0 and 1
      const o = omA[n],
        O = OmA[n],
        r = raA ? raA[n] : NaN,
        s = r * n; // sopfr(n) = n·(sopfr/n)
      cnt++;
      if (O === 1) pr++;
      if (o < omMin) omMin = o;
      if (o > omMax) omMax = o;
      omSum += o;
      if (O < OmMin) OmMin = O;
      if (O > OmMax) OmMax = O;
      OmSum += O;
      if (s < spMin) spMin = s;
      if (s > spMax) spMax = s;
      spSum += s;
      if (r < raMin) raMin = r;
      if (r > raMax) raMax = r;
      raSum += r;
    }
    const avg = (s) => (cnt ? s / cnt : NaN);
    const lines = [
      `── column x = ${fmt(x)} of ${fmt(gridW)} · ${fmt(ks.length)} cells · ` +
        `rows k = ${fmt(ks[0])}…${fmt(ks[ks.length - 1])}`,
      `${fmtPoly(q.a, q.b, q.c)}${q.exact ? '   (exact)' : '   (≈, floors dropped)'}`,
      `Δ²n = ${trimNum(2 * q.a)} · n ∈ [${fmt(ns[0])}, ${fmt(ns[ns.length - 1])}]`,
    ];
    if (cnt)
      lines.push(
        `primes    ${fmt(pr)} / ${fmt(cnt)}  (${((100 * pr) / cnt).toFixed(2)} %)`,
        `ω(n)      min ${fmt(omMin)} · avg ${avg(omSum).toFixed(3)} · max ${fmt(omMax)}`,
        `Ω(n)      min ${fmt(OmMin)} · avg ${avg(OmSum).toFixed(3)} · max ${fmt(OmMax)}`,
        `sopfr(n)  min ${fmt(Math.round(spMin))} · avg ${fmt(Math.round(avg(spSum)))} · ` +
          `max ${fmt(Math.round(spMax))}`,
        `sopfr/n   min ${raMin.toFixed(3)} · avg ${avg(raSum).toFixed(3)} · max ${raMax.toFixed(3)}`
      );
    text = lines.join('\n');
  }
  sliceColCache = { key, text };
  return text;
}

/* --- palette / legend controls -------------------------------------- */
const hueSource = () =>
  $('fieldHueSource') && $('fieldHueSource').value === 'Omega' ? 'Omega' : 'omega';
const sliceDeltaInput = () => {
  const el = $('sliceDelta'),
    v = el ? Number(el.value) : 2;
  return Number.isFinite(v) ? Math.max(0, Math.min(64, v)) : 2;
};
const sliceAlignInput = () => {
  const v = $('sliceAlign') ? $('sliceAlign').value : 'center';
  return v === 'left' || v === 'right' ? v : 'center';
};
/* δ / alignment only mean anything for the linear slice */
function syncGridControls() {
  const linear = ($('spiralGrid') && $('spiralGrid').value) === 'linear';
  for (const id of ['sliceDelta', 'sliceAlign']) {
    const el = $(id);
    if (!el) continue;
    el.disabled = !linear;
    el.title = linear ? '' : 'linear slice only';
  }
}
function syncPalette() {
  const num = (id, dflt) => {
    const el = $(id),
      v = el ? Number(el.value) : NaN;
    return Number.isFinite(v) ? v : dflt;
  };
  pal.set({
    omegaMax: Math.max(2, Math.round(num('fieldOmegaMax', 6))),
    gamma: num('fieldGamma', 0.55),
    sat: num('fieldSat', 0.95),
  });
  const legend = $('fieldLegend');
  if (legend) drawFieldLegend(legend, pal, { hueSource: hueSource() });
}

/* --- rendering ------------------------------------------------------ */
function scheduleRender() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {
    renderPending = false;
    renderSpiral();
  });
}

function viewportImageData(ctx, w, h) {
  if (!exactImg || exactImg.width !== w || exactImg.height !== h)
    exactImg = ctx.createImageData(w, h);
  return exactImg;
}

/* square (Ulam): one call — spiral-field.js picks exact vs. mip itself.
   view.cx / view.cy are the cell coordinates under the canvas centre. */
function drawSquareField(ctx, canvas, cellPx, ox, oy) {
  const W = canvas.width,
    H = canvas.height;
  const img = viewportImageData(ctx, W, H);
  const view = { cx: (W / 2 - ox) / cellPx, cy: (H / 2 - oy) / cellPx, scale: cellPx };
  const lvl = renderFieldTo(img, pyr, pal, view, { hueSource: hueSource(), bg: BG_PX });
  ctx.putImageData(img, 0, 0);
  return lvl === 0
    ? `exact field · ${cellPx.toFixed(1)} px/cell`
    : `field mip ${lvl} · ${1 << lvl}×${1 << lvl} cells/texel`;
}

/* ==================== hexagonal grid rendering ======================
      Screen ← axial is the linear map (device px):
        sx = ox + cellPx·(q + r/2 + R + 0.5)
        sy = oy + cellPx·√3/2·(r + R + 0.5)
      Its inverse (pixel → fractional axial) plus cube rounding gives the
      nearest hex, i.e. a crisp hexagonal Voronoi cell per integer; the
      same inverse, divided by the texel size, indexes the pyramid, whose
      axial box the hexagon is stored in.
      ==================================================================== */
function hexClip(canvas, cellPx, ox, oy) {
  const pad = cellPx; // hex corners poke ≈0.15 cell past the box
  const x0 = Math.max(0, Math.floor(ox - pad)),
    y0 = Math.max(0, Math.floor(oy - pad)),
    x1 = Math.min(canvas.width, Math.ceil(ox + spiralExtW * cellPx + pad)),
    y1 = Math.min(canvas.height, Math.ceil(oy + spiralExtH * cellPx + pad));
  return x1 <= x0 || y1 <= y0 ? null : { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}
/* visit every hex whose centre lands inside the pixel rect — the scatter
      direction, so no cell can be missed however small it is */
function forEachHexCentre(cellPx, ox, oy, x0, y0, x1, y1, hit) {
  const R = spiralR,
    pitch = cellPx * HEX_K;
  const rLo = Math.max(-R, Math.ceil((y0 - oy) / pitch - R - 0.5)),
    rHi = Math.min(R, Math.ceil((y1 - oy) / pitch - R - 0.5) - 1);
  for (let r = rLo; r <= rHi; r++) {
    const sy = Math.floor(oy + pitch * (r + R + 0.5));
    if (sy < y0 || sy >= y1) continue;
    const base = ox + cellPx * (r / 2 + R + 0.5); // screen x of q = 0
    const qLo = Math.max(-R, Math.ceil((x0 - base) / cellPx)),
      qHi = Math.min(R, Math.ceil((x1 - base) / cellPx) - 1);
    for (let q = qLo; q <= qHi; q++) {
      const sx = Math.floor(base + cellPx * q);
      if (sx >= x0 && sx < x1) hit(sx, sy, q, r);
    }
  }
}
function drawHexField(ctx, canvas, cellPx, ox, oy) {
  const c = hexClip(canvas, cellPx, ox, oy);
  if (!c) return 'off screen';
  const { x0, y0, x1, y1, w } = c;
  const img = viewportImageData(ctx, c.w, c.h);
  const px = new Uint32Array(img.data.buffer);
  px.fill(BG_PX);
  const R = spiralR,
    inv = 1 / cellPx,
    hs = hueSource();
  /* rows are only √3/2 cell apart: pick the level on *that* scale, so a
        texel is still ≥ 1 device px vertically and no row can be skipped */
  const lvl = pickLevel(pyr, cellPx * HEX_K);
  let label;
  if (lvl === 0) {
    for (let py = y0; py < y1; py++) {
      const rf = ((py + 0.5 - oy) * inv) / HEX_K - (R + 0.5);
      let qf = (x0 + 0.5 - ox) * inv - (R + 0.5) - rf / 2;
      const out = (py - y0) * w - x0;
      for (let p = x0; p < x1; p++, qf += inv) {
        /* cube rounding: round q, r, s = −q−r and repair the worst one */
        let q = Math.round(qf),
          r = Math.round(rf);
        const s = Math.round(-qf - rf);
        if (q + r + s !== 0) {
          const dq = Math.abs(q - qf),
            dr = Math.abs(r - rf),
            ds = Math.abs(s + qf + rf);
          if (dq > dr && dq > ds) q = -r - s;
          else if (dr > ds) r = -q - s;
        }
        const n = cellToN(q + R, r + R);
        if (n >= 0) px[out + p] = pal.packCell(fields, n, hs);
      }
    }
    /* a hex has area √3/2 < 1, so near cellPx = 1 some contain no pixel
          centre; light their centre pixel explicitly — nothing is dropped */
    forEachHexCentre(cellPx, ox, oy, x0, y0, x1, y1, (sx, sy, q, r) => {
      const n = cellToN(q + R, r + R);
      if (n >= 0) px[(sy - y0) * w - x0 + sx] = pal.packCell(fields, n, hs);
    });
    label = `exact hex field · ${cellPx.toFixed(1)} px/cell`;
  } else {
    const L = pyr.levels[lvl],
      texel = 1 << lvl;
    const src = hs === 'Omega' ? L.Om : L.om;
    for (let py = y0; py < y1; py++) {
      const rf = ((py + 0.5 - oy) * inv) / HEX_K - (R + 0.5);
      const ty = Math.floor((rf + R) / texel);
      if (ty < 0 || ty >= L.h) continue;
      const row = ty * L.w,
        out = (py - y0) * w - x0;
      let qf = (x0 + 0.5 - ox) * inv - (R + 0.5) - rf / 2;
      for (let p = x0; p < x1; p++, qf += inv) {
        const tx = Math.floor((qf + R) / texel);
        if (tx < 0 || tx >= L.w) continue;
        const i = row + tx,
          cnt = L.cnt[i];
        if (cnt === 0) continue; // texel entirely outside the hexagon
        px[out + p] = pal.packAvg(src[i] / cnt, L.ra[i] / cnt);
      }
    }
    label = `hex field mip ${lvl} · ${texel}×${texel} cells/texel`;
  }
  ctx.putImageData(img, x0, y0);
  return label;
}

function drawHud(ctx, canvas, dpr, mode) {
  if (!mode) return;
  const hCss = canvas.height / dpr;
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';
  const label = `${spiralLabel || `L = ${fmt(spiralL)}`} · ${mode}`;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = 'rgba(11,15,21,0.75)';
  ctx.fillRect(6, hCss - 23, tw + 12, 17);
  ctx.fillStyle = '#8b97a8';
  ctx.fillText(label, 12, hCss - 11);
  ctx.restore();
}
function tooltipEl() {
  return document.getElementById('spiralTooltip');
}
function hideSpiralTooltip() {
  const el = tooltipEl();
  if (el) el.style.display = 'none';
}
/* pixel → grid cell { gx, gy, n }, the exact inverse of the draw paths.
    n = −1 when the cell carries no integer (outside the spiral/slice) —
    the column read-out still needs gx there, so the cell is returned. */
function spiralCellAt(clientX, clientY) {
  const canvas = spiralCanvas();
  if (!canvas || !pyr) return null;
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;
  if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) return null;
  const dpr = window.devicePixelRatio || 1;
  const cellPx = spiralView.scale * dpr;
  const px = mx * dpr,
    py = my * dpr;
  const ox = spiralView.tx * dpr,
    oy = spiralView.ty * dpr;
  if (spiralMode === 'hex') {
    const R = spiralR;
    const inv = 1 / cellPx;
    const rd = ((py - oy) * inv) / HEX_K - (R + 0.5);
    const qd = (px - ox) * inv - (R + 0.5) - rd / 2;
    let q = Math.round(qd);
    let r = Math.round(rd);
    const s = Math.round(-qd - rd);
    if (q + r + s !== 0) {
      const dq = Math.abs(q - qd);
      const dr = Math.abs(r - rd);
      const ds = Math.abs(s + qd + rd);
      if (dq > dr && dq > ds) q = -r - s;
      else if (dr > ds) r = -q - s;
    }
    return { gx: q + R, gy: r + R, n: cellToN(q + R, r + R) };
  }
  const gx = Math.floor((px - ox) / cellPx);
  const gy = Math.floor((py - oy) / cellPx);
  const W = gridW || spiralL,
    H = gridH || spiralL;
  if (gx < 0 || gy < 0 || gx >= W || gy >= H) return null;
  return { gx, gy, n: cellToN(gx, gy) };
}
/* pixel → integer (null outside) — kept for callers that only want n */
function spiralIntegerAt(clientX, clientY) {
  const c = spiralCellAt(clientX, clientY);
  return !c || c.n < 0 ? null : c.n;
}

function moveSpiralTooltip(clientX, clientY) {
  const canvas = spiralCanvas();
  const el = tooltipEl();
  if (!canvas || !el || !fields) return;
  const cell = spiralCellAt(clientX, clientY);
  const n = cell && cell.n >= 0 ? cell.n : null;
  /* describeCell is three lines: n = … , ω/Ω, sopfr(n)/n.  In the linear
      slice the whole column under the cursor is summarised underneath —
      quadratic n(k) plus min/avg/max of ω, Ω, sopfr along the column. */
  let text = n == null ? '' : describeCell(n, fields);
  if (cell && spiralMode === 'linear') {
    const col = describeColumn(cell.gx);
    if (col) text = text ? `${text}\n${col}` : col;
  }
  if (!text) {
    el.style.display = 'none';
    return;
  }
  const rect = canvas.getBoundingClientRect();
  el.style.whiteSpace = 'pre'; // the column block is column-aligned
  el.style.maxWidth = 'none';
  el.textContent = text;
  el.style.display = 'block';
  let x = clientX - rect.left + 12;
  let y = clientY - rect.top + 12;
  const tw = el.offsetWidth;
  const th = el.offsetHeight;
  if (x + tw > rect.width - 4) x = Math.max(4, clientX - rect.left - tw - 8);
  if (y + th > rect.height - 4) y = Math.max(4, clientY - rect.top - th - 8);
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
}

function renderSpiral() {
  const canvas = spiralCanvas();
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const wCss = Math.max(1, rect.width);
  const hCss = Math.max(1, rect.height);
  if (canvas.width !== Math.round(wCss * dpr) || canvas.height !== Math.round(hCss * dpr)) {
    canvas.width = Math.round(wCss * dpr);
    canvas.height = Math.round(hCss * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0); // work in device pixels
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#0b0f15';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!pyr) return;
  const cellPx = spiralView.scale * dpr; // device px per spiral cell
  const ox = spiralView.tx * dpr, // device px of cell (0,0)
    oy = spiralView.ty * dpr;
  const mode =
    spiralMode === 'hex'
      ? drawHexField(ctx, canvas, cellPx, ox, oy)
      : drawSquareField(ctx, canvas, cellPx, ox, oy);
  drawHud(ctx, canvas, dpr, mode);
  const fi = $('fieldInfo');
  if (fi) fi.textContent = mode || '';
}

function resetSpiralView() {
  const canvas = spiralCanvas();
  if (!canvas || !pyr) return;
  hideSpiralTooltip();
  const rect = canvas.getBoundingClientRect();
  const pad = 12;
  const gw = spiralExtW || spiralL,
    gh = spiralExtH || spiralL;
  const fit = Math.min((rect.width - pad * 2) / gw, (rect.height - pad * 2) / gh);
  spiralView.scale = Math.max(1e-4, fit);
  spiralView.min = spiralView.scale / 4; // allow zooming a bit past "fit"
  spiralView.tx = (rect.width - gw * spiralView.scale) / 2;
  spiralView.ty = (rect.height - gh * spiralView.scale) / 2;
  renderSpiral();
}

/* π(hi) − π(lo−1) over the already-generated list, for the info line */
function countPrimesIn(primes, lo, hi) {
  let c = 0;
  for (let i = 0; i < primes.length; i++) {
    const p = primes[i];
    if (p > hi) break;
    if (p >= lo) c++;
  }
  return c;
}

async function drawUlamSpiral() {
  const info = document.getElementById('spiralInfo');
  const Ninput = Math.max(10, +$('limit').value | 0);
  const w = Math.max(1, Math.min(7, +$('w').value | 0));
  const gridSel = $('spiralGrid') ? $('spiralGrid').value : 'square';
  const grid = gridSel === 'hex' ? 'hex' : gridSel === 'linear' ? 'linear' : 'square';
  const origin = Math.max(0, Math.min(3, +$('spiralOrigin').value | 0));
  const delta = sliceDeltaInput();
  const align = sliceAlignInput();
  /* the walk numbers its cells origin … origin+cells−1, so the truncation
       offset eats into the cell budget that still fits below N */
  const budget = Math.max(4, Ninput - origin + 1);
  let L = 0,
    R = 0,
    Nspiral = 0,
    cells = 0,
    slice = null;
  if (grid === 'hex') {
    /* largest R with H_R = 1 + 3R(R+1) ≤ budget (sqrt then integer repair) */
    R = Math.max(0, Math.floor((Math.sqrt(12 * budget - 3) - 3) / 6));
    while (1 + 3 * (R + 1) * (R + 2) <= budget) R++;
    while (R > 0 && 1 + 3 * R * (R + 1) > budget) R--;
    if (R > MAX_RING) R = MAX_RING;
    if (R < 1) {
      info.textContent = 'N too small for a hex spiral.';
      return;
    }
    cells = 1 + 3 * R * (R + 1);
    Nspiral = origin + cells - 1; // largest n on the plot
  } else if (grid === 'linear') {
    /* rows are grown one at a time: the widths are ⌊1 + δk⌋ and the
          running total is exactly the cell count, so no repair pass */
    slice = buildSlice(budget, delta);
    if (!slice || slice.rows < 2) {
      info.textContent = 'N too small for a linear slice.';
      return;
    }
    cells = slice.cells;
    Nspiral = origin + cells - 1;
  } else {
    L = Math.floor(Math.sqrt(budget));
    if (L > MAX_SIDE) L = MAX_SIDE;
    if (L < 2) {
      info.textContent = 'N too small for a Ulam spiral.';
      return;
    }
    cells = L * L; // the spiral can only ever show L² integers
    Nspiral = origin + cells - 1;
  }
  /* the field needs every prime ≤ Nspiral (large factors drive sopfr) */
  const need = Math.max(2, Nspiral);
  let primes;
  if (lastSpiral && lastSpiral.N >= need) {
    primes = lastSpiral.primes; // the output is algorithm-independent
  } else {
    const algo = currentAlgo();
    info.textContent = `generating primes ≤ ${fmt(need)} with Algorithm ${algo.meta.id}…`;
    await sleep(0);
    const res = algo.run(need, { w });
    primes = res.primes;
    lastSpiral = { primes, N: need };
  }

  /* --- layout ----------------------------------------------------- */
  spiralMode = grid;
  spiralOrigin = origin;
  spiralCells = cells;
  spiralLast = Nspiral;
  let gw, gh;
  if (grid === 'hex') {
    spiralR = R;
    spiralL = 2 * R + 1; // axial box side
    gw = gh = spiralL;
    spiralExtW = spiralL; // X ∈ [0, D]
    spiralExtH = spiralL * HEX_K; // Y ∈ [0, D·√3/2]
    spiralLabel = `hex R = ${fmt(R)} · n₀ = ${origin}`;
  } else if (grid === 'linear') {
    sliceRowStart = slice.starts;
    sliceRowW = slice.widths;
    sliceAlign = align;
    sliceDelta = delta;
    spiralR = 0;
    spiralL = slice.maxW; // widest row = bounding-box width
    gw = slice.maxW;
    gh = slice.rows;
    spiralExtW = gw;
    spiralExtH = gh;
    spiralLabel = `slice δ = ${fmt(delta)} · ${fmt(slice.rows)} rows · ${align} · n₀ = ${origin}`;
  } else {
    spiralR = 0;
    spiralL = L;
    gw = gh = L;
    spiralExtW = L;
    spiralExtH = L;
    spiralLabel = `L = ${fmt(L)} · n₀ = ${origin}`;
  }
  gridW = gw; // cellToN / hit-testing need the box before the closure
  gridH = gh;
  assertExact(gw * gh, 'grid cells');
  if (gw * gh > FIELD_MAX_CELLS) {
    info.textContent = `grid of ${fmt(gw * gh)} cells exceeds FIELD_MAX_CELLS (${fmt(FIELD_MAX_CELLS)}).`;
    return;
  }
  cellToN = makeCellToN();
  sliceColCache = { key: null, text: '' }; // layout changed → column stats stale

  /* --- field + pyramid -------------------------------------------- */
  info.textContent = `factorising n ≤ ${fmt(Nspiral)} (ω, Ω, sopfr)…`;
  await sleep(0);
  fields = computeFactorFields(Nspiral, primes);
  info.textContent = `building the field pyramid (${fmt(gw)}×${fmt(gh)} cells)…`;
  await sleep(0);
  pyr = buildFieldPyramid(gw, gh, cellToN, fields);

  syncPalette();
  resetSpiralView();
  const pi = countPrimesIn(primes, Math.max(2, spiralOrigin), spiralLast);
  info.textContent =
    `${fmt(cells)} cells · ${fmt(pi)} primes · ` +
    (grid === 'hex'
      ? `${fmt(R)} rings (axial box ${fmt(spiralL)}²) · `
      : grid === 'linear'
        ? `${fmt(gridH)} rows · δ = ${fmt(sliceDelta)} · widest ${fmt(gridW)} · ${sliceAlign} · `
        : `side ${fmt(L)} · `) +
    `n ∈ [${fmt(spiralOrigin)}, ${fmt(spiralLast)}] · ` +
    `max ω = ${fmt(fields.omegaMax)} · ${pyr.levels.length - 1} mip levels · ` +
    `hue = ω(n), lightness = sopfr(n)/n · drag to pan, wheel to zoom`;
}

function initSpiralInteractions() {
  const canvas = spiralCanvas();
  if (!canvas) return;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    hideSpiralTooltip();
    canvas.classList.add('dragging');
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) {
      moveSpiralTooltip(e.clientX, e.clientY);
      return;
    }
    hideSpiralTooltip();
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    spiralView.tx += dx;
    spiralView.ty += dy;
    scheduleRender(); // coalesce to one draw per frame
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    canvas.classList.remove('dragging');
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointercancel', () => {
    dragging = false;
    canvas.classList.remove('dragging');
  });
  canvas.addEventListener('pointerleave', () => {
    hideSpiralTooltip();
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = Math.pow(1.0015, -e.deltaY);
      const oldScale = spiralView.scale;
      spiralView.scale = Math.max(spiralView.min, Math.min(400, oldScale * factor));
      const sx = mx - spiralView.tx;
      const sy = my - spiralView.ty;
      const nx = sx * (spiralView.scale / oldScale);
      const ny = sy * (spiralView.scale / oldScale);
      spiralView.tx = mx - nx;
      spiralView.ty = my - ny;
      scheduleRender();
    },
    { passive: false }
  );
  window.addEventListener('resize', () => {
    if (pyr) resetSpiralView();
  });
}

/* =====================================================================
minimal UI
===================================================================== */
const $ = (id) => document.getElementById(id);
const sleep = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const openLog = () => {
  const d = $('outWrap');
  if (d) d.open = true; // never write into a collapsed panel silently
};
const fmt = (n) =>
  typeof n === 'number' && isFinite(n)
    ? Number.isInteger(n)
      ? n.toLocaleString('en-US')
      : n.toFixed(4)
    : String(n);

function panel(id) {
  const el = $(id);
  return {
    clear() {
      el.innerHTML = '';
      el.classList.remove('dim');
    },
    line(text = '', cls) {
      const s = document.createElement('span');
      if (cls) s.className = cls;
      s.textContent = text + '\n';
      el.appendChild(s);
    },
    head(t) {
      this.line(t, 'head');
    },
    dim(t) {
      this.line(t, 'dim');
    },
  };
}

async function generate() {
  const P = panel('out'),
    badge = $('badge');
  P.clear();
  openLog();
  badge.innerHTML = '';
  const algo = currentAlgo();
  const N = Math.max(10, +$('limit').value | 0);
  const w = Math.max(1, Math.min(7, +$('w').value | 0));
  P.head(`${algo.meta.title}   [${algo.meta.ref}]`);
  P.line(`N = ${fmt(N)}${algo.meta.usesWheel ? `   w = ${w}` : '   (no wheel)'}`);
  await sleep(0);

  const res = algo.run(N, { w });
  const { primes, stats: st } = res;
  lastSpiral = { primes, N };

  P.line('');
  P.line(`primes emitted   : ${fmt(primes.length)}   (last = ${fmt(primes[primes.length - 1])})`);
  P.line(`first primes     : ${primes.slice(0, 15).join(' ')} …`);
  P.line(
    `time             : ${st.ms.toFixed(1)} ms   ` +
      `(${(N / st.ms / 1000).toFixed(2)} M integers/s, binary heap)`
  );
  P.line('');
  /* algorithm-specific report lives with the algorithm */
  for (const [text, cls] of algo.summary(res, N)) P.line(text, cls);

  if ($('verify').checked && N <= 3_000_000) {
    await sleep(0);
    const ref = sieveRef(N);
    const ok = ref.length === primes.length && ref.every((v, i) => v === primes[i]);
    P.line('');
    P.line(
      `vs Eratosthenes  : ${ok ? 'identical, ' + fmt(ref.length) + ' primes' : 'MISMATCH'}`,
      ok ? 'pass' : 'fail'
    );
    badge.innerHTML = ok
      ? '<span class="badge">verified</span>'
      : '<span class="badge bad">mismatch</span>';
  } else if ($('verify').checked) {
    P.line('');
    P.dim('verification skipped (N > 3·10⁶)');
  }
}

function firstPrimes() {
  const P = panel('out');
  $('badge').innerHTML = '';
  P.clear();
  openLog();
  const algo = currentAlgo();
  const w = Math.max(1, Math.min(7, +$('w').value | 0));
  P.head(
    `first 50 primes from the unbounded generator ` +
      `${algo.meta.id}.stream(${algo.meta.usesWheel ? `{ w: ${w} }` : '{}'})`
  );
  const g = algo.stream({ w }),
    got = [];
  for (let i = 0; i < 50; i++) got.push(g.next().value);
  P.line(got.join(' '));
  P.line('');
  const ref = sieveRef(300);
  const ok = got.every((v, i) => v === ref[i]);
  P.line(`vs Eratosthenes : ${ok ? 'identical' : 'MISMATCH'}`, ok ? 'pass' : 'fail');
  P.dim(algo.meta.streamNote);
}
function syncAlgo() {
  const algo = currentAlgo();
  $('algoNote').textContent = `${algo.meta.ref} — ${algo.meta.blurb}`;
  $('w').disabled = !algo.meta.usesWheel;
  $('w').title = algo.meta.usesWheel ? '' : 'Algorithm A uses no wheel';
}

$('btnRun').onclick = () => generate();
$('btnFirst').onclick = firstPrimes;
$('btnSpiral').onclick = () => drawUlamSpiral();
$('btnSpiralReset').onclick = () => resetSpiralView();
$('algo').onchange = () => {
  syncAlgo();
  firstPrimes(); // cheap, and shows the selected module actually running
};
$('spiralGrid').onchange = () => {
  syncGridControls();
  if (pyr) drawUlamSpiral(); // re-walk with the other geometry
};
$('spiralOrigin').onchange = () => {
  if (pyr) drawUlamSpiral(); // re-walk from the new origin
};
/* δ and the row alignment change the *layout*, so the walk (and with it
    the field pyramid) has to be rebuilt — same cost as "draw spiral" */
for (const id of ['sliceDelta', 'sliceAlign']) {
  const el = $(id);
  if (el)
    el.addEventListener('change', () => {
      if (pyr && spiralMode === 'linear') drawUlamSpiral();
    });
}
/* palette controls: rebuild the LUT + legend, redraw from the same
   pyramid — no re-factorisation, so this is interactive at any N */
for (const id of ['fieldHueSource', 'fieldOmegaMax', 'fieldGamma', 'fieldSat']) {
  const el = $(id);
  if (el)
    el.addEventListener('input', () => {
      syncPalette();
      if (pyr) scheduleRender();
    });
}
initSpiralInteractions();
syncAlgo();
syncGridControls();
syncPalette(); /* legend is meaningful before the first spiral */

firstPrimes(); /* cheap initial render */
