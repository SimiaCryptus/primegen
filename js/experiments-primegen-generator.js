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
  const L = spiralL;
  return (x, y) => {
    const i = squareIndex(x, y, L);
    return i >= 1 && i <= cells ? i + first - 1 : -1;
  };
}

/* --- palette / legend controls -------------------------------------- */
const hueSource = () =>
  $('fieldHueSource') && $('fieldHueSource').value === 'Omega' ? 'Omega' : 'omega';
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
/* pixel → integer, the exact inverse of the two draw paths */
function spiralIntegerAt(clientX, clientY) {
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
  let n;
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
    n = cellToN(q + R, r + R);
  } else {
    const gx = Math.floor((px - ox) / cellPx);
    const gy = Math.floor((py - oy) / cellPx);
    if (gx < 0 || gy < 0 || gx >= spiralL || gy >= spiralL) return null;
    n = cellToN(gx, gy);
  }
  return n < 0 ? null : n;
}

function moveSpiralTooltip(clientX, clientY) {
  const canvas = spiralCanvas();
  const el = tooltipEl();
  if (!canvas || !el || !fields) return;
  const n = spiralIntegerAt(clientX, clientY);
  if (n == null) {
    el.style.display = 'none';
    return;
  }
  const rect = canvas.getBoundingClientRect();
  /* describeCell is three lines: n = … , ω/Ω, sopfr(n)/n */
  el.style.whiteSpace = 'pre-line';
  el.textContent = describeCell(n, fields);
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
  const grid = $('spiralGrid').value === 'hex' ? 'hex' : 'square';
  const origin = Math.max(0, Math.min(3, +$('spiralOrigin').value | 0));
  /* the walk numbers its cells origin … origin+cells−1, so the truncation
       offset eats into the cell budget that still fits below N */
  const budget = Math.max(4, Ninput - origin + 1);
  let L = 0,
    R = 0,
    Nspiral = 0,
    cells = 0;
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
  } else {
    spiralR = 0;
    spiralL = L;
    gw = gh = L;
    spiralExtW = L;
    spiralExtH = L;
    spiralLabel = `L = ${fmt(L)} · n₀ = ${origin}`;
  }
  assertExact(gw * gh, 'grid cells');
  if (gw * gh > FIELD_MAX_CELLS) {
    info.textContent = `grid of ${fmt(gw * gh)} cells exceeds FIELD_MAX_CELLS (${fmt(FIELD_MAX_CELLS)}).`;
    return;
  }
  cellToN = makeCellToN();

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
    (grid === 'hex' ? `${fmt(R)} rings (axial box ${fmt(spiralL)}²) · ` : `side ${fmt(L)} · `) +
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
  if (pyr) drawUlamSpiral(); // re-walk with the other geometry
};
$('spiralOrigin').onchange = () => {
  if (pyr) drawUlamSpiral(); // re-walk from the new origin
};
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
syncPalette(); /* legend is meaningful before the first spiral */

firstPrimes(); /* cheap initial render */
