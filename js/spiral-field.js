/* =====================================================================
   spiral-field.js — the factorisation field plotted on the number spiral

     computeFactorFields(N, primes)   ω(n), Ω(n) and sopfr(n)/n for all n ≤ N
     FieldPalette                     (ω, sopfr/n) → HSL → RGB, via a LUT
     buildFieldPyramid(...)           area-correct sum pyramid over grid cells
     renderFieldTo(img, ...)          exact sampling zoomed in, mip zoomed out
     drawFieldLegend(canvas, pal)     2-D key of the colour map
     describeCell(n, fields)          tooltip text

   Colour semantics
     hue        = ω(n)  (or Ω(n)), the number of *distinct* prime factors:
                  ω = 1 → hue0 (gold), rising ω sweeps green → cyan → blue
                  → magenta.  Prime powers stay on the prime hue, which is
                  exactly what "number of unique primes" means.
     lightness  = sopfr(n)/n, the sum of the prime factors *with*
                  multiplicity divided by n.  sopfr(n) ≤ n with equality iff
                  n is prime (and n = 4), so primes sit at lightness 1;
                  n = 2p at 1/2, n = 3p at 1/3, n = 4p at 3/(4p) … .

   No DOM state, no network.  Depends only on primegen-core.js.
   ===================================================================== */

import { sieveRef } from './primegen-core.js';

/* A spiral of C cells needs 3·C bytes of field storage (ω, Ω, quantised
   sopfr/n) plus ~1.5·C·4 bytes of pyramid.  Keep that honest. */
export const FIELD_MAX_CELLS = 1 << 24; // 16 777 216

/* ImageData is byte-addressed; a Uint32 view is ~4× faster to fill. ------ */
const LITTLE_ENDIAN = (() => {
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = 0x01020304;
  return new Uint8Array(buf)[0] === 0x04;
})();

export function packRGBA(r, g, b, a = 255) {
  return LITTLE_ENDIAN
    ? ((a << 24) | (b << 16) | (g << 8) | r) >>> 0
    : ((r << 24) | (g << 16) | (b << 8) | a) >>> 0;
}

export const TRANSPARENT = 0;

/* ---------------------------------------------------------------------
   ω(n), Ω(n), sopfr(n)/n  for every n ≤ N, in blocks.

   For each prime p ≤ N and every multiple m of p:  ω += 1, Ω += 1, sopfr += p.
   For each higher power p^k (k ≥ 2) and every multiple of p^k: Ω += 1,
   sopfr += p — the standard "one pass per exponent level" trick, total work
   O(N log log N).  Blocking keeps the Float64 accumulator at O(B) instead of
   O(N); the two Uint8 outputs are the only O(N) allocations.

   sopfr(n)/n ∈ (0,1] is stored quantised to 0…255 — a 1/255 step is far
   below what any pixel can show, and it halves the memory.
   ------------------------------------------------------------------ */
export function computeFactorFields(N, primes = null, opts = {}) {
  const { block = 1 << 20, onProgress = null } = opts;
  if (!Number.isSafeInteger(N) || N < 1) throw new RangeError(`computeFactorFields: bad N = ${N}`);

  /* all primes ≤ N are needed (large prime factors matter for sopfr) */
  const ps = primes && primes.length && primes[primes.length - 1] >= N ? primes : sieveRef(N);

  const omega = new Uint8Array(N + 1); // distinct prime factors
  const Omega = new Uint8Array(N + 1); // with multiplicity
  const lum = new Uint8Array(N + 1); // round(255 · sopfr(n)/n)

  const B = Math.min(block, N + 1);
  const sop = new Float64Array(B);
  const om = new Uint8Array(B);
  const Om = new Uint8Array(B);
  let omegaMax = 0;

  for (let lo = 0; lo <= N; lo += B) {
    const hi = Math.min(lo + B, N + 1); // exclusive
    const len = hi - lo;
    sop.fill(0, 0, len);
    om.fill(0, 0, len);
    Om.fill(0, 0, len);

    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      if (p >= hi) break;
      let m = Math.max(p, Math.ceil(lo / p) * p);
      for (; m < hi; m += p) {
        const j = m - lo;
        om[j]++;
        Om[j]++;
        sop[j] += p;
      }
      for (let pk = p; pk <= (hi - 1) / p; ) {
        pk *= p; // p^2, p^3, …  (never leaves the block range)
        let q = Math.max(pk, Math.ceil(lo / pk) * pk);
        for (; q < hi; q += pk) {
          const j = q - lo;
          Om[j]++;
          sop[j] += p;
        }
      }
    }

    for (let j = 0, n = lo; n < hi; j++, n++) {
      omega[n] = om[j];
      Omega[n] = Om[j];
      if (om[j] > omegaMax) omegaMax = om[j];
      lum[n] = n > 1 ? Math.min(255, Math.round((255 * sop[j]) / n)) : 0;
    }
    if (onProgress) onProgress(hi - 1, N);
  }

  return { N, omega, Omega, lum, omegaMax: Math.max(1, omegaMax) };
}

/* ---------------------------------------------------------------------
   palette:  (ω̄, (sopfr/n)‾) → RGB, precomputed into a LUT
   ------------------------------------------------------------------ */
const HUE_STEPS = 128;
const LUM_STEPS = 256;

function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0,
    g = 0,
    b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

export class FieldPalette {
  constructor(opts = {}) {
    this.omegaMax = opts.omegaMax ?? 6; // ω mapped onto the full hue span
    this.hue0 = opts.hue0 ?? 48; // ω ≤ 1  → gold  (primes, prime powers)
    this.hueSpan = opts.hueSpan ?? 300; // → green → cyan → blue → magenta
    this.sat = opts.sat ?? 0.95;
    this.gamma = opts.gamma ?? 0.55; // lightness = lightMax·(sopfr/n)^γ
    this.lightMax = opts.lightMax ?? 0.62;
    this.build();
  }
  set(opts) {
    Object.assign(this, opts);
    this.build();
    return this;
  }
  build() {
    const lut = new Uint32Array(HUE_STEPS * LUM_STEPS);
    for (let hi = 0; hi < HUE_STEPS; hi++) {
      const t = hi / (HUE_STEPS - 1);
      const hue = this.hue0 + this.hueSpan * t;
      for (let li = 0; li < LUM_STEPS; li++) {
        const ratio = li / (LUM_STEPS - 1);
        const L = this.lightMax * Math.pow(ratio, this.gamma);
        /* desaturate the very darkest end so numerical dust stays neutral */
        const S = this.sat * Math.min(1, L / 0.06);
        const [r, g, b] = hslToRgb(hue, S, L);
        lut[hi * LUM_STEPS + li] = packRGBA(r, g, b, 255);
      }
    }
    this.lut = lut;
  }
  /* ω̄ may be fractional (mip average); ratio ∈ [0,1] */
  packAvg(omegaAvg, ratio) {
    const t = this.omegaMax > 1 ? (omegaAvg - 1) / (this.omegaMax - 1) : 0;
    const hi = Math.max(0, Math.min(HUE_STEPS - 1, Math.round(t * (HUE_STEPS - 1))));
    const li = Math.max(0, Math.min(LUM_STEPS - 1, Math.round(ratio * (LUM_STEPS - 1))));
    return this.lut[hi * LUM_STEPS + li];
  }
  /* exact, un-averaged cell — the quantised sopfr/n is already the LUT row */
  packCell(fields, n, hueSource = 'omega') {
    const w = hueSource === 'Omega' ? fields.Omega[n] : fields.omega[n];
    const t = this.omegaMax > 1 ? (w - 1) / (this.omegaMax - 1) : 0;
    const hi = Math.max(0, Math.min(HUE_STEPS - 1, Math.round(t * (HUE_STEPS - 1))));
    return this.lut[hi * LUM_STEPS + fields.lum[n]];
  }
}

/* ---------------------------------------------------------------------
   mip pyramid of *sums* over grid cells

   levels[0] is deliberately null: level 0 is the exact field, sampled
   straight out of `fields` through `cellToN`, so nothing is ever dropped by
   nearest-neighbour downscaling.  levels[k] (k ≥ 1) has one texel per
   2^k × 2^k cells and keeps
         cnt  Σ1 over cells that actually carry an n
         om   Σ ω(n)
         Om   Σ Ω(n)
         ra   Σ sopfr(n)/n
   Averaging happens at draw time, RGB conversion after that — hue and
   brightness therefore stay area-correct at every level.
   ------------------------------------------------------------------ */
function newLevel(w, h, f64) {
  const A = f64 ? Float64Array : Float32Array;
  return { w, h, cnt: new A(w * h), om: new A(w * h), Om: new A(w * h), ra: new A(w * h) };
}

export function buildFieldPyramid(width, height, cellToN, fields) {
  const cells = width * height;
  if (!Number.isFinite(cells) || cells > FIELD_MAX_CELLS)
    throw new RangeError(
      `spiral of ${width}×${height} = ${cells} cells exceeds FIELD_MAX_CELLS (${FIELD_MAX_CELLS})`
    );

  const w1 = Math.max(1, Math.ceil(width / 2)),
    h1 = Math.max(1, Math.ceil(height / 2));
  const L1 = newLevel(w1, h1, false); // biggest level — keep it Float32
  const { omega, Omega, lum } = fields;
  const INV = 1 / 255;

  for (let y = 0; y < height; y++) {
    const row = (y >> 1) * w1;
    for (let x = 0; x < width; x++) {
      const n = cellToN(x, y);
      if (n < 0) continue; // hole in the axial hex grid / beyond N
      const i = row + (x >> 1);
      L1.cnt[i] += 1;
      L1.om[i] += omega[n];
      L1.Om[i] += Omega[n];
      L1.ra[i] += lum[n] * INV;
    }
  }

  const levels = [null, L1];
  let cur = L1;
  while (cur.w > 1 || cur.h > 1) {
    const nw = Math.max(1, Math.ceil(cur.w / 2)),
      nh = Math.max(1, Math.ceil(cur.h / 2));
    const nx = newLevel(nw, nh, true);
    for (let y = 0; y < cur.h; y++) {
      const src = y * cur.w,
        dst = (y >> 1) * nw;
      for (let x = 0; x < cur.w; x++) {
        const i = src + x,
          j = dst + (x >> 1);
        nx.cnt[j] += cur.cnt[i];
        nx.om[j] += cur.om[i];
        nx.Om[j] += cur.Om[i];
        nx.ra[j] += cur.ra[i];
      }
    }
    levels.push(nx);
    cur = nx;
  }
  return { base: { width, height, cellToN, fields }, levels };
}

/* scale = device pixels per grid cell → coarsest level still ≥ 1 px/texel */
export function pickLevel(pyr, scale) {
  if (scale >= 1) return 0;
  const k = Math.ceil(Math.log2(1 / scale));
  return Math.max(0, Math.min(pyr.levels.length - 1, k));
}

/* ---------------------------------------------------------------------
   render — view = { cx, cy, scale }: cell coordinates at the canvas centre
   and device pixels per cell.
   ------------------------------------------------------------------ */
export function renderFieldTo(img, pyr, pal, view, opts = {}) {
  const { hueSource = 'omega', bg = TRANSPARENT } = opts;
  const px = new Uint32Array(img.data.buffer);
  const W = img.width,
    H = img.height;
  const { cx, cy, scale } = view;
  const lvl = pickLevel(pyr, scale);
  px.fill(bg);

  if (lvl === 0) {
    const { width: gw, height: gh, cellToN, fields } = pyr.base;
    for (let y = 0; y < H; y++) {
      const gy = Math.floor(cy + (y + 0.5 - H / 2) / scale);
      if (gy < 0 || gy >= gh) continue;
      const out = y * W;
      for (let x = 0; x < W; x++) {
        const gx = Math.floor(cx + (x + 0.5 - W / 2) / scale);
        if (gx < 0 || gx >= gw) continue;
        const n = cellToN(gx, gy);
        if (n < 0) continue;
        px[out + x] = pal.packCell(fields, n, hueSource);
      }
    }
    return lvl;
  }

  const L = pyr.levels[lvl];
  const texel = 1 << lvl; // cells per texel
  const src = hueSource === 'Omega' ? L.Om : L.om;
  for (let y = 0; y < H; y++) {
    const ty = Math.floor((cy + (y + 0.5 - H / 2) / scale) / texel);
    if (ty < 0 || ty >= L.h) continue;
    const row = ty * L.w,
      out = y * W;
    for (let x = 0; x < W; x++) {
      const tx = Math.floor((cx + (x + 0.5 - W / 2) / scale) / texel);
      if (tx < 0 || tx >= L.w) continue;
      const i = row + tx,
        c = L.cnt[i];
      if (c === 0) continue;
      px[out + x] = pal.packAvg(src[i] / c, L.ra[i] / c);
    }
  }
  return lvl;
}

/* ---------------------------------------------------------------------
   legend — ω on x, sopfr(n)/n on y, with the landmark rays marked
   ------------------------------------------------------------------ */
export function drawFieldLegend(canvas, pal, opts = {}) {
  const { hueSource = 'omega' } = opts;
  const ctx = canvas.getContext('2d');
  const W = canvas.width,
    H = canvas.height;
  const padL = 38,
    padR = 8,
    padT = 8,
    padB = 20;
  const gw = Math.max(1, W - padL - padR),
    gh = Math.max(1, H - padT - padB);

  ctx.clearRect(0, 0, W, H);
  const img = ctx.createImageData(gw, gh);
  const px = new Uint32Array(img.data.buffer);
  for (let y = 0; y < gh; y++) {
    const ratio = 1 - y / (gh - 1);
    for (let x = 0; x < gw; x++) {
      px[y * gw + x] = pal.packAvg((x / (gw - 1)) * pal.omegaMax, ratio);
    }
  }
  ctx.putImageData(img, padL, padT);

  ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = '#9aa4b2';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const [v, txt] of [
    [1, '1'],
    [0.5, '1/2'],
    [1 / 3, '1/3'],
    [0, '0'],
  ]) {
    const y = padT + (1 - v) * (gh - 1);
    ctx.fillText(txt, padL - 5, y);
  }
  ctx.save();
  ctx.translate(10, padT + gh / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('sopfr(n)/n', 0, 0);
  ctx.restore();

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const label = hueSource === 'Omega' ? 'Ω(n)' : 'ω(n)';
  for (let k = 1; k <= pal.omegaMax; k++) {
    const t = pal.omegaMax > 1 ? (k - 1) / (pal.omegaMax - 1) : 0;
    ctx.fillText(String(k), padL + t * (gw - 1), padT + gh + 3);
  }
  ctx.fillText(label, padL + gw / 2, padT + gh + 3 + 0); // axis name at centre-bottom
  ctx.textAlign = 'left';
  ctx.fillStyle = '#c9d1d9';
  ctx.fillText('primes → top-left (ω=1, ratio 1)', padL, padT + gh + 3);
}

/* ---------------------------------------------------------------------
   tooltip text — trial division is fine for a single hovered cell
   ------------------------------------------------------------------ */
export function factorString(n) {
  if (n < 2) return String(n);
  const parts = [];
  let m = n;
  for (let p = 2; p * p <= m; p += p === 2 ? 1 : 2) {
    if (m % p) continue;
    let e = 0;
    while (m % p === 0) {
      m /= p;
      e++;
    }
    parts.push(e === 1 ? `${p}` : `${p}^${e}`);
  }
  if (m > 1) parts.push(String(m));
  return parts.join('·');
}

export function describeCell(n, fields) {
  if (n < 2) return `n = ${n}`;
  const ratio = fields.lum[n] / 255;
  const w = fields.omega[n],
    O = fields.Omega[n];
  const kind = w === 1 && O === 1 ? ' — prime' : w === 1 ? ' — prime power' : '';
  return (
    `n = ${n} = ${factorString(n)}${kind}\n` +
    `ω = ${w}   Ω = ${O}\n` +
    `sopfr(n)/n ≈ ${ratio.toFixed(3)}`
  );
}
