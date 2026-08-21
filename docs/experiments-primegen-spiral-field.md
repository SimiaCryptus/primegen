# Wiring the factorisation field into `js/experiments-primegen-generator.js`

## 1. Delete the twin-prime path

Remove, in the spiral section of the driver:

- the `twinBits` bitset and everything that fills it (`p+2` / `p-2` lookups);
- the second channel of the old density mip (`twinCount`) and the red/white
  hue mix at draw time;
- the `#ff7b72` constant and any `isTwin(n)` helper.

The prime bitset itself may also go: the field arrays subsume it
(`omega[n] === 1 && Omega[n] === 1` ⇔ `n` prime, and `lum[n] === 255` ⇔
prime or 4).

## 2. Import and build

```js
import {
  computeFactorFields,
  buildFieldPyramid,
  renderFieldTo,
  drawFieldLegend,
  describeCell,
  FieldPalette,
  FIELD_MAX_CELLS,
} from './spiral-field.js';

const pal = new FieldPalette();
let fields = null,
  pyr = null;

// after the generator has produced `primes` (all primes ≤ Nspiral):
function buildSpiral(Nspiral, gridW, gridH, cellToN) {
  if (gridW * gridH > FIELD_MAX_CELLS) throw new RangeError('spiral too large');
  fields = computeFactorFields(Nspiral, primes);
  pyr = buildFieldPyramid(gridW, gridH, cellToN, fields);
  pal.set({ omegaMax: Number(fieldOmegaMax.value) });
  drawFieldLegend(fieldLegend, pal, { hueSource: fieldHueSource.value });
}
```

`cellToN(x, y)` is the _existing_ spiral layout function — square (Ulam) or
axial hex — and must return `-1` for cells outside the spiral or `> N`.
Nothing else about the layout changes.
In the driver both layouts are now **closed form** (`squareIndex`,
`hexIndex`) rather than a walk that fills an array: `buildFieldPyramid`
streams over the grid, so no per-cell storage is needed at all, and the
same closure is reused by the tooltip's inverse hit-test.
`MAX_SIDE` drops to **4096**, because `L² = 2^24` is exactly
`FIELD_MAX_CELLS`; the hex cap follows as `MAX_RING = (MAX_SIDE-1) >> 1`.
The `+2` on the prime budget disappears with the twins: the field only
needs the primes ≤ `Nspiral`.

## 3. Draw

Replace the old bitset/mip blit with one call; `view.scale` is device pixels
per cell and `view.cx / view.cy` are the cell coordinates under the canvas
centre (the same pan/zoom state as before).

```js
function draw() {
  const img = ctx.createImageData(canvas.width, canvas.height);
  const lvl = renderFieldTo(img, pyr, pal, view, { hueSource: fieldHueSource.value });
  ctx.putImageData(img, 0, 0);
  fieldInfo.textContent = lvl === 0 ? 'exact cells' : `mip level ${lvl} (${1 << lvl} cells/texel)`;
}
```

That is the square path verbatim (with `cx = (W/2 − ox)/cellPx`,
`cy = (H/2 − oy)/cellPx` derived from the existing `tx/ty` pan state and a
cached `ImageData`). **Hex** keeps its own loop, because `renderFieldTo`
samples an axis-aligned grid and the hex lattice is sheared: the driver
inverts `sx = ox + cellPx·(q + r/2 + R + ½)`, `sy = oy + cellPx·√3/2·(r + R + ½)`
per pixel and then either cube-rounds to a cell (`pal.packCell`, plus a
scatter pass over hex centres so no integer vanishes near 1 px/cell) or
indexes `pyr.levels[lvl]` and averages (`pal.packAvg`). The level is
chosen with `pickLevel(pyr, cellPx·√3/2)` — the row pitch, not the column
pitch — so a texel stays ≥ 1 device pixel tall.

## 4. Controls and tooltip

```js
for (const el of [fieldOmegaMax, fieldGamma, fieldSat, fieldHueSource]) {
  el.addEventListener('input', () => {
    pal.set({
      omegaMax: Number(fieldOmegaMax.value),
      gamma: Number(fieldGamma.value),
      sat: Number(fieldSat.value),
    });
    drawFieldLegend(fieldLegend, pal, { hueSource: fieldHueSource.value });
    draw();
  });
}

// hover: n comes from the same inverse layout used before
spiralTooltip.textContent = n >= 0 ? describeCell(n, fields) : '';
```

`describeCell` prints `n = 12 = 2^2·3`, `ω`, `Ω` and `sopfr(n)/n`, so the
colour under the cursor is always explainable.
