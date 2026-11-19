// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// A copy of the Vortex design system's components/lib/ascii-core.js. Edit the system, then copy it again.

// The rules of the ASCII mark and the logo banner, kept free of the DOM so they can be tested.

/** Characters from empty to dense, mapped from luminance 0..1. */
export const RAMP = " .:-=+*#%@";
const HEX = "0123456789abcdef";

/** The ramp character for luminance `v`, clamped to 0..1. */
export function rampChar(v) {
  const x = Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));
  return RAMP[Math.round(x * (RAMP.length - 1))];
}

/** 1 inside the solid core, 0 at and beyond `reach`, linear between. */
export function falloff(d, reach, core = 0.3) {
  const c = reach * core;
  if (d <= c) return 1;
  if (d >= reach) return 0;
  return 1 - (d - c) / (reach - c);
}

/** The strongest brush mark at cell centre (cx, cy). Each mark is {x, y, reach, k}. */
export function brushStrength(cx, cy, marks) {
  let s = 0;
  for (const m of marks) s = Math.max(s, falloff(Math.hypot(cx - m.x, cy - m.y), m.reach) * m.k);
  return s;
}

/** A hex digit for a cell that flickers with `tick`. Always one of 0-9a-f, for any integers. */
export function hexDigit(row, col, tick) {
  let h = Math.imul((row | 0) * 4096 + (col | 0), 2654435761) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  const speed = (h >>> 8) % 3 + 1;
  return HEX[(((h + Math.imul(tick | 0, speed)) >>> 0) % 16)];
}

/** How many copies of a logo set the banner needs so that two loops cover the view, always even. */
export function copiesNeeded(setWidth, viewWidth) {
  const n = Math.max(2, Math.ceil((2 * viewWidth) / Math.max(1, setWidth)));
  return n % 2 ? n + 1 : n;
}

/**
 * The camera's vertical field of view (degrees) for a host of `aspect` (width / height). The mark
 * is framed for a square: a wide host keeps `base` and shows more around it, and a tall host opens
 * the vertical field until the horizontal one equals `base`, so the mark's sides are never cropped.
 */
export function fitFov(base, aspect) {
  if (aspect >= 1) return base;
  return (2 * Math.atan(Math.tan((base * Math.PI) / 360) / aspect) * 180) / Math.PI;
}

/**
 * The tokens the mark draws with in a colour scheme ("dark" or "light"; anything else is Dark):
 * four inks that step away from the ground as the lit faces brighten, the field's dot, and the
 * brush's outer and inner colours. Light mirrors Dark's steps of contrast against its own ground.
 */
export function markPalette(scheme) {
  const light = scheme === "light";
  return {
    inks: light ? ["--grey-400", "--grey-450", "--grey-600", "--black"] : ["--grey-600", "--grey-500", "--grey-400", "--grey-50"],
    field: light ? "--grey-300" : "--grey-700",
    // The trail's edge, then its lighter core. Light's accent and accent-ink are one Blue, so the
    // brush names the scale to keep two tones in both themes.
    brush: light ? ["--blue", "--blue-bright"] : ["--blue-bright", "--blue-soft"],
  };
}

/**
 * Reads the mark's render back for the character grid without stalling the GPU every frame.
 * `render()` draws the scene into the render target. `readSync(buf)` reads it into `buf` at once,
 * so the page waits for the GPU. `readAsync(buf)`, when the renderer has one (WebGL2's pixel pack
 * buffer and fence), starts a read and resolves once `buf` holds it. Call `resize(bytes)` whenever
 * the target changes size, then `next()` once a frame: it returns the newest complete pixels.
 * The first frame after a resize reads at once, so the mark never shows an empty field. After it,
 * one read is in flight at a time and the scene renders only when a read starts, so the grid lags
 * the model by a frame or so. A failed async read falls back to reading at once.
 */
export function frameReader({ render, readSync, readAsync = null }) {
  let front = new Uint8Array(0);
  let back = front;
  let gen = 0;
  let busy = false;
  let stale = true;
  let read = readAsync;
  return {
    resize(bytes) {
      front = new Uint8Array(bytes);
      back = new Uint8Array(bytes);
      gen++;
      busy = false;
      stale = true;
    },
    next() {
      if (stale || !read) {
        render();
        readSync(front);
        stale = false;
        return front;
      }
      if (!busy) {
        const g = gen;
        const buf = back;
        busy = true;
        render();
        // The read starts now, right after the render, and lands on a later frame.
        let landed;
        try {
          landed = Promise.resolve(read(buf));
        } catch (err) {
          landed = Promise.reject(err);
        }
        landed.then(
            () => {
              if (g !== gen) return;
              back = front;
              front = buf;
              busy = false;
            },
            () => {
              if (g !== gen) return;
              read = null;
              busy = false;
            },
          );
      }
      return front;
    },
  };
}
