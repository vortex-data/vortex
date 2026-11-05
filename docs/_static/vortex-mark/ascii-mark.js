// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// A copy of the Vortex design system's components/lib/ascii-mark.js. Edit the system, then copy it again.

// The Vortex mark, turning in 3D and redrawn every frame as a grid of Geist Mono characters.
// The pointer leaves a short Blue trail in which the characters become flickering hex bytes.
// Under reduced motion it draws one still frame and ignores the pointer. Without WebGL, or when
// three.js cannot load, it shows the static symbol. It draws in the host's colour scheme and
// follows a change of data-theme on the page.
import { brushStrength, fitFov, frameReader, hexDigit, markPalette, rampChar } from "./ascii-core.js";
import { decode } from "./vortex-mark-geometry.js";

const SCALE = 3.0;
const TILT = -0.3;
const YAW = -0.55;
const THICK = 1.6;
const OFFSET = [-0.13, 0.04, 0];
const SPIN = 14; // degrees per second
const LOOK = 0.22;
const LOOK_EASE = 0.06;
const TRAIL_MS = 450;
const TRAIL_STEP = 12;
const TRAIL = 8;
const SUB = 2; // render-target pixels per cell side
const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

/**
 * Mounts the mark in `host` (positioned, sized by the caller).
 * Options: patch (brush reach in px, 31), lean (follow the pointer, false), size and lineHeight
 * of the character grid (12, 14), font (the grid's family; the host's --font-mono token by
 * default), fallbackSrc (the symbol image for the static mode), mode ("auto", or "static" for the
 * still symbol whatever the browser can do), three (a three.js module the site bundles, for pages
 * whose CSP blocks the CDN), threeUrl (where to load three.js otherwise), exposure (how bright the
 * lit faces read on the ramp, 2.5), field (draw the dot field around the mark, true; false when
 * a DotField already lies under the host, so the page shows one grid of dots, not two). The mark
 * is decorative: keep the host aria-hidden.
 * Colours follow the host's colour-scheme, which the tokens set per theme: Dark greys on a Dark
 * ground, mirrored greys in Light (markPalette in ascii-core.js), and a two-tone Blue brush. In
 * Light the symbol image is inverted, so pass the white one in both themes.
 * Resolves to { mode: "live" | "reduced" | "static", destroy() }. Call destroy() when the host
 * leaves the page: it stops the loop, removes the listeners and frees the WebGL context.
 */
export async function mountAsciiMark(host, opts = {}) {
  const cleanups = [];
  const destroy = () => {
    while (cleanups.length) cleanups.pop()();
  };
  // A step that throws part-way (a lost WebGL context, a bad three.js) must not leave the observer,
  // the canvas, or the context alive: the caller never gets destroy() from a rejected promise.
  try {
    return await mount(host, opts, cleanups, destroy);
  } catch (err) {
    destroy();
    throw err;
  }
}

async function mount(host, opts, cleanups, destroy) {
  const hostCss = getComputedStyle(host);
  const o = {
    font: hostCss.getPropertyValue("--font-mono").trim() || '"Geist Mono", ui-monospace, monospace',
    mode: "auto",
    size: 12,
    lineHeight: 14,
    patch: 31,
    lean: false,
    fallbackSrc: null,
    three: null,
    threeUrl: THREE_URL,
    exposure: 2.5,
    field: true,
    ...opts,
  };
  // The palette for the host's scheme, read again whenever a data-theme changes on the page.
  let inks, field, brush, light;
  const paint = () => {
    const css = getComputedStyle(host);
    light = css.colorScheme === "light";
    const p = markPalette(light ? "light" : "dark");
    const read = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    inks = p.inks.map((n) => read(n, "#888"));
    field = read(p.field, light ? "#d4d4d2" : "#2b2b29");
    brush = light ? [read(p.brush[0], "#1d0fed"), read(p.brush[1], "#2f63ff")] : [read(p.brush[0], "#2f63ff"), read(p.brush[1], "#8fa9ff")];
  };
  paint();
  const repaints = [];
  const themes = new MutationObserver(() => {
    paint();
    for (const f of repaints) f();
  });
  themes.observe(document.documentElement, { attributes: true, subtree: true, attributeFilter: ["data-theme"] });
  cleanups.push(() => themes.disconnect());
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const still = () => {
    if (o.fallbackSrc) {
      const img = document.createElement("img");
      img.src = o.fallbackSrc;
      img.alt = "";
      img.style.cssText = "position:absolute;inset:0;margin:auto;width:40%;height:auto;opacity:0.85";
      const tone = () => (img.style.filter = light ? "invert(1)" : "");
      tone();
      repaints.push(tone);
      host.appendChild(img);
      cleanups.push(() => img.remove());
    }
    return { mode: "static", destroy };
  };

  if (o.mode === "static") return still();
  let THREE;
  let renderer;
  try {
    // A bundler must not try to resolve the CDN URL: sites under a CSP pass `three` instead.
    THREE = o.three || (await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ /* @vite-ignore */ o.threeUrl));
    renderer = new THREE.WebGLRenderer({ antialias: false });
  } catch {
    renderer = null;
  }
  if (!renderer) return still();
  cleanups.push(() => {
    renderer.dispose();
    renderer.forceContextLoss();
  });

  // The grid's font is a nicety: if it fails to download, draw with the fallback rather than not at all.
  await document.fonts.load(`${o.size}px ${o.font}`).catch(() => {});
  const cv = document.createElement("canvas");
  cv.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  // The dot field fades out toward the host's edges instead of ending in a hard rectangle.
  cv.style.maskImage = cv.style.webkitMaskImage = "radial-gradient(closest-side, #000 78%, transparent 100%)";
  cv.setAttribute("aria-hidden", "true");
  host.appendChild(cv);
  cleanups.push(() => cv.remove());
  const ctx = cv.getContext("2d");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#000000");
  scene.add(new THREE.AmbientLight(0xffffff, 0.08));
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(1, 2, 6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.4);
  fill.position.set(-3, -1, 1);
  scene.add(fill);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 0, 6);

  const g = decode();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(g.positions, 3));
  if (g.normals) geometry.setAttribute("normal", new THREE.BufferAttribute(g.normals, 3));
  else geometry.computeVertexNormals();
  geometry.setIndex(new THREE.BufferAttribute(g.indices, 1));
  const material = new THREE.MeshStandardMaterial({ color: "#ffffff", metalness: 0, roughness: 0.6, flatShading: true });
  const mesh = new THREE.Mesh(geometry, material);
  // The render target, made at the first resize. Declared before the cleanup that frees it.
  let target = null;
  cleanups.push(() => {
    geometry.dispose();
    material.dispose();
    target?.dispose();
  });
  mesh.scale.set(1, THICK, 1);
  const lean = new THREE.Group();
  lean.position.set(...OFFSET);
  lean.rotation.y = YAW;
  lean.scale.setScalar(SCALE);
  const stand = new THREE.Group();
  stand.rotation.x = -Math.PI / 2 + TILT;
  const spin = new THREE.Group();
  spin.add(mesh);
  stand.add(spin);
  lean.add(stand);
  scene.add(lean);

  let cols = 0;
  let rows = 0;
  let cw = 0;
  let W = 0;
  let H = 0;
  // The grid reads the render back from the GPU. Where three.js has an async read (WebGL2's pixel
  // pack buffer and a fence), the page never waits for the GPU after the first frame: the grid
  // draws from the newest read that has landed. Elsewhere, and for the still frame under reduced
  // motion, it reads at once. three.js leaves its pack buffer bound while a read is in flight,
  // which would make a later synchronous read fail, so it is unbound at once.
  const gl = renderer.getContext();
  const canReadAsync =
    !reduced && typeof renderer.readRenderTargetPixelsAsync === "function" && typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
  const reader = frameReader({
    render: () => {
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
    },
    readSync: (buf) => {
      renderer.readRenderTargetPixels(target, 0, 0, cols * SUB, rows * SUB, buf);
      renderer.setRenderTarget(null);
    },
    readAsync: canReadAsync
      ? (buf) => {
          const landed = renderer.readRenderTargetPixelsAsync(target, 0, 0, cols * SUB, rows * SUB, buf);
          gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
          renderer.setRenderTarget(null);
          return landed;
        }
      : null,
  });
  // Lays the grid out for the host's size. A call at the same size changes nothing, so the
  // ResizeObserver's first callback costs no second synchronous read. Returns whether it changed.
  const layout = () => {
    if (target && host.clientWidth === W && host.clientHeight === H) return false;
    W = host.clientWidth;
    H = host.clientHeight;
    const dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = `${o.size}px ${o.font}`;
    ctx.textBaseline = "top";
    cw = ctx.measureText("M").width;
    cols = Math.max(1, Math.floor(W / cw));
    rows = Math.max(1, Math.floor(H / o.lineHeight));
    target?.dispose();
    target = new THREE.WebGLRenderTarget(cols * SUB, rows * SUB);
    reader.resize(cols * SUB * rows * SUB * 4);
    camera.aspect = (cols * cw) / (rows * o.lineHeight);
    camera.fov = fitFov(30, camera.aspect);
    camera.updateProjectionMatrix();
    return true;
  };
  layout();

  let look = null;
  let head = null;
  let trail = [];
  const toLocal = (e) => {
    const r = host.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) };
  };
  const onMove = (e) => {
    const p = toLocal(e);
    look = { x: (p.x / W) * 2 - 1, y: (p.y / H) * 2 - 1 };
    const last = trail.at(-1);
    if (!last || Math.hypot(p.x - last.x, p.y - last.y) >= TRAIL_STEP) trail = [...trail, { ...p, born: performance.now() }].slice(-TRAIL);
    head = { ...p, born: performance.now() };
  };
  const onLeave = () => {
    look = null;
    if (head) trail = [...trail, head].slice(-TRAIL);
    head = null;
  };

  const draw = (now) => {
    const pixels = reader.next();
    trail = trail.filter((s) => now - s.born < TRAIL_MS);
    const marks = [
      ...trail.map((s) => {
        const life = Math.max(0, 1 - (now - s.born) / TRAIL_MS);
        return { x: s.x, y: s.y, reach: o.patch * (0.6 + life * 0.4), k: life * 0.85 };
      }),
      ...(head ? [{ x: head.x, y: head.y, reach: o.patch, k: 1 }] : []),
    ];
    ctx.clearRect(0, 0, W, H);
    const ox = (W - cols * cw) / 2;
    const oy = (H - rows * o.lineHeight) / 2;
    const tick = Math.floor(now / 90);
    const tw = cols * SUB;
    const buckets = new Map();
    const put = (color, ch, x, y) => {
      let b = buckets.get(color);
      if (!b) buckets.set(color, (b = []));
      b.push(ch, x, y);
    };
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        let s = 0;
        for (let y = 0; y < SUB; y++)
          for (let x = 0; x < SUB; x++) {
            const i = (((rows - 1 - r) * SUB + y) * tw + c * SUB + x) * 4;
            s += pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114;
          }
        const v = Math.min(1, ((s / (SUB * SUB * 255)) * o.exposure) ** 0.8);
        const x = ox + c * cw;
        const y = oy + r * o.lineHeight;
        const b = marks.length && v > 0.03 ? brushStrength(x + cw / 2, y + o.lineHeight / 2, marks) : 0;
        if (b > 0.12) put(b > 0.55 ? brush[1] : brush[0], hexDigit(r, c, tick), x, y);
        else if (v > 0.03) put(inks[Math.min(inks.length - 1, Math.floor(v * inks.length))], rampChar(Math.max(v, 0.12)), x, y);
        else if (o.field && r % 2 === 0 && c % 2 === 0) put(field, ".", x, y);
      }
    for (const [color, list] of buckets) {
      ctx.fillStyle = color;
      for (let i = 0; i < list.length; i += 3) ctx.fillText(list[i], list[i + 1], list[i + 2]);
    }
  };

  // Reduced motion: one still frame, redrawn only when the host is resized. No brush.
  if (reduced) {
    draw(performance.now());
    repaints.push(() => draw(performance.now()));
    const ro = new ResizeObserver(() => {
      if (layout()) draw(performance.now());
    });
    ro.observe(host);
    cleanups.push(() => ro.disconnect());
    return { mode: "reduced", destroy };
  }

  host.addEventListener("pointermove", onMove);
  host.addEventListener("pointerleave", onLeave);
  cleanups.push(() => {
    host.removeEventListener("pointermove", onMove);
    host.removeEventListener("pointerleave", onLeave);
  });
  const ro = new ResizeObserver(layout);
  ro.observe(host);
  cleanups.push(() => ro.disconnect());

  // The loop runs only while the host is on screen.
  let raf = 0;
  let last = performance.now();
  const frame = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    spin.rotation.y += (SPIN * Math.PI * dt) / 180;
    const at = o.lean ? look : null;
    lean.rotation.y += (YAW + (at ? at.x * LOOK : 0) - lean.rotation.y) * LOOK_EASE;
    lean.rotation.x += ((at ? -at.y * LOOK : 0) - lean.rotation.x) * LOOK_EASE;
    draw(now);
    raf = requestAnimationFrame(frame);
  };
  const start = () => {
    if (raf) return;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  };
  const stop = () => {
    cancelAnimationFrame(raf);
    raf = 0;
  };
  const io = new IntersectionObserver(([entry]) => (entry.isIntersecting ? start() : stop()));
  io.observe(host);
  cleanups.push(() => {
    io.disconnect();
    stop();
  });
  start();
  return { mode: "live", destroy };
}
