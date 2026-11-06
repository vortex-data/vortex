// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// The 404's mark, as vortex.dev draws it (the design system's AsciiMark): the Vortex mark turning
// in 3D, drawn as Geist Mono characters. The cell stays empty while the page loads. Once the page
// is idle, the renderer and three.js load and the live or reduced-motion mark mounts. Without
// WebGL, when three.js cannot load, or when mounting throws, the static symbol shows, and the
// host's data-mark-mode says why. three.js is a copy the docs serve, because their CSP allows
// scripts from this origin only.
const host = document.querySelector("[data-ascii-mark]");

async function mount() {
  // A three.js module without WebGLRenderer makes mountAsciiMark return a static mark, so a file
  // that fails to load leaves the symbol.
  const [{ mountAsciiMark }, three] = await Promise.all([
    import("./vortex-mark/ascii-mark.js"),
    import("./vortex-mark/three.module.min.js").catch(() => ({})),
  ]);
  // The DotField under the cell draws the dots, so the mark leaves its own.
  const mark = await mountAsciiMark(host, { three, field: false });
  host.dataset.markMode = mark.mode;
}

function run() {
  mount().catch(() => {
    host.dataset.markMode = "error";
  });
}

if (host) {
  const idle = () =>
    typeof requestIdleCallback === "function" ? requestIdleCallback(run, { timeout: 2000 }) : setTimeout(run, 200);
  if (document.readyState === "complete") idle();
  else addEventListener("load", idle, { once: true });
}
