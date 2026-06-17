// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Workload } from '@/lib/synthesis';

/**
 * The small monochrome "blueprint" SVG of a claim's access pattern (not its
 * result), ported verbatim from `server/src/html/showcase.rs`. Stroke/fill read
 * the CSS theme vars (`--line-strong`, `--bar`, `--muted`) so the schematics
 * recolour with the page; a shared viewBox keeps the four the same size (small
 * multiples).
 */

const RANDOM_ACCESS_SVG = `<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Reading scattered individual rows by position">
<g stroke="var(--line-strong)" stroke-width="1">
<rect x="12" y="10" width="96" height="64"/>
<path d="M28 10V74M44 10V74M60 10V74M76 10V74M92 10V74"/>
<path d="M12 26H108M12 42H108M12 58H108"/>
</g>
<g fill="var(--bar)">
<rect x="29" y="11" width="14" height="14"/>
<rect x="77" y="27" width="14" height="14"/>
<rect x="13" y="43" width="14" height="14"/>
<rect x="61" y="59" width="14" height="14"/>
</g>
</svg>`;

const ANALYTICS_SVG = `<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Scanning whole columns into an aggregate">
<g stroke="var(--line-strong)" stroke-width="1">
<rect x="12" y="10" width="96" height="64"/>
<path d="M28 10V74M44 10V74M60 10V74M76 10V74M92 10V74"/>
<path d="M12 26H108M12 42H108M12 58H108"/>
</g>
<g fill="var(--bar)">
<rect x="29" y="11" width="14" height="62"/>
<rect x="77" y="11" width="14" height="62"/>
</g>
<g stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M108 42H117"/>
<path d="M113 38l4 4l-4 4"/>
</g>
<text x="128" y="47" fill="var(--muted)" font-family="monospace" font-size="13" text-anchor="middle">&#931;</text>
</svg>`;

const WRITES_SVG = `<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Encoding loose rows into a packed file">
<g stroke="var(--bar)" stroke-width="2.5" stroke-linecap="round">
<path d="M8 24H44"/>
<path d="M8 36H38"/>
<path d="M8 48H44"/>
<path d="M8 60H34"/>
</g>
<g stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M52 42H72"/>
<path d="M66 36l6 6l-6 6"/>
</g>
<rect x="84" y="12" width="48" height="60" stroke="var(--line-strong)" stroke-width="1"/>
<g fill="var(--bar)">
<rect x="86" y="14" width="44" height="12"/>
<rect x="86" y="29" width="44" height="12"/>
<rect x="86" y="44" width="44" height="12"/>
<rect x="86" y="59" width="44" height="11"/>
</g>
</svg>`;

const SIZE_SVG = `<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Compressing data onto disk">
<rect x="14" y="22" width="112" height="40" stroke="var(--line-strong)" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="44" y="22" width="52" height="40" fill="var(--bar)"/>
<g stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M24 34l10 8l-10 8"/>
<path d="M116 34l-10 8l10 8"/>
</g>
</svg>`;

const SVGS: Record<Workload, string> = {
  randomAccess: RANDOM_ACCESS_SVG,
  analytics: ANALYTICS_SVG,
  writes: WRITES_SVG,
  size: SIZE_SVG,
};

/** Render the blueprint schematic for a workload as the claim's figure. */
export function WorkloadFigure({ workload }: { workload: Workload }) {
  return <div className="claim-figure" dangerouslySetInnerHTML={{ __html: SVGS[workload] }} />;
}
