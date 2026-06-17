// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import Chart, { type Plugin } from 'chart.js/auto';
import { useEffect, useRef, useState } from 'react';

import type { EngineData, Metric } from '@/lib/synthesis';

/**
 * One facet's speedup-distribution figure — the React/Chart.js port of
 * `chart-init.js::buildSpeedupChart`. A horizontal diverging-bar chart of the
 * per-item `B / A` ratio: each bar is a floating `[0, log2(ratio)]` range on a
 * log2 x-axis, so every bar anchors at the 1× baseline; wins (ratio ≥ 1) fill
 * `--bar`, losses fill `--bad`. The A/B `<select>`s pick any pair of formats and
 * recompute the distribution, the geomean stat, and the win count.
 *
 * `sortMode` is owned by the enclosing section (the Speedup/Query toggle), so a
 * sort re-orders every chart in the group at once.
 */

const MONO_FONT = '"Geist Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace';
const ANTONYM: Record<Metric, string> = { faster: 'slower', smaller: 'larger' };
const HOVER_GOLD = '#f0b429';
// The one animation we keep: the sort slide. Only the bar rows (y) change, so an
// animated update glides each bar to its new row (matching `chart-init.js`).
const SORT_ANIM = { duration: 600, easing: 'easeInOutQuart' } as const;

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** A loose view over a Chart.js cartesian scale for the imperative post-build
 * mutations (Chart's option types are a deep-partial union that doesn't expose
 * `title`/`ticks` uniformly). The fields exist at runtime because the chart is
 * configured with them. */
interface MutableScale {
  min?: number;
  max?: number;
  title?: { text?: string; color?: string };
  ticks?: { color?: string };
}

interface Point {
  query: string;
  speedup: number;
  aDisp: string;
  bDisp: string;
}

function speedupPoints(ed: EngineData, a: string, b: string): Point[] {
  const pts: Point[] = [];
  for (const q of ed.queries) {
    const va = q.v[a];
    const vb = q.v[b];
    if (va !== undefined && vb !== undefined && va > 0 && vb > 0) {
      pts.push({ query: q.query, speedup: vb / va, aDisp: q.d[a], bDisp: q.d[b] });
    }
  }
  return pts;
}

/** Display row (0 = top) of each point under a sort mode. */
function speedupRanks(points: Point[], mode: 'speedup' | 'query'): number[] {
  const n = points.length;
  const rank = new Array<number>(n);
  if (mode === 'query') {
    for (let i = 0; i < n; i++) {
      rank[i] = i;
    }
    return rank;
  }
  const idx = points.map((_, i) => i);
  idx.sort((x, y) => points[y].speedup - points[x].speedup);
  idx.forEach((origIdx, pos) => {
    rank[origIdx] = pos;
  });
  return rank;
}

/** Geometric mean of positive, finite values (1 if none qualify). */
function geomeanArr(xs: number[]): number {
  const v = xs.filter((x) => x > 0 && Number.isFinite(x));
  if (v.length === 0) {
    return 1;
  }
  let s = 0;
  for (const x of v) {
    s += Math.log(x);
  }
  return Math.exp(s / v.length);
}

function labelOf(ed: EngineData, id: string): string {
  return ed.formats.find((f) => f.id === id)?.label ?? id;
}

function fmtSpeedupTick(logVal: number): string {
  const r = Math.pow(2, logVal);
  const s = r >= 100 ? String(Math.round(r)) : String(Math.round(r * 100) / 100);
  return `${s}×`;
}

function axisTitle(ed: EngineData, aL: string, bL: string, hasLoss: boolean): string {
  const metric = ed.metric;
  return hasLoss ? `← ${bL} ${metric}      ${aL} ${metric} →` : `${aL} ${metric} →`;
}

/** Win/loss/hover fill for each point, shared by the render and hover passes. */
function barColors(points: Point[], hovered: string | null, bar: string, bad: string): string[] {
  return points.map((p) =>
    hovered !== null && p.query === hovered ? HOVER_GOLD : p.speedup >= 1 ? bar : bad,
  );
}

/**
 * The pre-layout `barThickness` estimate can leave a sub-pixel gap between dense
 * rows, which reads as faint stripes of background between bars. After each
 * layout, size the bars to the measured row spacing (+1px so adjacent bars
 * overlap slightly and the distribution reads as one continuous mass). Ported
 * from `chart-init.js::fitSpeedupBarsPlugin`; the corrective update is deferred a
 * frame so it never recurses inside the current update.
 */
const fitSpeedupBarsPlugin: Plugin<'bar'> = {
  id: 'fitSpeedupBars',
  afterUpdate(chart) {
    const y = chart.scales.y;
    const ds = chart.data.datasets[0];
    const n = ds.data.length;
    if (!y || n < 2) {
      return;
    }
    const perRow = Math.abs(y.getPixelForValue(1) - y.getPixelForValue(0));
    if (!Number.isFinite(perRow) || perRow <= 0) {
      return;
    }
    const t = Math.max(3, Math.round(perRow) + 1);
    if (ds.barThickness !== t) {
      ds.barThickness = t;
      requestAnimationFrame(() => {
        if (chart.canvas) {
          chart.update('none');
        }
      });
    }
  },
};

export function SpeedupChart({
  data,
  sortMode,
  hoverQuery,
  onHover,
}: {
  data: EngineData;
  sortMode: 'speedup' | 'query';
  /** The item hovered anywhere in this section's grid, painted gold here too. */
  hoverQuery: string | null;
  /** Report the item hovered in this chart up to the section (null on leave). */
  onHover: (query: string | null) => void;
}) {
  const [a, setA] = useState(data.defaultA);
  const [b, setB] = useState(data.defaultB);
  const [stat, setStat] = useState('');
  const [wins, setWins] = useState('');
  const [themeKey, setThemeKey] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const pointsRef = useRef<Point[]>([]);
  const labelsRef = useRef<{ aL: string; bL: string }>({ aL: '', bL: '' });
  // Latest-callback / latest-value refs so the mount-only chart's onHover and
  // the render effect always see the current props.
  const onHoverRef = useRef(onHover);
  onHoverRef.current = onHover;
  const hoverQueryRef = useRef(hoverQuery);
  hoverQueryRef.current = hoverQuery;
  const prevSortRef = useRef(sortMode);
  const sortTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const height = 64 + data.queries.length * 13;

  // Build the chart once on mount; teardown on unmount.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const line = cssVar('--line');
    const fg = cssVar('--fg');
    const thickness = Math.max(
      4,
      Math.min(16, Math.floor((height - 44) / Math.max(1, data.queries.length)) - 2),
    );
    const chart = new Chart(canvas, {
      type: 'bar',
      plugins: [fitSpeedupBarsPlugin],
      data: {
        datasets: [
          {
            data: [],
            backgroundColor: [],
            borderWidth: 0,
            barThickness: thickness,
            hoverBackgroundColor: '#f0b429',
            hoverBorderColor: '#f0b429',
            hoverBorderWidth: 2,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { right: 10, left: 2 } },
        // Report the hovered item up to the section so every panel in the grid
        // can mirror the gold highlight (the synthesis `propagateHover`).
        onHover: (_e, els) => {
          const idx = els.length > 0 ? els[0].index : -1;
          onHoverRef.current(idx >= 0 ? (pointsRef.current[idx]?.query ?? null) : null);
        },
        scales: {
          x: {
            grid: {
              color: (ctx) => (ctx.tick && ctx.tick.value === 0 ? fg : line),
              lineWidth: (ctx) => (ctx.tick && ctx.tick.value === 0 ? 1.5 : 1),
              drawTicks: false,
            },
            border: { display: false },
            ticks: {
              stepSize: 1,
              color: cssVar('--muted'),
              font: { family: MONO_FONT, size: 12 },
              callback: (v) => fmtSpeedupTick(Number(v)),
            },
            title: {
              display: true,
              text: '',
              color: cssVar('--faint'),
              font: { family: MONO_FONT, size: 12 },
            },
          },
          y: {
            type: 'linear',
            reverse: true,
            min: -0.5,
            max: Math.max(0.5, data.queries.length - 0.5),
            grid: { display: false },
            border: { display: false },
            ticks: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            displayColors: false,
            bodyFont: { family: MONO_FONT, size: 12 },
            titleFont: { family: MONO_FONT, size: 12 },
            callbacks: {
              title: (items) =>
                items.length ? (pointsRef.current[items[0].dataIndex]?.query ?? '') : '',
              label: (ctx) => {
                const p = pointsRef.current[ctx.dataIndex];
                if (p === undefined) {
                  return '';
                }
                const { aL, bL } = labelsRef.current;
                const metric = data.metric;
                const anti = ANTONYM[metric];
                const rel =
                  p.speedup >= 1
                    ? `${p.speedup.toFixed(2)}× ${metric}`
                    : `${(1 / p.speedup).toFixed(2)}× ${anti}`;
                return [
                  `${aL} ${rel}`,
                  `${aL.toLowerCase()} ${p.aDisp}   ${bL.toLowerCase()} ${p.bDisp}`,
                ];
              },
            },
          },
        },
      },
    });
    chartRef.current = chart;
    // Chart.js's synthetic "no active element" on mouseout is unreliable on a
    // fast swipe, so clear the grid highlight explicitly on leave.
    const onLeave = () => onHoverRef.current(null);
    canvas.addEventListener('mouseleave', onLeave);
    return () => {
      canvas.removeEventListener('mouseleave', onLeave);
      chart.destroy();
      chartRef.current = null;
    };
    // Build once: subsequent visual changes go through the render effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recompute points / colours / scale / headline for the current selection,
  // sort, and theme (ported from `renderSpeedup` + per-chart `syncGridScale`).
  useEffect(() => {
    const chart = chartRef.current;
    if (chart === null) {
      return;
    }
    const points = speedupPoints(data, a, b);
    pointsRef.current = points;
    const ranks = speedupRanks(points, sortMode);
    const bar = cssVar('--bar');
    const bad = cssVar('--bad');
    const ds = chart.data.datasets[0];
    ds.data = points.map((p, i) => ({ x: [0, Math.log2(p.speedup)], y: ranks[i] }) as never);
    ds.backgroundColor = barColors(points, hoverQueryRef.current, bar, bad);

    let maxWin = 0;
    let maxLoss = 0;
    for (const p of points) {
      const v = Math.log2(p.speedup);
      if (v > maxWin) {
        maxWin = v;
      }
      if (-v > maxLoss) {
        maxLoss = -v;
      }
    }
    const pad = Math.max((maxWin + maxLoss) * 0.06, 0.15);
    const scales = chart.options.scales as unknown as { x: MutableScale; y: MutableScale };
    scales.x.min = maxLoss > 0 ? -(maxLoss + pad) : 0;
    scales.x.max = Math.max(maxWin, 0.5) + pad;
    scales.y.max = Math.max(0.5, points.length - 0.5);

    const aL = labelOf(data, a);
    const bL = labelOf(data, b);
    labelsRef.current = { aL, bL };
    if (scales.x.title !== undefined) {
      scales.x.title.text = axisTitle(data, aL, bL, maxLoss > 0);
      scales.x.title.color = cssVar('--faint');
    }
    if (scales.x.ticks !== undefined) {
      scales.x.ticks.color = cssVar('--muted');
    }
    // Only a sort change animates (the bars slide to their new rows); selection
    // and theme updates stay instant. Re-arm `animation: false` after the slide
    // (Chart.js reads the option per frame, so reset must wait for it to finish).
    const isSort = prevSortRef.current !== sortMode;
    prevSortRef.current = sortMode;
    chart.options.animation = isSort ? SORT_ANIM : false;
    chart.update();
    if (isSort) {
      clearTimeout(sortTimerRef.current);
      sortTimerRef.current = setTimeout(() => {
        if (chartRef.current !== null) {
          chartRef.current.options.animation = false;
        }
      }, SORT_ANIM.duration + 80);
    }

    const speeds = points.map((p) => p.speedup);
    const geo = geomeanArr(speeds);
    const metric = data.metric;
    const winCount = speeds.filter((s) => s >= 1).length;
    setStat(
      geo >= 1 ? `${geo.toFixed(2)}× ${metric}` : `${(1 / geo).toFixed(2)}× ${ANTONYM[metric]}`,
    );
    setWins(`${aL} wins ${winCount}/${speeds.length}`);
  }, [data, a, b, sortMode, themeKey]);

  // Re-bake the palette when the theme flips (canvas fills are drawn, not CSS).
  useEffect(() => {
    const onTheme = () => setThemeKey((k) => k + 1);
    window.addEventListener('bench:themechange', onTheme);
    return () => window.removeEventListener('bench:themechange', onTheme);
  }, []);

  // Mirror the gold highlight when any chart in the grid reports a hovered item
  // (the cross-panel `propagateHover`). Repaints colours only — no re-layout.
  useEffect(() => {
    const chart = chartRef.current;
    if (chart === null) {
      return;
    }
    chart.data.datasets[0].backgroundColor = barColors(
      pointsRef.current,
      hoverQuery,
      cssVar('--bar'),
      cssVar('--bad'),
    );
    chart.update('none');
  }, [hoverQuery]);

  // Keep A and B distinct: if a change collides, bump the other select.
  const onSelectA = (value: string) => {
    setA(value);
    if (value === b) {
      const other = data.formats.find((f) => f.id !== value);
      if (other !== undefined) {
        setB(other.id);
      }
    }
  };
  const onSelectB = (value: string) => {
    setB(value);
    if (value === a) {
      const other = data.formats.find((f) => f.id !== value);
      if (other !== undefined) {
        setA(other.id);
      }
    }
  };

  return (
    <figure className="speedup" data-role="speedup-chart">
      <figcaption className="speedup-head">
        {data.facet !== '' && <span className="speedup-engine">{data.facet}</span>}
        <div className="speedup-compare">
          <select
            className="speedup-select"
            data-role="speedup-a"
            aria-label="Comparison format"
            value={a}
            onChange={(e) => onSelectA(e.target.value)}
          >
            {data.formats.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
          <span className="speedup-vs">vs</span>
          <select
            className="speedup-select"
            data-role="speedup-b"
            aria-label="Comparison format"
            value={b}
            onChange={(e) => onSelectB(e.target.value)}
          >
            {data.formats.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </div>
        <span className="speedup-stat" data-role="speedup-stat">
          {stat} <span className="speedup-stat-note">(geo. mean)</span>
        </span>
        <span className="speedup-wins" data-role="speedup-wins">
          {wins}
        </span>
      </figcaption>
      <div className="speedup-chart-wrap" style={{ height: `${height}px` }}>
        <canvas ref={canvasRef} data-role="speedup-canvas" />
      </div>
    </figure>
  );
}
