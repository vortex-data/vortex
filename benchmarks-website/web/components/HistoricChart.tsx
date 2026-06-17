// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import Chart, { type ChartDataset } from 'chart.js/auto';
import { useCallback, useEffect, useRef, useState } from 'react';

import type { ChartResponse, UnitKind } from '@/lib/queries';

/**
 * One Historic-Data chart card's robust band chart — the minimal React/Chart.js
 * port of `chart-init.js::constructRobustChart`. Per series it draws:
 *   - the faint raw per-commit **outlier dots** (prominent for sparse series),
 *   - the p25–p75 **band** (a filled ribbon), and
 *   - the rolling inter-quartile-mean **median-tracking line**
 * coloured by format (Vortex is the hero), dashed by engine. The zoom / pan /
 * range-slider / y-toggle machinery from the synthesis explorer is intentionally
 * omitted here — the band + median is the unit.
 *
 * Load performance is preserved: the payload is fetched lazily via an
 * IntersectionObserver, so a chart only hits `/api/chart/[slug]` when its card
 * scrolls into view (and a collapsed group's grid is `display:none`, so nothing
 * fetches until the group is expanded — matching v4's lazy-on-expand model).
 */

const ROBUST_WINDOW = 9;
const ROBUST_WINDOW_MAX = 75;
const SPARSE_DOT_SAMPLES = 600;
const DOTTED_DASH = [3, 4];
const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Human label for a physical format id (the client-safe copy of
 * `lib/synthesis::formatLabel`; importing the value from `lib/synthesis` would
 * pull the server-only `pg` graph into this client bundle). */
function formatLabel(id: string): string {
  switch (id) {
    case 'vortex-file-compressed':
      return 'Vortex';
    case 'vortex-compact':
      return 'Vortex-compact';
    case 'parquet':
      return 'Parquet';
    case 'arrow':
      return 'Arrow';
    case 'duckdb':
      return 'DuckDB';
    case 'lance':
      return 'Lance';
    default:
      return id;
  }
}

function formatColorVar(format: string | undefined): string {
  switch (format) {
    case 'vortex-file-compressed':
      return '--fmt-vortex';
    case 'vortex-compact':
      return '--fmt-vortex-compact';
    case 'parquet':
      return '--fmt-parquet';
    case 'arrow':
      return '--fmt-arrow';
    case 'duckdb':
      return '--fmt-duckdb';
    case 'lance':
      return '--fmt-lance';
    default:
      return '--fmt-other';
  }
}

/** Dash pattern encoding the query engine (colour encodes the format). Both
 * engines get a distinct broken pattern so they read apart at a glance and the
 * line-style key reads as "dotted DuckDB · dashed DataFusion". */
function engineDash(engine: string | undefined): number[] {
  switch (engine) {
    case 'duckdb':
      return [2, 3]; // dotted
    case 'datafusion':
      return [6, 4]; // dashed
    case undefined:
      return []; // no engine dimension (compression / size / random access) — solid
    default:
      return [1, 4]; // fine dotted (future engines)
  }
}

function engineLabel(engine: string): string {
  return engine === 'datafusion' ? 'DataFusion' : engine === 'duckdb' ? 'DuckDB' : engine;
}

/** Legend ordering for a format id: Vortex variants first, then the baselines. */
function formatOrder(id: string): number {
  switch (id) {
    case 'vortex-file-compressed':
      return 0;
    case 'vortex-compact':
      return 1;
    case 'parquet':
      return 2;
    case 'arrow':
      return 3;
    case 'duckdb':
      return 4;
    case 'lance':
      return 5;
    default:
      return 6;
  }
}

/** Legend label for a series: format, distinguished by engine or op when the
 * card carries more than one series per format. */
function seriesLabel(format: string | undefined, engine: string | undefined, name: string): string {
  const base = formatLabel(format ?? name);
  if (engine !== undefined) {
    return `${base} · ${engineLabel(engine)}`;
  }
  const idx = name.lastIndexOf(':');
  if (idx !== -1) {
    return `${base} · ${name.slice(idx + 1)}`;
  }
  return base;
}

/** Apply `pick` to the window of `w` non-null samples centred on each sample,
 * emitting the result at that sample's original index. */
function rollingSampleStat(
  v: readonly (number | null)[],
  w: number,
  pick: (sorted: number[]) => number | null,
): (number | null)[] {
  const idx: number[] = [];
  const val: number[] = [];
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (x !== null) {
      idx.push(i);
      val.push(x);
    }
  }
  const out = new Array<number | null>(v.length).fill(null);
  const h = w >> 1;
  for (let j = 0; j < val.length; j++) {
    const s = val.slice(Math.max(0, j - h), Math.min(val.length, j + h + 1)).sort((a, b) => a - b);
    out[idx[j]] = pick(s);
  }
  return out;
}

function rollingIqMean(v: readonly (number | null)[], w: number): (number | null)[] {
  return rollingSampleStat(v, w, (s) => {
    const n = s.length;
    if (n === 0) {
      return null;
    }
    const a = Math.floor(n * 0.25);
    const b = Math.max(a + 1, Math.ceil(n * 0.75));
    let sum = 0;
    for (let m = a; m < b; m++) {
      sum += s[m];
    }
    return sum / (b - a);
  });
}

function rollingQuantile(v: readonly (number | null)[], w: number, q: number): (number | null)[] {
  return rollingSampleStat(v, w, (s) =>
    s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] : null,
  );
}

function adaptiveWindow(present: number): number {
  return Math.max(ROBUST_WINDOW, Math.min(ROBUST_WINDOW_MAX, Math.round(present / 50)));
}

/** Hold the median's first/last real value across leading/trailing gaps and
 * return a `segment.borderDash` fn that dashes only those held extensions. */
function extendMedianToEdges(
  med: (number | null)[],
): ((ctx: { p0DataIndex: number; p1DataIndex: number }) => number[] | undefined) | null {
  const n = med.length;
  let first = -1;
  let last = -1;
  for (let i = 0; i < n; i++) {
    if (med[i] !== null) {
      if (first < 0) {
        first = i;
      }
      last = i;
    }
  }
  if (first < 0) {
    return null;
  }
  const fv = med[first];
  const lv = med[last];
  for (let i = 0; i < first; i++) {
    med[i] = fv;
  }
  for (let i = last + 1; i < n; i++) {
    med[i] = lv;
  }
  return (ctx) => (ctx.p1DataIndex <= first || ctx.p0DataIndex >= last ? DOTTED_DASH : undefined);
}

function magnitudeReference(values: number[]): number | null {
  const sample: number[] = [];
  for (const v of values) {
    if (Number.isFinite(v)) {
      const a = Math.abs(v);
      if (a !== 0) {
        sample.push(a);
      }
    }
  }
  if (sample.length === 0) {
    return null;
  }
  sample.sort((a, b) => a - b);
  const mid = Math.floor(sample.length / 2);
  return sample.length % 2 ? sample[mid] : (sample[mid - 1] + sample[mid]) / 2;
}

interface DisplayUnit {
  multiplier: number;
  suffix: string;
  axisLabel: string;
  decimals: number;
}

function pickDisplayUnit(unitKind: UnitKind, values: number[]): DisplayUnit {
  const ref = magnitudeReference(values);
  if (unitKind === 'time_ns') {
    if (ref === null || ref < 1e3) {
      return { multiplier: 1, suffix: 'ns', axisLabel: 'Time (ns)', decimals: 0 };
    }
    if (ref < 1e6) {
      return { multiplier: 1e-3, suffix: 'µs', axisLabel: 'Time (µs)', decimals: 2 };
    }
    if (ref < 1e9) {
      return { multiplier: 1e-6, suffix: 'ms', axisLabel: 'Time (ms)', decimals: 2 };
    }
    return { multiplier: 1e-9, suffix: 's', axisLabel: 'Time (s)', decimals: 2 };
  }
  if (unitKind === 'bytes') {
    const k = 1024;
    if (ref === null || ref < k) {
      return { multiplier: 1, suffix: 'B', axisLabel: 'Size (B)', decimals: 0 };
    }
    if (ref < k * k) {
      return { multiplier: 1 / k, suffix: 'KiB', axisLabel: 'Size (KiB)', decimals: 2 };
    }
    if (ref < k * k * k) {
      return { multiplier: 1 / (k * k), suffix: 'MiB', axisLabel: 'Size (MiB)', decimals: 2 };
    }
    if (ref < k * k * k * k) {
      return { multiplier: 1 / (k * k * k), suffix: 'GiB', axisLabel: 'Size (GiB)', decimals: 2 };
    }
    return { multiplier: 1 / (k * k * k * k), suffix: 'TiB', axisLabel: 'Size (TiB)', decimals: 2 };
  }
  if (unitKind === 'throughput_mb_s') {
    return { multiplier: 1, suffix: 'MB/s', axisLabel: 'Throughput (MB/s)', decimals: 2 };
  }
  return { multiplier: 1, suffix: '', axisLabel: '', decimals: unitKind === 'count' ? 0 : 2 };
}

function collectAllValues(payload: ChartResponse): number[] {
  const out: number[] = [];
  for (const arr of Object.values(payload.series)) {
    for (const v of arr) {
      if (v !== null && Number.isFinite(v)) {
        out.push(v);
      }
    }
  }
  return out;
}

function formatAxisDate(ts: string): string {
  if (ts.length < 10) {
    return '';
  }
  const parts = ts.slice(0, 10).split('-');
  if (parts.length !== 3) {
    return '';
  }
  const month = MONTH_NAMES[Number.parseInt(parts[1], 10) - 1];
  const day = Number.parseInt(parts[2], 10);
  if (month === undefined || !Number.isFinite(day)) {
    return '';
  }
  return `${month} ${day}`;
}

/** Format colour swatches + the engine line-style key shown beneath a chart. */
interface LegendModel {
  formats: { id: string; label: string; color: string }[];
  engines: { id: string; label: string; dash: number[] }[];
}

export function HistoricChart({ slug }: { slug: string }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const payloadRef = useRef<ChartResponse | null>(null);
  // format id -> the dataset indices (dots/lo/hi/median) it owns, for toggling.
  const fmtIdxRef = useRef<Map<string, number[]>>(new Map());
  const hiddenRef = useRef<Set<string>>(new Set());
  const [legend, setLegend] = useState<LegendModel | null>(null);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());

  const build = useCallback(() => {
    const payload = payloadRef.current;
    const canvas = canvasRef.current;
    if (payload === null || canvas === null) {
      return;
    }
    chartRef.current?.destroy();

    const meta = payload.series_meta ?? {};
    const raw = payload.series;
    const names = Object.keys(raw).sort();
    if (names.length === 0) {
      return;
    }
    const unit = pickDisplayUnit(payload.unit_kind, collectAllValues(payload));
    const mul = unit.multiplier;
    const labels = payload.commits.map((c) => (c.sha ? c.sha.slice(0, 7) : ''));
    const datasets: ChartDataset<'line', (number | null)[]>[] = [];
    const loAll: number[] = [];
    const hiAll: number[] = [];

    // Legend bookkeeping: colour encodes the format and dash encodes the engine,
    // so the legend is one colour swatch per FORMAT plus a separate line-style
    // key per engine — instead of a hard-to-read `Format · Engine` label per
    // series. Each format owns the dataset indices of all its series.
    const fmtIdx = new Map<string, number[]>();
    const fmtLabel = new Map<string, string>();
    const fmtColor = new Map<string, string>();
    const engineDashes = new Map<string, number[]>();

    for (const name of names) {
      const tag = meta[name] ?? {};
      const color = cssVar(formatColorVar(tag.format));
      const arr = (raw[name] ?? []).map((x) => (x === null ? null : x * mul));
      let present = 0;
      for (const x of arr) {
        if (x !== null) {
          present += 1;
        }
      }
      const win = adaptiveWindow(present);
      const med = rollingIqMean(arr, win);
      const lo = rollingQuantile(arr, win, 0.25);
      const hi = rollingQuantile(arr, win, 0.75);
      for (const x of lo) {
        if (x !== null) {
          loAll.push(x);
        }
      }
      for (const x of hi) {
        if (x !== null) {
          hiAll.push(x);
        }
      }
      const base = datasets.length;
      const dotsSparse = present <= SPARSE_DOT_SAMPLES;
      // Raw per-commit dots behind the band.
      datasets.push({
        data: arr,
        borderColor: 'transparent',
        backgroundColor: color + (dotsSparse ? 'aa' : '44'),
        pointRadius: dotsSparse ? 2.2 : 1.5,
        pointHoverRadius: dotsSparse ? 2.8 : 1.5,
        pointBorderWidth: 0,
        showLine: false,
        spanGaps: false,
      });
      // p25 (lower band edge, transparent) then p75 filling down to it.
      datasets.push({
        data: lo,
        borderColor: 'transparent',
        pointRadius: 0,
        fill: false,
        tension: 0.3,
        spanGaps: true,
      });
      datasets.push({
        data: hi,
        borderColor: 'transparent',
        pointRadius: 0,
        fill: '-1',
        backgroundColor: color + '26',
        tension: 0.3,
        spanGaps: true,
      });
      // The rolling IQ-mean median line: colour = format, dash = engine.
      const medDash = extendMedianToEdges(med);
      datasets.push({
        label: seriesLabel(tag.format, tag.engine, name),
        data: med,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.8,
        pointRadius: 0,
        pointHoverRadius: 3,
        fill: false,
        tension: 0.3,
        spanGaps: true,
        borderDash: engineDash(tag.engine),
        ...(medDash !== null ? { segment: { borderDash: medDash } } : {}),
      });

      const fid = tag.format;
      if (fid !== undefined) {
        let idxs = fmtIdx.get(fid);
        if (idxs === undefined) {
          idxs = [];
          fmtIdx.set(fid, idxs);
          fmtLabel.set(fid, formatLabel(fid));
          fmtColor.set(fid, color);
        }
        idxs.push(base, base + 1, base + 2, base + 3);
      }
      if (tag.engine !== undefined) {
        engineDashes.set(tag.engine, engineDash(tag.engine));
      }
    }

    loAll.sort((a, b) => a - b);
    hiAll.sort((a, b) => a - b);
    const ymin = loAll.length ? loAll[Math.floor(loAll.length * 0.02)] * 0.9 : undefined;
    const ymax = hiAll.length ? hiAll[Math.floor(hiAll.length * 0.98)] * 1.12 : undefined;

    const chart = new Chart(canvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            grid: { color: cssVar('--line'), drawTicks: false },
            ticks: {
              maxTicksLimit: 7,
              autoSkip: true,
              maxRotation: 0,
              color: cssVar('--muted'),
              callback: (val, idx, ticks) => {
                const t = ticks[idx];
                const ci = t && t.value != null ? t.value : Number(val);
                const c = payload.commits[ci];
                return c ? formatAxisDate(c.timestamp) : '';
              },
            },
          },
          y: {
            type: 'linear',
            min: ymin,
            max: ymax,
            grid: { color: cssVar('--line'), drawTicks: false },
            ticks: { color: cssVar('--muted') },
            title: { display: true, text: unit.axisLabel, color: cssVar('--faint') },
          },
        },
        plugins: {
          // Replaced by the custom format/engine legend rendered below the canvas.
          legend: { display: false },
          tooltip: {
            displayColors: true,
            filter: (it) => Boolean(it.dataset.label),
            callbacks: {
              label: (ctx) => {
                const y = ctx.parsed.y;
                const shown = y === null ? '—' : `${y.toFixed(unit.decimals)} ${unit.suffix}`;
                return `${ctx.dataset.label}: ${shown}`;
              },
            },
          },
        },
      },
    });
    chartRef.current = chart;
    fmtIdxRef.current = fmtIdx;

    // Re-apply any format the user toggled off (survives the theme rebuild).
    for (const fid of hiddenRef.current) {
      for (const i of fmtIdx.get(fid) ?? []) {
        chart.setDatasetVisibility(i, false);
      }
    }
    chart.update();

    setLegend({
      formats: [...fmtIdx.keys()]
        .sort((a, b) => formatOrder(a) - formatOrder(b) || (a < b ? -1 : a > b ? 1 : 0))
        .map((id) => ({ id, label: fmtLabel.get(id) ?? id, color: fmtColor.get(id) ?? '' })),
      engines: [...engineDashes.keys()]
        .sort()
        .map((id) => ({ id, label: engineLabel(id), dash: engineDashes.get(id) ?? [] })),
    });
  }, []);

  const toggleFormat = useCallback((fid: string) => {
    const chart = chartRef.current;
    if (chart === null) {
      return;
    }
    const next = new Set(hiddenRef.current);
    const nowHidden = !next.has(fid);
    if (nowHidden) {
      next.add(fid);
    } else {
      next.delete(fid);
    }
    hiddenRef.current = next;
    for (const i of fmtIdxRef.current.get(fid) ?? []) {
      chart.setDatasetVisibility(i, !nowHidden);
    }
    chart.update();
    setHidden(next);
  }, []);

  // Lazy-fetch the payload the first time the card scrolls into view.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (wrap === null) {
      return;
    }
    let fetched = false;
    const io = new IntersectionObserver(
      (entries) => {
        if (fetched || !entries.some((e) => e.isIntersecting)) {
          return;
        }
        fetched = true;
        io.disconnect();
        fetch(`/api/chart/${encodeURIComponent(slug)}?n=100`)
          .then((r) => (r.ok ? r.json() : null))
          .then((p: ChartResponse | null) => {
            if (p !== null) {
              payloadRef.current = p;
              build();
            }
          })
          .catch(() => {
            /* a transient fetch failure leaves the empty shell; reload retries */
          });
      },
      { rootMargin: '200px' },
    );
    io.observe(wrap);
    return () => {
      io.disconnect();
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [slug, build]);

  // Re-bake the palette when the theme flips (canvas colours are drawn, not CSS).
  useEffect(() => {
    const onTheme = () => {
      if (payloadRef.current !== null) {
        build();
      }
    };
    window.addEventListener('bench:themechange', onTheme);
    return () => window.removeEventListener('bench:themechange', onTheme);
  }, [build]);

  return (
    <>
      <div className="chart-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} />
      </div>
      {legend !== null && (legend.formats.length > 0 || legend.engines.length > 0) && (
        <div className="chart-legend">
          {legend.formats.length > 0 && (
            <div className="chart-legend-formats">
              {legend.formats.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  className={`chart-legend-fmt${hidden.has(f.id) ? ' chart-legend-fmt--off' : ''}`}
                  onClick={() => toggleFormat(f.id)}
                  aria-pressed={!hidden.has(f.id)}
                >
                  <span className="chart-legend-swatch" style={{ background: f.color }} />
                  {f.label}
                </button>
              ))}
            </div>
          )}
          {legend.engines.length > 0 && (
            <div className="chart-legend-engines">
              {legend.engines.map((e) => (
                <span className="chart-legend-engine" key={e.id}>
                  <DashSwatch dash={e.dash} />
                  {e.label}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}

/** A short line sample drawn with an engine's dash pattern, for the line-style
 * key (colour-independent — the colour key is the format swatches). */
function DashSwatch({ dash }: { dash: number[] }) {
  return (
    <svg className="chart-legend-dash" width="22" height="8" aria-hidden="true">
      <line
        x1="1"
        y1="4"
        x2="21"
        y2="4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        {...(dash.length > 0 ? { strokeDasharray: dash.join(' ') } : {})}
      />
    </svg>
  );
}
