// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * The "synthesis" aggregation layer powering the Overview (Showcase) and Latest
 * Commit (head-to-head) facings — the TypeScript port of
 * `server/src/html/current.rs` + `server/src/html/showcase.rs`.
 *
 * Every chart on those pages answers one controlled comparison: holding dataset,
 * engine, operation and scale constant, vary only the format. For each *item* (a
 * query, dataset, or access pattern) we take the `B / A` ratio of two formats
 * (default Vortex vs Parquet) and reduce a suite to a single geometric mean.
 *
 * These functions consume the same [`ChartResponse`] shape the read API serves
 * (`series` value arrays + per-series `series_meta` engine/format tags + the
 * commit timeline), so the numbers are sourced from exactly the data v4 already
 * exposes — the speedup-distribution `EngineData`, the per-commit `HistoryData`
 * trajectory, and the Overview's per-facet `FacetGeomean` headline.
 */

import { compareCodeUnits } from './families';
import {
  chartPayload,
  collectGroups,
  type ChartLink,
  type Group,
  type NamedChartResponse,
  type UnitKind,
} from './queries';
import { chartKeyFromSlug, groupKeyFromSlug } from './slug';
import type { CommitWindow } from './window';

/** The canonical "Vortex" format the published Vortex-vs-Parquet numbers use. */
export const VORTEX_FORMAT = 'vortex-file-compressed';
/** The baseline format every comparison races against. */
export const PARQUET_FORMAT = 'parquet';

/** Comparison verb derived from a unit; both times and sizes are lower-is-better. */
export type Metric = 'faster' | 'smaller';

/** One format option for a comparison dropdown. */
export interface FormatOpt {
  /** Physical format id (`vortex-file-compressed`, `parquet`, …). */
  id: string;
  /** Human label shown in the dropdown / axis / headline. */
  label: string;
}

/**
 * One item's value for every format measured for it under one facet. The client
 * divides the two selected formats to get the per-item ratio. `v` is the raw
 * value (base unit) keyed by format id; `d` is the server-formatted display.
 */
export interface QueryRow {
  query: string;
  v: Record<string, number>;
  d: Record<string, string>;
}

/**
 * One facet's full per-item/per-format matrix, emitted to the client. The two
 * comparison dropdowns pick any pair of `formats`; the chart shows the `B / A`
 * ratio per item. Defaults reproduce the Vortex-vs-Parquet view. The JSON field
 * names (`defaultA` / `defaultB`) match the synthesis wire contract that
 * `chart-init.js`'s speedup renderer reads.
 */
export interface EngineData {
  /** Engine, operation, or `""` for the no-facet (single-chart) groups. */
  facet: string;
  metric: Metric;
  formats: FormatOpt[];
  defaultA: string;
  defaultB: string;
  queries: QueryRow[];
}

/**
 * One facet's Overview headline: the geomean of the default Vortex-vs-Parquet
 * ratios (`B / A`, so > 1 = Vortex wins), with its win count and item total.
 * Exactly the number the Latest page headlines for the same facet, so the
 * Overview claim and its proof cannot disagree.
 */
export interface FacetGeomean {
  facet: string;
  geomean: number;
  wins: number;
  total: number;
}

/** One commit on the headline chart's x-axis. */
export interface HistoryCommit {
  sha: string;
  msg: string;
  timestamp: string;
}

/** One line on the headline chart: a format's geomean speedup vs Parquet under
 * one facet, at each commit (`null` where that commit has no data). */
export interface HistoryLine {
  label: string;
  facet: string;
  engine: string;
  format: string;
  speedups: (number | null)[];
}

/** Inline payload for the Previous-Versions headline line chart. */
export interface HistoryData {
  metric: Metric;
  commits: HistoryCommit[];
  lines: HistoryLine[];
}

// ---------------------------------------------------------------------------
// Pure helpers (ported 1:1 from current.rs / showcase.rs)
// ---------------------------------------------------------------------------

/** Latest finite value in a series' value array (arrays run oldest-first). */
export function latestValue(values: readonly (number | null)[]): number | undefined {
  for (let i = values.length - 1; i >= 0; i--) {
    const v = values[i];
    if (v !== null && Number.isFinite(v)) {
      return v;
    }
  }
  return undefined;
}

/** Comparison verb for a unit. */
export function metricFor(unit: UnitKind): Metric {
  return unit === 'bytes' ? 'smaller' : 'faster';
}

/** The facet a series belongs to: its engine (query suites), else the operation
 * suffix of a `format:op` name (compression), else `""` (random access, size). */
export function facetOf(name: string, engine: string | undefined): string {
  if (engine !== undefined) {
    return engine;
  }
  const idx = name.lastIndexOf(':');
  return idx === -1 ? '' : name.slice(idx + 1);
}

/** Sort key for a format in the comparison dropdowns: Vortex variants first. */
export function formatOrder(id: string): number {
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

/** Human label for a physical format id. */
export function formatLabel(id: string): string {
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

/** Pretty label for a facet name (engine or op). */
export function facetLabel(facet: string): string {
  switch (facet) {
    case 'datafusion':
      return 'DataFusion';
    case 'duckdb':
      return 'DuckDB';
    case 'encode':
      return 'Encode';
    case 'decode':
      return 'Decode';
    default:
      return facet;
  }
}

/** Headline-split key for one line: engine when set, else facet. */
export function historySplitKey(line: HistoryLine): string {
  return line.engine !== '' ? line.engine : line.facet;
}

/** Pretty label for a headline-split key (engine or op). */
export function prettySplitLabel(key: string): string {
  switch (key) {
    case 'datafusion':
      return 'DataFusion';
    case 'duckdb':
      return 'DuckDB';
    case 'encode':
      return 'Encode';
    case 'decode':
      return 'Decode';
    default:
      return key;
  }
}

/** Server-formatted value for a unit (the tooltip display), ported from
 * `showcase.rs::format_value`. */
export function formatValue(v: number, unit: UnitKind): string {
  switch (unit) {
    case 'time_ns':
      return formatTimeNs(v);
    case 'bytes':
      return formatBytes(v);
    case 'throughput_mb_s':
      return `${v.toFixed(0)} MB/s`;
    case 'ratio':
    case 'count':
      return v.toFixed(2);
  }
}

function formatTimeNs(ns: number): string {
  const abs = Math.abs(ns);
  if (abs >= 1e9) {
    return `${(ns / 1e9).toFixed(1)} s`;
  }
  if (abs >= 1e6) {
    return `${(ns / 1e6).toFixed(1)} ms`;
  }
  if (abs >= 1e3) {
    return `${(ns / 1e3).toFixed(1)} µs`;
  }
  return `${ns.toFixed(0)} ns`;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) {
    return `${(bytes / 1_073_741_824).toFixed(1)} GiB`;
  }
  if (bytes >= 1_048_576) {
    return `${(bytes / 1_048_576).toFixed(1)} MiB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${bytes.toFixed(0)} B`;
}

/** Geometric mean of positive, finite values, or `null` when none qualify. */
export function geomeanOf(ratios: readonly number[]): number | null {
  let sumLn = 0;
  let n = 0;
  for (const r of ratios) {
    if (r > 0 && Number.isFinite(r)) {
      sumLn += Math.log(r);
      n += 1;
    }
  }
  return n > 0 ? Math.exp(sumLn / n) : null;
}

/** Format a multiplier: whole-number for big ratios (`80×`), two decimals close
 * in (`1.30×`). Ported from `showcase.rs::mult`. */
export function mult(v: number): string {
  return v >= 10 ? `${v.toFixed(0)}×` : `${v.toFixed(2)}×`;
}

/** Stable section anchor for a group slug, matching `mod.rs::anchor_for` so the
 * Overview's deep links land on the Latest page's section ids. */
export function anchorFor(slug: string): string {
  let s = 'g-';
  for (const c of slug) {
    s += /[a-zA-Z0-9]/.test(c) ? c : '-';
  }
  return s;
}

// ---------------------------------------------------------------------------
// Speedup distribution (the synthesis model, used by every group)
// ---------------------------------------------------------------------------

/**
 * Build the per-facet comparison data for a group's charts: one [`EngineData`]
 * per facet (engine / operation / `""`). Returns the facets and whether any were
 * engine-faceted (so callers can label the natural sort order). A facet needs
 * two distinct formats to be comparable. Ported from `current.rs::build_facets`.
 */
export function buildFacets(charts: readonly NamedChartResponse[]): {
  facets: EngineData[];
  facetedByEngine: boolean;
} {
  const perFacet = new Map<string, { formats: Set<string>; queries: QueryRow[] }>();
  let unit: UnitKind = 'time_ns';
  let facetedByEngine = false;

  for (const chart of charts) {
    unit = chart.unit_kind;
    const meta = chart.series_meta ?? {};
    const rows = new Map<string, QueryRow>();
    for (const [name, tag] of Object.entries(meta)) {
      const format = tag.format;
      if (format === undefined) {
        continue;
      }
      const series = chart.series[name];
      const v = series === undefined ? undefined : latestValue(series);
      if (v === undefined || v <= 0) {
        continue;
      }
      if (tag.engine !== undefined) {
        facetedByEngine = true;
      }
      const facet = facetOf(name, tag.engine);
      let row = rows.get(facet);
      if (row === undefined) {
        row = { query: chart.name, v: {}, d: {} };
        rows.set(facet, row);
      }
      row.v[format] = v;
      row.d[format] = formatValue(v, unit);
    }
    for (const [facet, row] of rows) {
      if (Object.keys(row.v).length === 0) {
        continue;
      }
      let acc = perFacet.get(facet);
      if (acc === undefined) {
        acc = { formats: new Set(), queries: [] };
        perFacet.set(facet, acc);
      }
      for (const f of Object.keys(row.v)) {
        acc.formats.add(f);
      }
      acc.queries.push(row);
    }
  }

  const metric = metricFor(unit);
  const facets: EngineData[] = [];
  // Sorted facet iteration matches the Rust `BTreeMap` order.
  for (const facet of [...perFacet.keys()].sort(compareCodeUnits)) {
    const acc = perFacet.get(facet);
    if (acc === undefined || acc.formats.size < 2 || acc.queries.length === 0) {
      continue;
    }
    const formats = [...acc.formats].sort(
      (a, b) => formatOrder(a) - formatOrder(b) || compareCodeUnits(a, b),
    );
    const defaultA =
      formats.find((f) => f === VORTEX_FORMAT) ??
      formats.find((f) => f.includes('vortex')) ??
      formats[0] ??
      '';
    const defaultB =
      formats.find((f) => f === PARQUET_FORMAT && f !== defaultA) ??
      formats.find((f) => f !== defaultA) ??
      defaultA;
    facets.push({
      facet,
      metric,
      formats: formats.map((id) => ({ id, label: formatLabel(id) })),
      defaultA,
      defaultB,
      queries: acc.queries,
    });
  }
  return { facets, facetedByEngine };
}

/**
 * Per-facet Overview geomeans for a group's charts — one per engine / operation
 * / the single no-facet chart. Facets with no comparable items are skipped.
 * Ported from `current.rs::facet_geomeans`.
 */
export function facetGeomeans(charts: readonly NamedChartResponse[]): FacetGeomean[] {
  const { facets } = buildFacets(charts);
  const out: FacetGeomean[] = [];
  for (const ed of facets) {
    const ratios: number[] = [];
    let wins = 0;
    for (const q of ed.queries) {
      const a = q.v[ed.defaultA];
      const b = q.v[ed.defaultB];
      if (a === undefined || b === undefined) {
        continue;
      }
      if (a > 0 && b > 0) {
        const ratio = b / a;
        if (ratio >= 1) {
          wins += 1;
        }
        ratios.push(ratio);
      }
    }
    if (ratios.length === 0) {
      continue;
    }
    const geomean = Math.exp(ratios.reduce((s, r) => s + Math.log(r), 0) / ratios.length);
    out.push({ facet: ed.facet, geomean, wins, total: ratios.length });
  }
  return out;
}

/** Geomean pooled across every facet's items: `exp(Σ nᵢ·ln gᵢ / Σ nᵢ)`. Ported
 * from `showcase.rs::pooled_geomean`. */
export function pooledGeomean(facets: readonly FacetGeomean[]): number | null {
  let total = 0;
  for (const f of facets) {
    total += f.total;
  }
  if (total === 0) {
    return null;
  }
  let logSum = 0;
  for (const f of facets) {
    logSum += f.total * Math.log(f.geomean);
  }
  return Math.exp(logSum / total);
}

// ---------------------------------------------------------------------------
// Synthesis over time (Previous Versions headline chart)
// ---------------------------------------------------------------------------

/**
 * Compute, for every `(facet, non-Parquet format)`, the per-commit geomean of
 * `parquet / format` across a group's charts — each becomes one headline line.
 * Parquet is the implicit 1× baseline. Ported from `current.rs::build_history`.
 */
export function buildHistory(charts: readonly NamedChartResponse[]): HistoryData | null {
  const acc = new Map<string, { facet: string; fmt: string; bySha: Map<string, number[]> }>();
  const order: string[] = []; // full shas, oldest first, deduped
  const labels = new Map<string, string>();
  const shorts = new Map<string, string>();
  const times = new Map<string, string>();
  const seen = new Set<string>();
  const engineFacets = new Set<string>();
  let unit: UnitKind = 'time_ns';
  const keyOf = (facet: string, fmt: string): string => `${facet} ${fmt}`;

  for (const chart of charts) {
    unit = chart.unit_kind;
    for (const c of chart.commits) {
      if (!seen.has(c.sha)) {
        seen.add(c.sha);
        order.push(c.sha);
        shorts.set(c.sha, c.sha.slice(0, 7));
        labels.set(c.sha, c.message);
        times.set(c.sha, c.timestamp);
      }
    }
    const meta = chart.series_meta ?? {};
    const byFacet = new Map<string, Map<string, string>>();
    for (const [name, tag] of Object.entries(meta)) {
      const fmt = tag.format;
      if (fmt === undefined) {
        continue;
      }
      const facet = facetOf(name, tag.engine);
      if (tag.engine !== undefined) {
        engineFacets.add(facet);
      }
      let m = byFacet.get(facet);
      if (m === undefined) {
        m = new Map();
        byFacet.set(facet, m);
      }
      m.set(fmt, name);
    }
    for (const [facet, formats] of byFacet) {
      const pname = formats.get(PARQUET_FORMAT);
      if (pname === undefined) {
        continue; // no baseline in this facet
      }
      const parr = chart.series[pname];
      if (parr === undefined) {
        continue;
      }
      for (const [fmt, sname] of formats) {
        if (fmt === PARQUET_FORMAT) {
          continue;
        }
        const sarr = chart.series[sname];
        if (sarr === undefined) {
          continue;
        }
        for (let i = 0; i < chart.commits.length; i++) {
          const pv = parr[i];
          const sv = sarr[i];
          if (pv === null || pv === undefined || sv === null || sv === undefined) {
            continue;
          }
          if (pv > 0 && sv > 0) {
            const k = keyOf(facet, fmt);
            let entry = acc.get(k);
            if (entry === undefined) {
              entry = { facet, fmt, bySha: new Map() };
              acc.set(k, entry);
            }
            const sha = chart.commits[i].sha;
            let arr = entry.bySha.get(sha);
            if (arr === undefined) {
              arr = [];
              entry.bySha.set(sha, arr);
            }
            arr.push(pv / sv);
          }
        }
      }
    }
  }

  if (acc.size === 0 || order.length === 0) {
    return null;
  }
  const entries = [...acc.values()];
  const faceted = entries.some((e) => e.facet !== '');
  const multiFormat = new Set(entries.map((e) => e.fmt)).size > 1;
  // Order lines by format (Vortex first) then facet, so colour assignment is stable.
  entries.sort(
    (a, b) => formatOrder(a.fmt) - formatOrder(b.fmt) || compareCodeUnits(a.facet, b.facet),
  );

  const commits: HistoryCommit[] = order.map((sha) => ({
    sha: shorts.get(sha) ?? '',
    msg: labels.get(sha) ?? '',
    timestamp: times.get(sha) ?? '',
  }));
  const lines: HistoryLine[] = entries.map(({ facet, fmt, bySha }) => {
    const parts: string[] = [];
    if (faceted && facet !== '') {
      parts.push(facetLabel(facet));
    }
    if (multiFormat) {
      parts.push(formatLabel(fmt));
    }
    const label = parts.length === 0 ? formatLabel(fmt) : parts.join(' · ');
    const speedups = order.map((sha) => {
      const arr = bySha.get(sha);
      return arr === undefined ? null : geomeanOf(arr);
    });
    const engine = engineFacets.has(facet) ? facet : '';
    return { label, facet, engine, format: fmt, speedups };
  });
  return { metric: metricFor(unit), commits, lines };
}

// ---------------------------------------------------------------------------
// DB-backed collection (the page-level entry points)
// ---------------------------------------------------------------------------

/** Fetch the full-history payloads for a group's chart links, dropping links
 * whose chart has no rows. The synthesis built over `generation.chart_payload`;
 * v4 resolves each via `chartPayload(chartKeyFromSlug(slug))`. */
async function fetchGroupCharts(
  links: readonly ChartLink[],
  window: CommitWindow,
): Promise<NamedChartResponse[]> {
  const out: NamedChartResponse[] = [];
  for (const link of links) {
    const payload = await chartPayload(chartKeyFromSlug(link.slug), window);
    if (payload === null) {
      continue;
    }
    out.push({ name: link.name, slug: link.slug, ...payload });
  }
  return out;
}

/** One group's worth of head-to-head data for the Latest page. */
export interface SpeedupGroup {
  name: string;
  slug: string;
  anchor: string;
  description?: string;
  facets: EngineData[];
  facetedByEngine: boolean;
  metric: Metric;
  history: HistoryData | null;
}

/**
 * The Latest page's groups: every discovered group, with its per-facet
 * speedup-distribution matrices and per-commit history. Groups with no
 * comparable facet (fewer than two formats) are dropped. TPC suites currently
 * render one section per `(storage, scale-factor)` group; the synthesis storage
 * × SF pill clustering is a follow-up.
 */
export async function collectSpeedupGroups(): Promise<SpeedupGroup[]> {
  const groups = await collectGroups();
  const out: SpeedupGroup[] = [];
  for (const group of groups) {
    const charts = await fetchGroupCharts(group.charts, { kind: 'all' });
    const { facets, facetedByEngine } = buildFacets(charts);
    if (facets.length === 0) {
      continue;
    }
    out.push({
      name: group.name,
      slug: group.slug,
      anchor: anchorFor(group.slug),
      description: group.description,
      facets,
      facetedByEngine,
      metric: facets[0].metric,
      history: buildHistory(charts),
    });
  }
  return out;
}

const RANDOM_ACCESS_WHY =
  'Point lookups — reading specific rows by position — drive feature stores, vector search, and ' +
  'anything that serves individual records. Parquet packs rows into large row groups that must be ' +
  'decoded almost whole to return a single value, so one row costs as much as thousands. Vortex ' +
  'addresses rows directly.';
const ANALYTICS_WHY =
  'Dashboards, reports, and the read side of ETL are mostly column scans and aggregations. ' +
  "ClickBench — ClickHouse's 43-query suite over real web-analytics data — is the field's standard " +
  'test. Vortex is a drop-in: keep your engine, swap Parquet (or the engine’s own native format) ' +
  'for a Vortex file, and the same queries return faster — on DataFusion and DuckDB alike.';
const WRITES_WHY =
  'Data is encoded once and read many times, but the encode step gates ingestion — how quickly new ' +
  'data becomes queryable. Parquet spends heavily in its row-group encoder; Vortex encodes the same ' +
  'data faster, so pipelines clear backlogs and data goes live sooner.';
const SIZE_WHY =
  'Storage cost and the bytes a query must move both track file size, so a faster format that ' +
  "bloated on disk would trade one bill for another. Vortex holds within a few percent of Parquet's " +
  'compression ratio — the speed above comes at no size penalty.';

/** The workload schematic a claim renders (a small blueprint SVG of the access
 * pattern). The Showcase component maps this to the matching figure. */
export type Workload = 'randomAccess' | 'analytics' | 'writes' | 'size';

/** One Overview claim: a live geomean headline beside a workload schematic. */
export interface ShowcaseClaim {
  hero: string;
  label: string;
  detail: string | null;
  why: string;
  workload: Workload;
  href: string | null;
}

/** Pick the representative query group for a dataset (NVMe-preferred, largest
 * SF), matching `showcase.rs::query_group` so deep links land on the right
 * Latest-page section. */
function queryGroup(groups: readonly Group[], dataset: string): Group | undefined {
  const storageRank = (s: string): number => (s === 'nvme' ? 0 : s === 's3' ? 1 : 2);
  let best: { sr: number; sf: number; group: Group } | undefined;
  for (const g of groups) {
    let key;
    try {
      key = groupKeyFromSlug(g.slug);
    } catch {
      continue;
    }
    if (key.k !== 'QueryGroup' || key.dataset !== dataset) {
      continue;
    }
    const sf = key.scale_factor !== null ? Number.parseFloat(key.scale_factor) || 0 : 0;
    const sr = storageRank(key.storage);
    if (best === undefined || sr < best.sr || (sr === best.sr && sf > best.sf)) {
      best = { sr, sf, group: g };
    }
  }
  return best?.group;
}

function pickFacet(facets: readonly FacetGeomean[], facet: string): FacetGeomean | undefined {
  return facets.find((f) => f.facet === facet);
}

function moreHref(group: Group | undefined): string | null {
  return group !== undefined ? `/latest#${anchorFor(group.slug)}` : null;
}

/**
 * The Overview's four live Vortex-vs-Parquet claims, each sourced from the same
 * `facetGeomeans` the Latest page renders. Ported from `showcase.rs`.
 */
export async function collectShowcaseClaims(): Promise<ShowcaseClaim[]> {
  const groups = await collectGroups();
  const randomAccess = groups.find((g) => g.summary?.type === 'randomAccess');
  const compression = groups.find((g) => g.summary?.type === 'compression');
  const size = groups.find((g) => g.summary?.type === 'compressionSize');
  const clickbench = queryGroup(groups, 'clickbench');

  const geomeansFor = async (group: Group | undefined): Promise<FacetGeomean[]> =>
    group === undefined ? [] : facetGeomeans(await fetchGroupCharts(group.charts, { kind: 'all' }));

  const raFacets = await geomeansFor(randomAccess);
  const ra = pickFacet(raFacets, '');
  const randomAccessClaim: ShowcaseClaim = {
    hero: ra !== undefined ? mult(ra.geomean) : '—',
    label: 'faster random access',
    detail: ra !== undefined ? `Vortex wins ${ra.wins}/${ra.total}` : null,
    why: RANDOM_ACCESS_WHY,
    workload: 'randomAccess',
    href: moreHref(randomAccess),
  };

  const cbFacets = await geomeansFor(clickbench);
  const pooled = pooledGeomean(cbFacets);
  const df = pickFacet(cbFacets, 'datafusion');
  const duck = pickFacet(cbFacets, 'duckdb');
  let analyticsDetail = 'ClickBench';
  if (df !== undefined) {
    analyticsDetail += ` · DataFusion ${mult(df.geomean)}`;
  }
  if (duck !== undefined) {
    analyticsDetail += ` · DuckDB ${mult(duck.geomean)}`;
  }
  const analyticsClaim: ShowcaseClaim = {
    hero: pooled !== null ? mult(pooled) : '—',
    label: 'faster data analytics',
    detail: analyticsDetail,
    why: ANALYTICS_WHY,
    workload: 'analytics',
    href: moreHref(clickbench),
  };

  const ctFacets = await geomeansFor(compression);
  const enc = pickFacet(ctFacets, 'encode');
  const writesClaim: ShowcaseClaim = {
    hero: enc !== undefined ? mult(enc.geomean) : '—',
    label: 'faster writes',
    detail: enc !== undefined ? `Compression encode · Vortex wins ${enc.wins}/${enc.total}` : null,
    why: WRITES_WHY,
    workload: 'writes',
    href: moreHref(compression),
  };

  const szFacets = await geomeansFor(size);
  const sz = pickFacet(szFacets, '');
  const sizeClaim: ShowcaseClaim = {
    hero: sz !== undefined && sz.geomean > 0 ? mult(1 / sz.geomean) : '—',
    label: 'the size of Parquet',
    detail: sz !== undefined ? `geomean across ${sz.total} datasets` : null,
    why: SIZE_WHY,
    workload: 'size',
    href: moreHref(size),
  };

  return [randomAccessClaim, analyticsClaim, writesClaim, sizeClaim];
}
