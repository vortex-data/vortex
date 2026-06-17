// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Cluster the discovered groups for the Historic Data page, the TypeScript port
 * of the TPC fan-out in `server/src/html/mod.rs::collect_landing_groups`.
 *
 * TPC query suites are ingested as one group per `(storage, scale_factor)` pair
 * (`TPC-H (NVMe) (SF=1)`, `TPC-H (S3) (SF=10)`, …). This clusters them by
 * `(dataset, dataset_variant)` into a single suite whose storage and scale
 * factor become in-place toggle buttons, so the page shows one `TPC-H` section
 * instead of a row per combination. Everything else (random access, compression,
 * ClickBench — which has no scale factor — vector search) passes through as a
 * plain group.
 */

import { tpcSuiteDescription } from './descriptions';
import type { ChartLink, Group } from './queries';
import { groupKeyFromSlug } from './slug';
import type { Summary } from './summary';

const STORAGE_LABELS: Record<string, string> = { nvme: 'NVMe', s3: 'S3' };

function storageLabel(storage: string): string {
  return STORAGE_LABELS[storage] ?? storage;
}

/** Canonical storage button order: NVMe before S3, unknowns last. */
function storageOrder(storage: string): number {
  return storage === 'nvme' ? 0 : storage === 's3' ? 1 : 2;
}

/** Format a scale factor without a trailing `.0` (`10.0` -> `10`). */
function fmtScale(sf: number): string {
  return Number.isInteger(sf) ? String(sf) : String(sf);
}

/** Drop the trailing ` (NVMe|S3)` and ` (SF=N)` parentheticals from a TPC group
 * name — both become in-section toggles, so the heading reads as the bare suite
 * (`TPC-H`, `TPC-DS`). Ported from `current.rs::strip_tpc_parentheticals`. */
function stripTpcParentheticals(name: string): string {
  let s = name;
  const sfIdx = s.lastIndexOf(' (SF=');
  if (sfIdx !== -1) {
    s = s.slice(0, sfIdx);
  }
  for (const tier of [' (NVMe)', ' (S3)']) {
    if (s.endsWith(tier)) {
      return s.slice(0, -tier.length);
    }
  }
  return s;
}

/** One `(storage, scale-factor)` panel of a TPC suite. */
export interface TpcPill {
  slug: string;
  storage: string;
  sfValue: string;
  sfLabel: string;
  charts: ChartLink[];
  summary?: Summary;
  current: boolean;
}

/** A clustered TPC suite: a name, the toggle dimensions, and one panel per
 * `(storage, scale-factor)` combination present. */
export interface TpcSuite {
  name: string;
  slug: string;
  description?: string;
  storages: { value: string; label: string }[];
  scaleFactors: { value: string; label: string }[];
  pills: TpcPill[];
}

/** A Historic-page section: either a plain group or a clustered TPC suite. */
export type HistoricSection = { kind: 'group'; group: Group } | { kind: 'tpc'; suite: TpcSuite };

interface Variant {
  sfStr: string;
  sfNum: number;
  storage: string;
  group: Group;
}

/**
 * Partition groups into plain sections and clustered TPC suites, preserving the
 * order groups arrive in (a cluster takes its first variant's slot). A TPC
 * suite's default panel is NVMe (when present) at its largest scale factor — the
 * published-headline combination.
 */
export function clusterHistoricGroups(groups: readonly Group[]): HistoricSection[] {
  const slots: ({ kind: 'group'; group: Group } | { kind: 'cluster'; key: string })[] = [];
  const clusters = new Map<string, Variant[]>();

  for (const group of groups) {
    let parsed;
    try {
      parsed = groupKeyFromSlug(group.slug);
    } catch {
      slots.push({ kind: 'group', group });
      continue;
    }
    if (parsed.k === 'QueryGroup' && parsed.scale_factor !== null) {
      const key = JSON.stringify([parsed.dataset, parsed.dataset_variant]);
      let entry = clusters.get(key);
      if (entry === undefined) {
        entry = [];
        clusters.set(key, entry);
        slots.push({ kind: 'cluster', key });
      }
      entry.push({
        sfStr: parsed.scale_factor,
        sfNum: Number.parseFloat(parsed.scale_factor) || 0,
        storage: parsed.storage,
        group,
      });
    } else {
      slots.push({ kind: 'group', group });
    }
  }

  const out: HistoricSection[] = [];
  for (const slot of slots) {
    if (slot.kind === 'group') {
      out.push({ kind: 'group', group: slot.group });
      continue;
    }
    const variants = clusters.get(slot.key);
    if (variants === undefined || variants.length === 0) {
      continue;
    }
    // Default storage: NVMe when present, else the first storage seen.
    const defaultStorage = variants.some((v) => v.storage === 'nvme')
      ? 'nvme'
      : variants[0].storage;
    // Default SF: largest available under the default storage.
    let defaultSf = Number.NEGATIVE_INFINITY;
    for (const v of variants) {
      if (v.storage === defaultStorage && v.sfNum > defaultSf) {
        defaultSf = v.sfNum;
      }
    }
    const rep =
      variants.find((v) => v.storage === defaultStorage && v.sfNum === defaultSf) ?? variants[0];

    // Pills sorted by (storage, SF) for a stable button-derived order.
    const sorted = [...variants].sort(
      (a, b) => storageOrder(a.storage) - storageOrder(b.storage) || a.sfNum - b.sfNum,
    );
    const pills: TpcPill[] = sorted.map((v) => ({
      slug: v.group.slug,
      storage: v.storage,
      sfValue: v.sfStr,
      sfLabel: `SF${fmtScale(v.sfNum)}`,
      charts: v.group.charts,
      summary: v.group.summary,
      current: v.group.slug === rep.group.slug,
    }));

    // Distinct storage buttons (NVMe first) and SF buttons (smallest first).
    const storages: { value: string; label: string }[] = [];
    const seenStorage = new Set<string>();
    for (const v of sorted) {
      if (!seenStorage.has(v.storage)) {
        seenStorage.add(v.storage);
        storages.push({ value: v.storage, label: storageLabel(v.storage) });
      }
    }
    const scaleFactors: { value: string; label: string }[] = [];
    const seenSf = new Set<string>();
    for (const v of [...variants].sort((a, b) => a.sfNum - b.sfNum)) {
      if (!seenSf.has(v.sfStr)) {
        seenSf.add(v.sfStr);
        scaleFactors.push({ value: v.sfStr, label: `SF${fmtScale(v.sfNum)}` });
      }
    }

    const suiteName = stripTpcParentheticals(rep.group.name);
    out.push({
      kind: 'tpc',
      suite: {
        name: suiteName,
        slug: rep.group.slug,
        description: tpcSuiteDescription(suiteName) ?? undefined,
        storages,
        scaleFactors,
        pills,
      },
    });
  }
  return out;
}
