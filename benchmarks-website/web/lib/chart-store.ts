// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Client-side singletons shared by the chart islands, the stateful counterpart
 * to the pure helpers in `lib/chart-format.ts`: the bounded-concurrency fetch
 * queues, the global engine/format filter store, and the per-group filter/Y
 * stores. One instance of each lives per browser tab (module scope), mirroring
 * the page-level closures of `server/static/chart-init.js`.
 *
 * The stores expose a `subscribe`/`getSnapshot` pair compatible with React's
 * `useSyncExternalStore`, so the islands (every `Chart`, the global
 * `FilterBar`, each `GroupToolbar`) re-render and re-apply filters when any of
 * them mutates shared state. Snapshots are replaced wholesale on mutation;
 * a stable reference means "nothing changed".
 *
 * This module holds browser-tab state and is only imported from `'use client'`
 * components; it has no DOM dependency itself, so node-environment vitest can
 * exercise the queues and stores directly.
 */

import {
  FULL_HISTORY_CONCURRENCY,
  GROUP_OPEN_PRIORITY_STEP,
  HYDRATION_CONCURRENCY,
  seedActiveFromAllowlist,
  type FilterUniverse,
  type GlobalFilterState,
} from '@/lib/chart-format';
import type { SeriesTag } from '@/lib/queries';

// ---------------------------------------------------------------------------
// Bounded priority queues (ported from chart-init.js sections 11 and 15).
// ---------------------------------------------------------------------------

/** A queued fetch task; `priority` may be bumped while still queued. */
export interface QueueEntry {
  priority: number;
  /** Resolves/rejects with the task's outcome once the queue runs it. */
  promise: Promise<unknown>;
}

interface InternalEntry extends QueueEntry {
  task: () => Promise<unknown>;
  resolve: (value: unknown) => void;
  reject: (err: unknown) => void;
}

/** A bounded-concurrency, priority-ordered task queue. */
export interface TaskQueue {
  /** Enqueue `task` and start draining; higher `priority` runs first. */
  schedule(task: () => Promise<unknown>, priority?: number): QueueEntry;
  /** Re-sort and start any idle slots (called after a priority bump). */
  drain(): void;
}

function makeQueue(concurrency: number): TaskQueue {
  let active = 0;
  const queue: InternalEntry[] = [];

  function drain(): void {
    while (active < concurrency && queue.length > 0) {
      queue.sort((a, b) => b.priority - a.priority);
      const item = queue.shift();
      if (item === undefined) {
        return;
      }
      active += 1;
      Promise.resolve()
        .then(item.task)
        .then(
          (value) => {
            active -= 1;
            item.resolve(value);
            drain();
          },
          (err: unknown) => {
            active -= 1;
            item.reject(err);
            drain();
          },
        );
    }
  }

  function schedule(task: () => Promise<unknown>, priority = 0): QueueEntry {
    let resolve!: (value: unknown) => void;
    let reject!: (err: unknown) => void;
    const promise = new Promise<unknown>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    const entry: InternalEntry = { task, priority, promise, resolve, reject };
    queue.push(entry);
    drain();
    return entry;
  }

  return { schedule, drain };
}

/** Per-tab queue for the initial latest-100 chart fetches. */
export const hydrationQueue: TaskQueue = makeQueue(HYDRATION_CONCURRENCY);

/** Per-tab queue for the background `?n=all` full-history upgrades. */
export const fullHistoryQueue: TaskQueue = makeQueue(FULL_HISTORY_CONCURRENCY);

let groupOpenPriority = 0;

/**
 * Bump and return the group-open priority. Charts in the most recently opened
 * group enqueue at the highest base priority, so their fetches drain ahead of
 * still-pending work from groups opened earlier.
 */
export function nextGroupOpenPriority(): number {
  groupOpenPriority += GROUP_OPEN_PRIORITY_STEP;
  return groupOpenPriority;
}

// ---------------------------------------------------------------------------
// Global filter store.
// ---------------------------------------------------------------------------

/** The global filter's full state: the chip universe plus the active sets. */
export interface GlobalFilterSnapshot {
  universe: FilterUniverse;
  active: GlobalFilterState;
}

const EMPTY_GLOBAL: GlobalFilterSnapshot = {
  universe: { engines: [], formats: [] },
  active: { engines: [], formats: [] },
};

let globalSnapshot: GlobalFilterSnapshot = EMPTY_GLOBAL;
const globalListeners = new Set<() => void>();

function notifyGlobal(): void {
  for (const cb of globalListeners) {
    cb();
  }
}

/** Subscribe to global-filter changes; returns the unsubscribe function. */
export function subscribeGlobalFilter(cb: () => void): () => void {
  globalListeners.add(cb);
  return () => {
    globalListeners.delete(cb);
  };
}

/** Current global-filter snapshot (stable reference until a mutation). */
export function getGlobalFilterSnapshot(): GlobalFilterSnapshot {
  return globalSnapshot;
}

/**
 * Seed the store from server-provided props: the chip universe plus the URL
 * `?engine=`/`?format=` allowlists (empty allowlist means every chip active).
 * Called by the filter bar on mount and again on soft navigation, so the store
 * tracks the URL state of the page that most recently mounted it.
 */
export function initGlobalFilter(
  universe: FilterUniverse,
  engineAllowlist: readonly string[],
  formatAllowlist: readonly string[],
): void {
  globalSnapshot = {
    universe: { engines: [...universe.engines], formats: [...universe.formats] },
    active: {
      engines: seedActiveFromAllowlist(engineAllowlist, universe.engines),
      formats: seedActiveFromAllowlist(formatAllowlist, universe.formats),
    },
  };
  notifyGlobal();
}

/**
 * Toggle one chip independently. The `'*'` value is the one-shot reset chip:
 * it forces every chip in that dimension back to active. Specific chips flip
 * their own membership in the active set.
 */
export function toggleGlobalFilterValue(dim: 'engine' | 'format', value: string): void {
  const key = dim === 'engine' ? 'engines' : 'formats';
  const active = { ...globalSnapshot.active };
  if (value === '*') {
    active[key] = [...globalSnapshot.universe[key]];
  } else {
    const list = [...active[key]];
    const idx = list.indexOf(value);
    if (idx === -1) {
      list.push(value);
    } else {
      list.splice(idx, 1);
    }
    active[key] = list;
  }
  globalSnapshot = { ...globalSnapshot, active };
  notifyGlobal();
}

// ---------------------------------------------------------------------------
// Per-group stores (ported from chart-init.js section 17's section-node state).
// ---------------------------------------------------------------------------

/** One group's override state plus the series labels its charts have surfaced. */
export interface GroupSnapshot {
  /** Dataset labels the user has toggled off via the group's filter dropdown. */
  hiddenSeries: string[];
  /** The group-level Y override; `null` means "no override; defer to charts". */
  groupY: 'linear' | 'log' | null;
  /** Series labels (with engine/format tags) surfaced by hydrated charts. */
  knownSeries: Record<string, SeriesTag>;
}

const EMPTY_GROUP: GroupSnapshot = { hiddenSeries: [], groupY: null, knownSeries: {} };

interface GroupStore {
  snapshot: GroupSnapshot;
  listeners: Set<() => void>;
}

const groupStores = new Map<string, GroupStore>();

function groupStore(slug: string): GroupStore {
  let store = groupStores.get(slug);
  if (store === undefined) {
    store = { snapshot: EMPTY_GROUP, listeners: new Set() };
    groupStores.set(slug, store);
  }
  return store;
}

function setGroupSnapshot(slug: string, next: GroupSnapshot): void {
  const store = groupStore(slug);
  store.snapshot = next;
  for (const cb of store.listeners) {
    cb();
  }
}

/** Subscribe to one group's changes; returns the unsubscribe function. */
export function subscribeGroup(slug: string, cb: () => void): () => void {
  const store = groupStore(slug);
  store.listeners.add(cb);
  return () => {
    store.listeners.delete(cb);
  };
}

/** Current snapshot for one group (stable reference until a mutation). */
export function getGroupSnapshot(slug: string): GroupSnapshot {
  return groupStore(slug).snapshot;
}

/** The shared empty snapshot, for islands that have no enclosing group. */
export function emptyGroupSnapshot(): GroupSnapshot {
  return EMPTY_GROUP;
}

/** Toggle a single series label in/out of the group's hidden set. */
export function toggleGroupSeries(slug: string, label: string): void {
  const prev = groupStore(slug).snapshot;
  const hiddenSeries = prev.hiddenSeries.includes(label)
    ? prev.hiddenSeries.filter((l) => l !== label)
    : [...prev.hiddenSeries, label];
  setGroupSnapshot(slug, { ...prev, hiddenSeries });
}

/**
 * Apply an engine/format macro click: find every known series whose tag
 * matches. If every match is currently visible, hide them all; otherwise (any
 * match already hidden) show them all, so the macro chip toggles between "all
 * matching visible" and "all matching hidden".
 */
export function applyGroupMacro(slug: string, dim: 'engine' | 'format', value: string): void {
  const prev = groupStore(slug).snapshot;
  const matching = Object.keys(prev.knownSeries).filter(
    (label) => prev.knownSeries[label]?.[dim] === value,
  );
  if (matching.length === 0) {
    return;
  }
  const allVisible = matching.every((l) => !prev.hiddenSeries.includes(l));
  const hiddenSeries = allVisible
    ? [...prev.hiddenSeries, ...matching.filter((l) => !prev.hiddenSeries.includes(l))]
    : prev.hiddenSeries.filter((l) => !matching.includes(l));
  setGroupSnapshot(slug, { ...prev, hiddenSeries });
}

/** Clear the group's series filter (the `'*'` reset chip). */
export function clearGroupSeriesFilter(slug: string): void {
  const prev = groupStore(slug).snapshot;
  setGroupSnapshot(slug, { ...prev, hiddenSeries: [] });
}

/** Set the group-level Y override broadcast to non-sticky charts. */
export function setGroupY(slug: string, y: 'linear' | 'log'): void {
  const prev = groupStore(slug).snapshot;
  setGroupSnapshot(slug, { ...prev, groupY: y });
}

/** Reset the group: empty series filter and no Y override. Per-card legend
 * overrides and per-card sticky Y choices are intentionally NOT cleared,
 * matching the v3 reset semantics. */
export function resetGroup(slug: string): void {
  const prev = groupStore(slug).snapshot;
  setGroupSnapshot(slug, { ...prev, hiddenSeries: [], groupY: null });
}

/**
 * Fold a hydrated chart's `series_meta` labels into the group's running set
 * (idempotent), so the group toolbar's series chip row grows as charts in the
 * group hydrate. No-op when every label is already known.
 */
export function noteGroupSeries(slug: string, meta: Record<string, SeriesTag> | undefined): void {
  if (!meta) {
    return;
  }
  const prev = groupStore(slug).snapshot;
  let added = false;
  const knownSeries = { ...prev.knownSeries };
  for (const [label, tag] of Object.entries(meta)) {
    if (!(label in knownSeries)) {
      knownSeries[label] = tag ?? {};
      added = true;
    }
  }
  if (added) {
    setGroupSnapshot(slug, { ...prev, knownSeries });
  }
}
