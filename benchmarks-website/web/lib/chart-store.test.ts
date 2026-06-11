// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';

import {
  applyGroupMacro,
  clearGroupSeriesFilter,
  getGlobalFilterSnapshot,
  getGroupSnapshot,
  hydrationQueue,
  initGlobalFilter,
  noteGroupSeries,
  resetGroup,
  setGroupY,
  subscribeGlobalFilter,
  subscribeGroup,
  toggleGlobalFilterValue,
  toggleGroupSeries,
} from './chart-store';

// The stores are module-scope singletons (one per tab, one per test file run);
// each describe block uses its own group slug so state does not leak between
// tests, and the global-filter tests re-init the store as a fresh page mount
// would.

const UNIVERSE = { engines: ['datafusion', 'duckdb'], formats: ['parquet', 'vortex'] };

describe('global filter store', () => {
  it('seeds every chip active with no URL allowlist', () => {
    initGlobalFilter(UNIVERSE, [], []);
    const snap = getGlobalFilterSnapshot();
    expect(snap.active.engines).toEqual(['datafusion', 'duckdb']);
    expect(snap.active.formats).toEqual(['parquet', 'vortex']);
  });

  it('seeds verbatim from a URL allowlist and toggles chips independently', () => {
    initGlobalFilter(UNIVERSE, ['duckdb'], []);
    expect(getGlobalFilterSnapshot().active.engines).toEqual(['duckdb']);

    toggleGlobalFilterValue('engine', 'datafusion');
    expect(getGlobalFilterSnapshot().active.engines).toEqual(['duckdb', 'datafusion']);

    toggleGlobalFilterValue('engine', 'duckdb');
    expect(getGlobalFilterSnapshot().active.engines).toEqual(['datafusion']);
  });

  it('resets a dimension to all-active via the "*" chip', () => {
    initGlobalFilter(UNIVERSE, ['duckdb'], ['vortex']);
    toggleGlobalFilterValue('format', '*');
    expect(getGlobalFilterSnapshot().active.formats).toEqual(['parquet', 'vortex']);
    // The engine dimension is untouched by a format reset.
    expect(getGlobalFilterSnapshot().active.engines).toEqual(['duckdb']);
  });

  it('notifies subscribers with a fresh snapshot reference per mutation', () => {
    initGlobalFilter(UNIVERSE, [], []);
    const before = getGlobalFilterSnapshot();
    let notified = 0;
    const unsubscribe = subscribeGlobalFilter(() => {
      notified += 1;
    });
    toggleGlobalFilterValue('engine', 'duckdb');
    expect(notified).toBe(1);
    expect(getGlobalFilterSnapshot()).not.toBe(before);
    unsubscribe();
    toggleGlobalFilterValue('engine', 'duckdb');
    expect(notified).toBe(1);
  });
});

describe('per-group store', () => {
  it('accumulates known series idempotently and notifies', () => {
    const slug = 'group-known-series';
    let notified = 0;
    subscribeGroup(slug, () => {
      notified += 1;
    });
    noteGroupSeries(slug, { 'duckdb:parquet': { engine: 'duckdb', format: 'parquet' } });
    expect(notified).toBe(1);
    // Re-noting the same labels is a no-op (no notification, same snapshot).
    const snap = getGroupSnapshot(slug);
    noteGroupSeries(slug, { 'duckdb:parquet': { engine: 'duckdb', format: 'parquet' } });
    expect(notified).toBe(1);
    expect(getGroupSnapshot(slug)).toBe(snap);
  });

  it('toggles single series in and out of the hidden set', () => {
    const slug = 'group-series-toggle';
    toggleGroupSeries(slug, 'a');
    expect(getGroupSnapshot(slug).hiddenSeries).toEqual(['a']);
    toggleGroupSeries(slug, 'a');
    expect(getGroupSnapshot(slug).hiddenSeries).toEqual([]);
  });

  it('bulk-toggles matching series via engine/format macros', () => {
    const slug = 'group-macros';
    noteGroupSeries(slug, {
      'duckdb:parquet': { engine: 'duckdb', format: 'parquet' },
      'duckdb:vortex': { engine: 'duckdb', format: 'vortex' },
      'datafusion:vortex': { engine: 'datafusion', format: 'vortex' },
    });
    // All duckdb series visible: the macro hides them all.
    applyGroupMacro(slug, 'engine', 'duckdb');
    expect(getGroupSnapshot(slug).hiddenSeries.sort()).toEqual(['duckdb:parquet', 'duckdb:vortex']);
    // Any match hidden: the macro shows them all.
    applyGroupMacro(slug, 'engine', 'duckdb');
    expect(getGroupSnapshot(slug).hiddenSeries).toEqual([]);
    // A macro with no matching series is inert.
    applyGroupMacro(slug, 'engine', 'unknown-engine');
    expect(getGroupSnapshot(slug).hiddenSeries).toEqual([]);
  });

  it('clears the series filter via the "*" chip without touching Y', () => {
    const slug = 'group-clear';
    toggleGroupSeries(slug, 'a');
    setGroupY(slug, 'log');
    clearGroupSeriesFilter(slug);
    expect(getGroupSnapshot(slug).hiddenSeries).toEqual([]);
    expect(getGroupSnapshot(slug).groupY).toBe('log');
  });

  it('resets the filter and Y override but keeps known series', () => {
    const slug = 'group-reset';
    noteGroupSeries(slug, { s1: {} });
    toggleGroupSeries(slug, 's1');
    setGroupY(slug, 'log');
    resetGroup(slug);
    const snap = getGroupSnapshot(slug);
    expect(snap.hiddenSeries).toEqual([]);
    expect(snap.groupY).toBeNull();
    expect(Object.keys(snap.knownSeries)).toEqual(['s1']);
  });
});

describe('bounded priority queue', () => {
  it('runs at most the concurrency cap at once and drains by priority', async () => {
    // The hydration queue caps at 4 concurrent tasks. Fill the running slots
    // with 4 gate-blocked tasks, then enqueue three more with distinct
    // priorities and assert they run in priority order once the gates open.
    let release!: () => void;
    const gate = new Promise<void>((res) => {
      release = res;
    });
    let running = 0;
    let maxRunning = 0;
    const blocker = async (): Promise<void> => {
      running += 1;
      maxRunning = Math.max(maxRunning, running);
      await gate;
      running -= 1;
    };
    const blockers = Array.from({ length: 4 }, () => hydrationQueue.schedule(blocker, 0));

    const order: string[] = [];
    const queued = [
      hydrationQueue.schedule(async () => {
        order.push('low');
      }, 1),
      hydrationQueue.schedule(async () => {
        order.push('high');
      }, 100),
      hydrationQueue.schedule(async () => {
        order.push('mid');
      }, 50),
    ];
    // Nothing beyond the cap starts while the gates are closed.
    await Promise.resolve();
    expect(maxRunning).toBe(4);
    expect(order).toEqual([]);

    release();
    await Promise.all([...blockers.map((e) => e.promise), ...queued.map((e) => e.promise)]);
    expect(order).toEqual(['high', 'mid', 'low']);
  });

  it('rejects the entry promise when the task throws and keeps draining', async () => {
    const failing = hydrationQueue.schedule(async () => {
      throw new Error('boom');
    }, 0);
    await expect(failing.promise).rejects.toThrow('boom');
    const ok = hydrationQueue.schedule(async () => 'fine', 0);
    await expect(ok.promise).resolves.toBe('fine');
  });
});
