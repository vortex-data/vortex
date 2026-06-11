// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it, vi } from 'vitest';

import { createChartJsLoader, type ChartJsImporter } from './chart-js';

// Minimal stand-ins for the two imported modules; the loader only touches
// `Chart.register`, `registerables`, and the zoom plugin's default export.
function fakeModules() {
  const register = vi.fn();
  const chartModule = { Chart: { register }, registerables: ['a', 'b'] };
  const zoomModule = { default: { id: 'zoom' } };
  return { register, modules: [chartModule, zoomModule] as const };
}

describe('createChartJsLoader', () => {
  it('loads and registers exactly once across repeated calls', async () => {
    const { register, modules } = fakeModules();
    const importer = vi.fn(async () => modules);
    const load = createChartJsLoader(importer as unknown as ChartJsImporter);
    const first = await load();
    const second = await load();
    expect(first).toBe(second);
    expect(importer).toHaveBeenCalledTimes(1);
    expect(register).toHaveBeenCalledTimes(1);
    expect(register).toHaveBeenCalledWith('a', 'b', { id: 'zoom' });
  });

  it('retries after a failed import instead of caching the rejection', async () => {
    const { modules } = fakeModules();
    const importer = vi
      .fn()
      .mockRejectedValueOnce(new Error('chunk load failed'))
      .mockResolvedValue(modules);
    const load = createChartJsLoader(importer as unknown as ChartJsImporter);
    await expect(load()).rejects.toThrow('chunk load failed');
    // The failed load reset the cache; the next call re-imports and succeeds.
    await expect(load()).resolves.toBeDefined();
    expect(importer).toHaveBeenCalledTimes(2);
  });
});
