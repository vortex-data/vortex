// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Lazy, register-once Chart.js loader. The chart islands call [`loadChartJs`]
 * right before constructing a chart, so Chart.js and the zoom plugin stay out
 * of the initial bundle and only download when a chart actually hydrates,
 * mirroring v3's deferred `<script>` tags.
 *
 * The dynamic import also keeps `chartjs-plugin-zoom` (whose hammerjs
 * dependency expects a browser global) off the server-side render path of the
 * `'use client'` components that import this module.
 */

import type { Chart } from 'chart.js';

/** The two dynamic imports the loader performs, injectable for tests. */
export type ChartJsImporter = () => Promise<
  [typeof import('chart.js'), typeof import('chartjs-plugin-zoom')]
>;

const defaultImporter: ChartJsImporter = () =>
  Promise.all([import('chart.js'), import('chartjs-plugin-zoom')]);

/**
 * Build a register-once loader over `importer`. A SUCCESSFUL load is cached for
 * the lifetime of the tab; a FAILED load resets the cache before rejecting, so
 * the next chart interaction retries the import instead of replaying the same
 * cached rejection forever (a one-off chunk-load failure, e.g. hashed assets
 * rotated by a deploy mid-session, would otherwise leave every chart
 * unconstructable until a full page reload).
 */
export function createChartJsLoader(importer: ChartJsImporter): () => Promise<typeof Chart> {
  let loader: Promise<typeof Chart> | null = null;
  return () => {
    if (loader === null) {
      loader = importer().then(([chartModule, zoomModule]) => {
        chartModule.Chart.register(...chartModule.registerables, zoomModule.default);
        return chartModule.Chart;
      });
      loader.catch(() => {
        loader = null;
      });
    }
    return loader;
  };
}

/** Load Chart.js plus the zoom plugin, registering everything exactly once. */
export const loadChartJs = createChartJsLoader(defaultImporter);
