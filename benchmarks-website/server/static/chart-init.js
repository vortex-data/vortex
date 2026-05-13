// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Hydrate Chart.js charts on /, /chart/:slug, and /group/:slug, plus the
// lazy-fetch-on-toggle behaviour for closed `<details>` groups.
//
// File map (in source order):
//   1. Constants                       — throttle delays, fetch knobs, caps.
//   2. Canvas state contract           — every `canvas.__bench_*` field.
//   3. Per-card DOM contract           — every `data-role` selector.
//   4. Global filter state             — engines/formats from the navbar.
//   5. Palette + helpers               — colours, formatting, throttle.
//   6. Display unit picker             — bytes/time/count formatter switch.
//   7. LTTB                            — pure largest-triangle downsampler.
//   8. Crosshair plugin                — inline Chart.js plugin.
//   9. External tooltip handler        — factory that returns a Chart.js
//                                        external tooltip handler.
//  10. Payload + datasets              — readInlinePayload, buildDatasets,
//                                        rebuildVisibleAndUpdate.
//  11. Full-history warmup             — ensureFullHistory,
//                                        replaceChartPayload, plus the
//                                        slider + downsample-badge sync
//                                        helpers.
//  12. Per-card construction           — constructChart.
//  13. Range scrollbar strip           — bindRangeStrip + pointer math.
//  14. Per-chart toolbar wiring        — bindToolbar, attachWheelPan,
//                                        applyScope, applyY.
//  15. Group hydration                 — bounded shard fetch queue + UI
//                                        helpers.
//  16. Global filter wiring            — chip toggle, URL sync, bindings.
//  17. Per-group toolbar wiring        — group-level filter + Y override.
//  18. Header controls                 — theme toggle, expand/collapse all.
//  19. Page wiring                     — IntersectionObserver, init.
//
// Per-chart UX (for orientation):
//   - Each `.chart-card` carries `data-chart-slug`. The card *owns* its own
//     toolbar (`.toolbar--card`) — there is no page-level toolbar.
//   - Landing groups fetch materialized latest-100 group shards, with bounded
//     concurrency. Opening a group then queues `?n=all` for that group's charts
//     so the first paint is fast but full history is already warming.
//   - `rebuildVisibleAndUpdate` is the single source of truth for the
//     rendered point count. The cap is one constant: at most
//     `MAX_VISIBLE_POINTS` *unique commit indices* (x-positions) are
//     rendered, **shared across every series**. Below the cap we render
//     every commit that has data; above it we LTTB the per-commit
//     "max-y across series" to pick that many representatives, then
//     every series renders at those shared indices. This is what the
//     cap is *supposed* to mean: visually, the chart never has more
//     than that many x-axis columns regardless of how many lines are
//     on it. (Earlier per-series LTTB picked different peaks for each
//     series and the union of x-positions blew past the cap.)
//   - The slider is throttled to ~16ms (one frame at 60fps) per v2's
//     `CONFIG.ZOOM_THROTTLE_DELAY` so dragging the slider feels continuous.
//   - Mouse wheel pans horizontally (chartjs-plugin-zoom does not expose
//     pan-on-wheel, so a manual `wheel` listener calls `chart.pan(...)`).
//   - Drag-pan + drag-rectangle-zoom are wired through the plugin and
//     trigger the same `rebuildVisibleAndUpdate` via `onPan`/`onZoom`.
//   - A custom inline plugin draws a vertical crosshair at the hovered
//     commit; the external tooltip is offset and `pointer-events: none`
//     to fix the flicker described in the per-chart UX rebuild brief.
//
// Canvas state contract — every per-chart property we plant on the canvas:
//   canvas.__bench_chart              Chart.js instance, set in constructChart.
//   canvas.__bench_payload            Last-fetched ChartResponse (raw,
//                                     unmodified by LTTB). Source of truth
//                                     the tooltip + LTTB rebuild read.
//   canvas.__bench_state              { y: "linear"|"log", scope: number|"all" }
//                                     — the per-chart toolbar state.
//   canvas.__bench_overrides          Map<seriesLabel, true> of series the
//                                     user has manually toggled on this card.
//                                     Once set, the global filter no longer
//                                     drives that label's visibility here.
//   canvas.__bench_strip_render       Function bound by bindRangeStrip; called
//                                     from any path that mutates scales.x.
//   canvas.__bench_rebuild            Throttled `rebuildVisibleAndUpdate`
//                                     wrapper; called from pan/zoom/wheel.
//   canvas.__bench_wheel_attached     true once attachWheelPan has wired
//                                     a wheel listener (idempotency).
//   canvas.__bench_inline_trimmed     true if the current payload is a
//                                     virtual latest-100 view over a longer
//                                     full-history x-axis.
//   canvas.__bench_full_loaded        true once a `?n=all` refetch has
//                                     replaced the payload.
//   canvas.__bench_full_fetch_pending true while a `?n=all` refetch is in
//                                     flight; dedupes queue/pan promotion.
//   canvas.__bench_full_fetch_entry   Full-history queue entry, if the fetch
//                                     is queued but not yet complete.
//   canvas.__bench_payload_window     The server-side commit window used for
//                                     the current payload (`"100"` for
//                                     shard hydration, `"all"` for warmed
//                                     full history).
//   canvas.__bench_display_unit       The picked display unit (`format`,
//                                     `axisLabel`, `multiplier`) used by the
//                                     tooltip and y-axis label. Recomputed
//                                     after every payload swap and after each
//                                     LTTB rebuild changes the visible window.
//   canvas.__bench_y_user_set         true once the user has explicitly
//                                     clicked the per-chart Y-axis toolbar.
//                                     The per-group Y override skips charts
//                                     where this flag is set so the local
//                                     click stays sticky.
//
// Per-card DOM contract — every selector the chart cards are queried by:
//   .chart-card[data-chart-index][data-chart-slug]    The card itself.
//   canvas[data-chart-index]                          The chart canvas.
//   .chart-tooltip-host                               External tooltip host.
//   .chart-wrap                                       Canvas wrapper.
//   [data-role="downsample-badge"]                    LTTB badge slot.
//   [data-role="scope-slider"]                        Toolbar scope slider.
//   .toolbar--card                                    Toolbar root.
//   .toolbar-btn[data-y]                              Y-axis switch buttons.
//   [data-role="range-strip"]                         Range scrollbar root.
//   [data-role="range-window"]                        Range strip's window.
//   [data-role="range-handle-left"]                   Left resize handle.
//   [data-role="range-handle-right"]                  Right resize handle.
//   .group-disclosure                                 The <details> wrapper.
//   .group-details                                    The wrapping <section>.
//   [data-role="global-filter-bar"]                   Filter dropdown root.
//   [data-role="filter-trigger"]                      Filter dropdown button.
//   [data-role="filter-panel"]                        Filter dropdown body.
//   .filter-chip[data-filter][data-value]             A single filter chip.
//   [data-role="filter-badge"]                        Badge on the trigger.
//   [data-action="expand-all"]                        Header button.
//   [data-action="collapse-all"]                      Header button.
//   [data-role="theme-toggle"]                        Header button.
//   #bench-filter-state                               Server-emitted filter
//                                                     state JSON (script id).
(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // Constants
  // -----------------------------------------------------------------------
  var ZOOM_THROTTLE_MS = 16;     // one frame at ~60fps for slider drag
  var PAN_THROTTLE_MS = 50;      // pan/zoom throttle — looser than slider
  var FETCH_N = "all";           // explicit full-history upgrade
  var DEFAULT_VISIBLE = 100;     // initial visible window (last 100 of fetched)
  var CHART_FETCH_N = String(DEFAULT_VISIBLE); // materialized shard window
  var HYDRATION_CONCURRENCY = 4; // per-tab cap for latest-100 shard requests
  var FULL_HISTORY_CONCURRENCY = 2; // per-tab cap for background `?n=all`
  var GROUP_OPEN_PRIORITY_STEP = 100;
  var INTERACTION_FULL_PRIORITY = 1000000;

  // Resolve the default scope for a chart card. Group-open hydration always
  // starts with the latest-100 visible window, even when the x-axis is virtual
  // and spans the full chart history.
  function defaultScopeForCard(_card) {
    return DEFAULT_VISIBLE;
  }
  // Mirror of the server's default latest-window size. Latest-100 payloads now
  // include `history` metadata, so this constant is only a fallback for older
  // payloads and comments/tests that reason about the default window.
  var LANDING_INLINE_N = 100;
  // Hard cap on how many points a single series can render at once. When
  // the visible commit range has more raw non-null points than this, we
  // LTTB-downsample to exactly this number; below it we render raw. So
  // the user always sees at most this many points per series, regardless
  // of how far they zoom out, and the rule is one sentence:
  //
  //   visible <= MAX_VISIBLE_POINTS  → raw
  //   visible >  MAX_VISIBLE_POINTS  → LTTB to MAX_VISIBLE_POINTS
  //
  // Chart cards are ~600–900px on desktop and Chart.js draws ~2px point
  // markers, so 500 points gives roughly 1.5px of horizontal space per
  // point — about as dense as the eye can resolve. Bumping higher costs
  // render time without visible improvement; lowering loses detail on
  // wide cards.
  var MAX_VISIBLE_POINTS = 500;

  // -----------------------------------------------------------------------
  // Global filter state (engine/format chips inside the navbar dropdown).
  //
  // Model:
  //   `globalFilter.engines` / `.formats` track the *active* (visible) set
  //   for that dimension. The chip's displayed active state mirrors
  //   visibility — every chip active means no filter is applied, exactly
  //   one chip inactive hides only that engine/format, and so on. The
  //   URL `?engine=`/`?format=` stay as allowlists for stability across
  //   refreshes; we omit the param when every chip is active (i.e. the
  //   active set equals the universe), so the no-filter URL is clean.
  //
  // Per-card overrides:
  //   Clicking a chart's legend toggles `dataset.hidden` and adds the label
  //   to that card's `canvas.__bench_overrides` set. The global apply pass
  //   skips overridden labels, so the user's manual call sticks even after
  //   subsequent global filter changes.
  // -----------------------------------------------------------------------
  var globalFilter = readFilterState();
  var filterUniverse = readFilterUniverseFromDom();
  // `seedFromUrl` translates the URL state (allowlist) into the active set.
  // Empty allowlist in the URL is treated as "no filter" → every chip
  // active. Non-empty is taken verbatim, even if a chip has since been
  // added or removed from the universe — keeps stale URLs deterministic.
  seedActiveFromUrlState();

  function readFilterState() {
    var fallback = { engines: [], formats: [] };
    var node = document.getElementById("bench-filter-state");
    if (!node) return fallback;
    try {
      var parsed = JSON.parse(node.textContent);
      return {
        engines: Array.isArray(parsed.engines) ? parsed.engines.slice() : [],
        formats: Array.isArray(parsed.formats) ? parsed.formats.slice() : [],
      };
    } catch (e) {
      return fallback;
    }
  }

  // Pull the chip universe straight from the rendered panel, so the JS
  // doesn't have to mirror the server's enum. If the dropdown isn't on the
  // page (shouldn't happen — the header always renders it when there's
  // data) we fall back to whatever is in the URL state.
  function readFilterUniverseFromDom() {
    var u = { engines: [], formats: [] };
    document.querySelectorAll(
      '[data-role="filter-panel"] .filter-chip[data-value]:not([data-value="*"])',
    ).forEach(function (chip) {
      var dim = chip.getAttribute("data-filter");
      var value = chip.getAttribute("data-value");
      if (!dim || !value) return;
      var bucket = dim === "engine" ? u.engines : u.formats;
      if (bucket.indexOf(value) === -1) bucket.push(value);
    });
    return u;
  }

  function seedActiveFromUrlState() {
    if (!globalFilter.engines.length) {
      globalFilter.engines = filterUniverse.engines.slice();
    }
    if (!globalFilter.formats.length) {
      globalFilter.formats = filterUniverse.formats.slice();
    }
  }

  // Any-of-universe-missing-from-active means the dimension is filtered.
  function dimensionIsFiltered(key) {
    return globalFilter[key].length < filterUniverse[key].length;
  }

  // A series is hidden when its engine/format dimension is filtered AND its
  // tag isn't in the active set. Series without an engine tag (e.g.
  // compression-time `format:op` series) are unaffected by the engine
  // filter — symmetric for format. This keeps the chip semantics intuitive:
  // hiding an engine doesn't nuke charts that have no engine dimension.
  function seriesPassesFilter(meta) {
    if (!meta) meta = {};
    if (meta.engine && dimensionIsFiltered("engines")
        && globalFilter.engines.indexOf(meta.engine) === -1) {
      return false;
    }
    if (meta.format && dimensionIsFiltered("formats")
        && globalFilter.formats.indexOf(meta.format) === -1) {
      return false;
    }
    return true;
  }

  // Per-group filter layer. State is a single `hiddenSeries` array of dataset
  // labels the user has toggled off via the group's filter dropdown. Engine
  // and format chips in the dropdown are macros: clicking them bulk-toggles
  // every known series whose `engine`/`format` matches (see
  // `applyMacroToHiddenSeries`). The series list itself populates as charts
  // in the group hydrate and surface their `payload.series_meta`.
  function seriesPassesGroupFilter(filter, label) {
    if (!filter || !filter.hiddenSeries) return true;
    return filter.hiddenSeries.indexOf(label) === -1;
  }

  // -----------------------------------------------------------------------
  // Palette + helpers
  // -----------------------------------------------------------------------
  var palette = [
    "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed",
    "#0891b2", "#ca8a04", "#db2777", "#65a30d", "#475569",
  ];

  function colorFor(i) { return palette[i % palette.length]; }

  function shortSha(sha) {
    return typeof sha === "string" ? sha.slice(0, 7) : String(sha);
  }

  function shortDate(ts) {
    if (typeof ts !== "string") return "";
    return ts.slice(0, 10);
  }

  function truncate(s, max) {
    if (typeof s !== "string") return "";
    return s.length > max ? s.slice(0, max - 1) + "…" : s;
  }

  function firstLine(s) {
    if (typeof s !== "string") return "";
    var nl = s.indexOf("\n");
    return nl >= 0 ? s.slice(0, nl) : s;
  }

  // Vortex commits to `develop` are squash-merged from PRs; the squash subject
  // ends with `(#NNNN)`. Returning just the number lets callers build either a
  // PR or commit URL.
  function parsePrNumber(message) {
    if (typeof message !== "string") return null;
    var m = message.match(/\(#(\d+)\)/);
    return m ? m[1] : null;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // -----------------------------------------------------------------------
  // Display unit picker. The wire payload's `unit_kind` says *what* the
  // values are (`time_ns`, `bytes`, …); this helper turns that plus the
  // magnitude of the loaded values into a `(multiplier, suffix, axisLabel,
  // decimals)` tuple. The chart locks that tuple on construction (and again
  // after the lazy `?n=all` refetch swaps the payload) so the y-axis stays
  // stable while the user pans/zooms — recomputing per-frame would shift
  // the unit out from under them.
  //
  // Worked example: a `time_ns` series whose median is 12,000,000,000 ns
  // picks `{ multiplier: 1e-9, suffix: "s", axisLabel: "Time (s)" }`, so
  // `12,000,000,000` renders as `12 s` on the axis and in the tooltip.
  // -----------------------------------------------------------------------
  var IDENTITY_UNIT = {
    multiplier: 1,
    suffix: "",
    axisLabel: "",
    decimals: 2,
  };

  // Median of finite, nonzero |v|. Zeros and NaNs aren't informative for the
  // magnitude pick (a chart with all zeros isn't readable anyway), so we
  // skip them; if every value is filtered out, return `null` and callers
  // fall back to the kind's smallest display unit.
  function magnitudeReference(values) {
    if (!Array.isArray(values) || values.length === 0) return null;
    var sample = [];
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      if (v === null || v === undefined) continue;
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      var a = Math.abs(v);
      if (a === 0) continue;
      sample.push(a);
    }
    if (sample.length === 0) return null;
    sample.sort(function (a, b) { return a - b; });
    var mid = Math.floor(sample.length / 2);
    return (sample.length % 2)
      ? sample[mid]
      : (sample[mid - 1] + sample[mid]) / 2;
  }

  // Walk every series in the loaded payload and concatenate the non-null
  // values. The picker works off the merged distribution so a chart with one
  // very fast and one very slow series still picks the unit that keeps the
  // larger magnitudes readable. Toggling a series visibility via the global
  // filter does NOT call this — the unit is locked at payload-load time.
  function collectAllValues(payload) {
    var out = [];
    var series = (payload && payload.series) || {};
    var keys = Object.keys(series);
    for (var i = 0; i < keys.length; i++) {
      var arr = series[keys[i]];
      if (!Array.isArray(arr)) continue;
      for (var j = 0; j < arr.length; j++) {
        var v = arr[j];
        if (v !== null && v !== undefined && Number.isFinite(v)) out.push(v);
      }
    }
    return out;
  }

  function pickTimeUnit(ref) {
    // Steps: ns → µs (1e3) → ms (1e6) → s (1e9). Pick by the median's
    // magnitude so the y-axis tick numbers fit in 1–4 digits.
    if (ref === null || ref < 1e3) {
      return { multiplier: 1, suffix: "ns", decimals: 0 };
    }
    if (ref < 1e6) return { multiplier: 1e-3, suffix: "µs", decimals: 2 };
    if (ref < 1e9) return { multiplier: 1e-6, suffix: "ms", decimals: 2 };
    return { multiplier: 1e-9, suffix: "s", decimals: 2 };
  }

  function pickBytesUnit(ref) {
    // Binary multiples to match how DuckDB and on-disk file sizes are
    // typically reported. Steps: B → KiB (1024) → MiB → GiB → TiB.
    var k = 1024;
    if (ref === null || ref < k) {
      return { multiplier: 1, suffix: "B", decimals: 0 };
    }
    if (ref < k * k) return { multiplier: 1 / k, suffix: "KiB", decimals: 2 };
    if (ref < k * k * k) return { multiplier: 1 / (k * k), suffix: "MiB", decimals: 2 };
    if (ref < k * k * k * k) return { multiplier: 1 / (k * k * k), suffix: "GiB", decimals: 2 };
    return { multiplier: 1 / (k * k * k * k), suffix: "TiB", decimals: 2 };
  }

  function pickDisplayUnit(unitKind, values) {
    var ref = magnitudeReference(values);
    if (unitKind === "time_ns") {
      var t = pickTimeUnit(ref);
      return {
        multiplier: t.multiplier,
        suffix: t.suffix,
        axisLabel: "Time (" + t.suffix + ")",
        decimals: t.decimals,
      };
    }
    if (unitKind === "bytes") {
      var b = pickBytesUnit(ref);
      return {
        multiplier: b.multiplier,
        suffix: b.suffix,
        axisLabel: "Size (" + b.suffix + ")",
        decimals: b.decimals,
      };
    }
    if (unitKind === "throughput_mb_s") {
      return {
        multiplier: 1,
        suffix: "MB/s",
        axisLabel: "Throughput (MB/s)",
        decimals: 2,
      };
    }
    if (unitKind === "ratio" || unitKind === "count") {
      // Dimensionless: no scaling, no suffix, no axis title — leaving the
      // axis unlabeled keeps a "1.2× speedup" axis from being read as
      // "1200 m" by an axis-title-driven label.
      return {
        multiplier: 1,
        suffix: "",
        axisLabel: "",
        decimals: unitKind === "count" ? 0 : 2,
      };
    }
    // Unknown kind (forward-compat with a future server enum). Identity is
    // the safest fallback — values render verbatim, no unit.
    return IDENTITY_UNIT;
  }

  // Tooltip formatter: applies the chart's locked display unit so the tooltip
  // value matches the y-axis tick numbers exactly. Raw `null`/`NaN` collapse
  // to an em-dash so a missing data point reads as a clear gap rather than
  // a literal `0`.
  function formatDisplayValue(rawValue, displayUnit) {
    if (rawValue === null || rawValue === undefined || Number.isNaN(rawValue)) {
      return "—";
    }
    var u = displayUnit || IDENTITY_UNIT;
    var scaled = rawValue * u.multiplier;
    var text = Number.isFinite(scaled) ? scaled.toFixed(u.decimals) : "—";
    return u.suffix ? text + " " + u.suffix : text;
  }

  // Throttle to a max call rate; trailing call is preserved so the final
  // slider position is honoured. (`requestAnimationFrame` is conceptually
  // similar but we want a hard ceiling regardless of when the browser
  // schedules a frame.)
  function throttle(fn, ms) {
    var lastRan = 0;
    var pending = null;
    var pendingArgs = null;
    return function () {
      var now = Date.now();
      pendingArgs = arguments;
      if (now - lastRan >= ms) {
        lastRan = now;
        fn.apply(null, pendingArgs);
      } else if (!pending) {
        var wait = ms - (now - lastRan);
        pending = setTimeout(function () {
          lastRan = Date.now();
          pending = null;
          fn.apply(null, pendingArgs);
        }, wait);
      }
    };
  }

  // -----------------------------------------------------------------------
  // LTTB (Largest-Triangle-Three-Buckets) downsampler.
  //
  // Returns the indices into `xs` / `ys` to keep, including index 0 and
  // `n - 1`. `xs` must be strictly increasing. When `threshold >= n` or
  // `threshold < 3`, returns `[0, 1, ..., n-1]` unchanged.
  //
  // Algorithm: <https://skemman.is/handle/1946/15343>. Per-bucket pick the
  // point that forms the largest triangle with the previously kept point
  // and the average of the next bucket.
  // -----------------------------------------------------------------------
  function lttbIndices(xs, ys, threshold) {
    var n = xs.length;
    if (threshold >= n || threshold < 3) {
      var all = new Array(n);
      for (var i = 0; i < n; i++) all[i] = i;
      return all;
    }
    var out = new Array(threshold);
    out[0] = 0;
    var bucket = (n - 2) / (threshold - 2);
    var a = 0;
    for (var bi = 0; bi < threshold - 2; bi++) {
      // Average of the *next* bucket — the "C" point in the triangle.
      var nextStart = Math.floor((bi + 1) * bucket) + 1;
      var nextEnd = Math.min(n, Math.floor((bi + 2) * bucket) + 1);
      var count = Math.max(1, nextEnd - nextStart);
      var ax = 0, ay = 0;
      for (var j = nextStart; j < nextEnd; j++) { ax += xs[j]; ay += ys[j]; }
      ax /= count; ay /= count;

      // Search this bucket for the point with the largest triangle area
      // against (a, avg_next).
      var rangeStart = Math.floor(bi * bucket) + 1;
      var rangeEnd = Math.floor((bi + 1) * bucket) + 1;
      var pax = xs[a], pay = ys[a];
      var maxArea = -1;
      var maxIdx = rangeStart;
      for (var k = rangeStart; k < rangeEnd; k++) {
        var area = Math.abs((pax - ax) * (ys[k] - pay) - (pax - xs[k]) * (ay - pay)) * 0.5;
        if (area > maxArea) { maxArea = area; maxIdx = k; }
      }
      out[bi + 1] = maxIdx;
      a = maxIdx;
    }
    out[threshold - 1] = n - 1;
    return out;
  }

  // -----------------------------------------------------------------------
  // Crosshair plugin: draws a vertical line at the chart's active hover
  // index. Using an inline plugin is cheaper than pulling in
  // chartjs-plugin-crosshair, which is overkill for this one feature.
  // -----------------------------------------------------------------------
  var crosshairPlugin = {
    id: "benchCrosshair",
    afterDatasetsDraw: function (chart) {
      var active = chart.tooltip && chart.tooltip.getActiveElements
        ? chart.tooltip.getActiveElements()
        : [];
      if (!active || !active.length) return;
      var x = active[0].element.x;
      var ya = chart.scales && chart.scales.y;
      if (!ya || !Number.isFinite(x)) return;
      var ctx = chart.ctx;
      ctx.save();
      // `--muted` from the page theme — read it lazily so dark mode picks
      // up the right colour.
      var muted = getComputedStyle(document.documentElement)
        .getPropertyValue("--muted").trim() || "#9ca3af";
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, ya.top);
      ctx.lineTo(x, ya.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  // -----------------------------------------------------------------------
  // External tooltip with offset + flip-on-overflow.
  //
  // Factory contract: returns a Chart.js `external` tooltip handler closed
  // over `canvas` (the rendered canvas element, used to read the cached
  // payload via `canvas.__bench_payload`) and `host` (the
  // `<div class="chart-tooltip-host">` element to render markup into;
  // `host.parentNode` is the chart-card and is used as the positioning
  // origin). The returned handler is invoked by Chart.js with one argument
  // `context = { tooltip, chart }`; it mutates `host` in place and is a
  // no-op when `tooltip.opacity === 0`.
  //
  // Flicker fix: the tooltip host is **always** `pointer-events: none`. The
  // previous implementation flipped it to `auto` when visible; the cursor
  // would land on the tooltip, fire mouseout on the canvas, the tooltip
  // would hide, the cursor would re-enter the canvas, and the cycle would
  // repeat at event-loop frequency. Clicks on a data point are handled by
  // the chart's `onClick` (opens the PR or commit URL in a new tab), so the
  // tooltip itself never needs to be interactive.
  // -----------------------------------------------------------------------
  function externalTooltipHandler(canvas, host) {
    return function (context) {
      var tt = context.tooltip;
      if (!host) return;
      if (tt.opacity === 0) {
        host.style.opacity = "0";
        return;
      }

      var chart = context.chart;
      var payload = canvas.__bench_payload || { commits: [] };
      var firstDp = tt.dataPoints && tt.dataPoints[0];
      if (!firstDp) {
        host.style.opacity = "0";
        return;
      }
      // Snap to a single commit. We use `mode: "nearest"` on the chart
      // options, so `firstDp.dataIndex` is the single closest data point
      // to the cursor (skipping nulls in `dataset.data`). If the cursor
      // falls between two LTTB-kept points, exactly one wins — no more
      // rendering both columns at once.
      var idx = firstDp.dataIndex;
      var commit = (payload.commits || [])[idx] || {};
      // Tooltip values must match the locked y-axis unit. Raw values still
      // live on `dataset.rawData`; the display unit is what scales them
      // into ms / MiB / etc. for the visible text.
      var displayUnit = canvas.__bench_display_unit || IDENTITY_UNIT;
      var rawLen = (chart.data.labels || []).length;

      // Build one row per dataset, reading values from each series'
      // `rawData` (the unmodified payload) so the tooltip shows raw
      // measurements even when LTTB has nulled out `dataset.data[idx]`.
      // Iterating `chart.data.datasets` directly — instead of mapping
      // `tt.dataPoints` — guarantees one row per series at this single
      // commit; `tt.dataPoints` could otherwise contain points from
      // multiple `dataIndex` values when the cursor sits between two
      // closely-packed LTTB columns.
      var rowItems = chart.data.datasets.map(function (ds, dsIndex) {
        // Skip datasets the user (or filter bar) has hidden.
        var meta = chart.getDatasetMeta && chart.getDatasetMeta(dsIndex);
        if (meta && meta.hidden) return null;
        if (ds.hidden) return null;
        var raw = (ds.rawData || [])[idx];
        if (raw === null || raw === undefined || Number.isNaN(raw)) {
          return null;
        }
        // Per-row delta is `(current - previous) / previous`, where
        // "previous" is the chronologically preceding commit. The
        // `commits[]` array is sorted oldest-first by SQL — index 0 is
        // the oldest commit, the last index is the newest — so the
        // predecessor lives at `idx - 1`. Walk further back across
        // null-valued slots so series that didn't run on every commit
        // still get a meaningful baseline.
        var prevIdx = idx - 1;
        var prevRaw = null;
        while (prevIdx >= 0) {
          var pv = (ds.rawData || [])[prevIdx];
          if (pv !== null && pv !== undefined && !Number.isNaN(pv)) {
            prevRaw = pv;
            break;
          }
          prevIdx--;
        }
        var deltaHtml = "";
        if (prevRaw !== null && prevRaw !== 0) {
          var pct = ((raw - prevRaw) / prevRaw) * 100;
          var cls = pct > 0 ? "tt-delta tt-delta--worse"
                  : pct < 0 ? "tt-delta tt-delta--better" : "tt-delta";
          var sign = pct > 0 ? "+" : "";
          deltaHtml = '<span class="' + cls + '">' + sign + pct.toFixed(1) + "%</span>";
        }
        return {
          label: ds.label,
          color: ds.borderColor,
          raw: raw,
          deltaHtml: deltaHtml,
        };
      }).filter(Boolean);

      // Top-to-bottom order matches the visual stack of lines at this x.
      rowItems.sort(function (a, b) { return b.raw - a.raw; });

      var rows = rowItems
        .map(function (r) {
          return '<div class="tt-row">'
            + '<span class="tt-swatch" style="background:' + r.color + '"></span>'
            + '<span class="tt-label">' + escapeHtml(r.label) + '</span>'
            + '<span class="tt-value">'
            + escapeHtml(formatDisplayValue(r.raw, displayUnit)) + '</span>'
            + r.deltaHtml
            + "</div>";
        })
        .join("");

      // If every series was hidden / had no value at this commit, treat
      // this as a no-op hover instead of flashing an empty popup.
      if (!rows) {
        host.style.opacity = "0";
        return;
      }

      var titleHtml = '<div class="tt-title">'
        + escapeHtml(shortSha(commit.sha)) + ' · '
        + escapeHtml(shortDate(commit.timestamp))
        + "</div>";

      // Show short SHA + first-line commit message, truncated. The full URL
      // (or PR URL) is wired up via the chart's onClick handler, so we don't
      // render it as text here.
      var msg = truncate(firstLine(commit.message || ""), 80);
      var footerLine = commit.sha
        ? (msg ? escapeHtml(shortSha(commit.sha)) + " · " + escapeHtml(msg)
                : escapeHtml(shortSha(commit.sha)))
        : escapeHtml(msg);
      var footerHtml = footerLine
        ? '<div class="tt-footer"><div class="tt-msg">' + footerLine + "</div></div>"
        : "";

      host.innerHTML = titleHtml + '<div class="tt-rows">' + rows + "</div>" + footerHtml;

      // Position the tooltip relative to its container, offset 12px from
      // the cursor. Flip horizontally if it would overflow.
      var canvasRect = context.chart.canvas.getBoundingClientRect();
      var hostRect = host.parentNode.getBoundingClientRect();
      var x = canvasRect.left - hostRect.left + tt.caretX;
      var y = canvasRect.top - hostRect.top + tt.caretY;
      host.style.opacity = "1";
      host.style.left = x + "px";
      host.style.top = y + "px";
      // Measure after content swap so flipping is correct.
      var ttWidth = host.offsetWidth || 0;
      var containerWidth = host.parentNode.clientWidth || 0;
      var flip = (x + ttWidth + 24) > containerWidth;
      host.style.transform = flip
        ? "translate(calc(-100% - 12px), 12px)"
        : "translate(12px, 12px)";
    };
  }

  // -----------------------------------------------------------------------
  // Payload + datasets
  // -----------------------------------------------------------------------
  function readInlinePayload(idx) {
    var s = document.getElementById("chart-data-" + idx);
    if (!s) return null;
    try { return JSON.parse(s.textContent); } catch (e) { return null; }
  }

  function labelForCommit(commit) {
    return commit && commit.sha ? shortSha(commit.sha) : "";
  }

  function canonicalHistory(payload) {
    var commits = Array.isArray(payload && payload.commits) ? payload.commits : [];
    var history = payload && payload.history ? payload.history : {};
    var loaded = Number.isFinite(history.loaded_commits)
      ? history.loaded_commits
      : commits.length;
    var total = Number.isFinite(history.total_commits)
      ? history.total_commits
      : commits.length;
    var start = Number.isFinite(history.start_index) ? history.start_index : 0;
    loaded = Math.max(0, Math.floor(loaded));
    total = Math.max(loaded, Math.floor(total));
    start = Math.max(0, Math.min(Math.floor(start), Math.max(0, total - loaded)));
    return {
      total_commits: total,
      start_index: start,
      loaded_commits: loaded,
      complete: history.complete === true || (start === 0 && loaded === total),
    };
  }

  function normalizeChartPayload(payload) {
    if (!payload) return payload;
    if (payload.__bench_normalized) return payload;
    var commits = Array.isArray(payload.commits) ? payload.commits : [];
    var history = canonicalHistory(payload);
    if (history.complete && history.start_index === 0 && history.total_commits === commits.length) {
      payload.history = history;
      payload.__bench_normalized = true;
      return payload;
    }

    var total = history.total_commits;
    var start = history.start_index;
    var normalizedCommits = new Array(total);
    for (var i = 0; i < total; i++) normalizedCommits[i] = null;
    for (var ci = 0; ci < commits.length && start + ci < total; ci++) {
      normalizedCommits[start + ci] = commits[ci];
    }

    var rawSeries = payload.series || {};
    var normalizedSeries = {};
    Object.keys(rawSeries).forEach(function (name) {
      var values = Array.isArray(rawSeries[name]) ? rawSeries[name] : [];
      var out = new Array(total);
      for (var zi = 0; zi < total; zi++) out[zi] = null;
      for (var vi = 0; vi < values.length && start + vi < total; vi++) {
        out[start + vi] = values[vi];
      }
      normalizedSeries[name] = out;
    });

    var clone = {};
    Object.keys(payload).forEach(function (key) {
      if (key !== "commits" && key !== "series") clone[key] = payload[key];
    });
    clone.history = history;
    clone.commits = normalizedCommits;
    clone.series = normalizedSeries;
    clone.__bench_normalized = true;
    return clone;
  }

  function rangeTouchesUnloadedHistory(payload, min, max) {
    var history = payload && payload.history;
    if (!history || history.complete) return false;
    var start = history.start_index || 0;
    var end = start + (history.loaded_commits || 0) - 1;
    return Math.floor(min) < start || Math.ceil(max) > end;
  }

  // Build the per-series dataset shells. `data` starts as a full-length
  // null-padded array; `rebuildVisibleAndUpdate` fills it in based on the
  // current visible range. `rawData` holds a reference to the original
  // payload so the tooltip can show raw values regardless of LTTB.
  function buildDatasets(payload) {
    var raw = payload.series || {};
    var meta = payload.series_meta || {};
    var n = (payload.commits || []).length;
    var names = Object.keys(raw).sort();
    return names.map(function (name, i) {
      var seriesMeta = meta[name] || {};
      var rawValues = Array.isArray(raw[name]) ? raw[name] : [];
      // `data` starts null-padded; `rebuildVisibleAndUpdate` fills the
      // current visible window with raw or LTTB-kept values. Chart.js's
      // `spanGaps: true` connects the line across nulls so a series with
      // partial coverage (a benchmark crashed at one commit, a series
      // only runs nightly, etc.) still draws as a continuous trend
      // through the surrounding measurements. The point markers
      // themselves are only drawn at non-null indices, so the missing
      // commits are visible as a "no marker" beat in the line — the line
      // itself bridges to the next available data point.
      var data = new Array(n);
      for (var j = 0; j < n; j++) data[j] = null;
      return {
        label: name,
        data: data,
        rawData: rawValues,
        borderColor: colorFor(i),
        backgroundColor: colorFor(i) + "20",
        borderWidth: 1.5,
        spanGaps: true,
        tension: 0,
        pointRadius: 2,
        pointHoverRadius: 5,
        pointHitRadius: 8,
        pointStyle: "cross",
        // Custom field (Chart.js ignores unknown keys). Used by the global
        // filter to decide which datasets to hide/show in bulk.
        benchMeta: { engine: seriesMeta.engine, format: seriesMeta.format },
        hidden: !seriesPassesFilter(seriesMeta),
      };
    });
  }

  // -----------------------------------------------------------------------
  // The single source of truth for the rendered point count.
  //
  // Walks the visible `[rangeMin, rangeMax]` window of the raw payload and,
  // for each series, renders raw when the visible count is at or below
  // `MAX_VISIBLE_POINTS` and LTTB-downsamples to exactly that number when
  // above. The result is written into `dataset.data` with nulls outside
  // the kept set so Chart.js renders just the kept points; with
  // `spanGaps: true`, the line connects across the nulls to the next
  // non-null point so a sparse series still reads as a continuous trend.
  //
  // Mutates `dataset.data` in place to avoid GC churn on every pan frame.
  // Updates the per-card downsample badge as a side effect.
  // -----------------------------------------------------------------------
  function rebuildVisibleAndUpdate(card, chart, rangeMin, rangeMax, allowFullFetch) {
    var canvas = chart.canvas;
    var payload = canvas.__bench_payload;
    if (!payload) return;
    var datasets = chart.data.datasets;
    var n = (payload.commits || []).length;
    if (n === 0) return;

    var min = Math.max(0, Math.floor(rangeMin));
    var max = Math.min(n - 1, Math.ceil(rangeMax));
    if (max < min) max = min;

    // Build one "virtual series" for LTTB: walk every commit index in the
    // visible range and, for each index, take the max non-null value
    // across all datasets. This is the union of x-positions, with a
    // representative y per position. Series in a Vortex chart share both
    // unit and overall scale (they're the same benchmark with different
    // engines/formats), so max-across-series picks visually salient peaks
    // without per-series scale skew.
    //
    // This becomes our LTTB input: we then pick AT MOST MAX_VISIBLE_POINTS
    // commit indices and every dataset renders only at those shared
    // indices. Without this, per-series LTTB picked different peaks for
    // each series and the union of x-positions grew with the series
    // count — visually you saw way more than MAX_VISIBLE_POINTS dots
    // even though each line only had MAX_VISIBLE_POINTS.
    var unionIdxs = [];
    var unionVals = [];
    for (var i = min; i <= max; i++) {
      var bestY = null;
      for (var di = 0; di < datasets.length; di++) {
        var rawValues = datasets[di].rawData;
        if (!Array.isArray(rawValues)) continue;
        var v = rawValues[i];
        if (v !== null && v !== undefined && !Number.isNaN(v)
            && (bestY === null || v > bestY)) {
          bestY = v;
        }
      }
      if (bestY !== null) {
        unionIdxs.push(i);
        unionVals.push(bestY);
      }
    }

    // Decide which commit indices to render — shared across all series.
    var keptSet = {};
    var anyDownsampled = false;
    if (unionIdxs.length <= MAX_VISIBLE_POINTS) {
      // Below the cap: render every commit that has data anywhere.
      for (var u = 0; u < unionIdxs.length; u++) keptSet[unionIdxs[u]] = true;
    } else {
      // Above the cap: LTTB the union down to MAX_VISIBLE_POINTS exactly.
      // The selected indices are then *shared* across every dataset; that
      // is the cap's only correct interpretation of "max points on the
      // chart at a time".
      var localIndices = lttbIndices(unionIdxs, unionVals, MAX_VISIBLE_POINTS);
      for (var li = 0; li < localIndices.length; li++) {
        keptSet[unionIdxs[localIndices[li]]] = true;
      }
      anyDownsampled = true;
    }

    // Plant the shared kept set into every dataset.data. Series that have
    // no value at a kept index simply remain null there; with
    // `spanGaps: true`, the line connects to the next non-null point so
    // a series with partial coverage (a benchmark crashed, a series only
    // runs nightly) still draws as a continuous trend through the
    // surrounding measurements. Markers are only drawn at non-null
    // indices, so the gap is still visible as a missing point — just not
    // as a broken line.
    //
    // We deliberately do NOT plant nearest-neighbour values for indices
    // outside `[min, max]`: extending the line past the visible edges
    // sounds nice (the line goes off-screen toward the next real
    // measurement instead of stopping at the rightmost in-range point),
    // but Chart.js's y-axis auto-scale uses every non-null value in the
    // dataset regardless of `scales.x.min/max`. An off-screen neighbour
    // with a very different y value (an old benchmark configuration, a
    // first-run cold cache, anything) blows up the y-axis range and
    // squashes the in-window values into a flat line near the floor.
    // Fixing that would mean overriding `scales.y.min/max` per rebuild
    // from only the in-window values, which changes the "y-axis stays
    // stable across x-zoom" UX. Out of scope here; if a user wants to
    // see how the line connects across the edge they can zoom out.
    // Pull the chart's locked display-unit multiplier. Applied here, not on
    // ingest or in the SQL, so the wire payload stays in base units (ns,
    // bytes, …) — the unit transform is purely cosmetic.
    var displayUnit = canvas.__bench_display_unit || IDENTITY_UNIT;
    var multiplier = displayUnit.multiplier;
    for (var dj = 0; dj < datasets.length; dj++) {
      var ds = datasets[dj];
      var dsRaw = ds.rawData;
      if (!Array.isArray(dsRaw)) continue;
      var data = ds.data;
      if (!Array.isArray(data) || data.length !== n) {
        data = new Array(n);
        ds.data = data;
      }
      for (var z = 0; z < n; z++) data[z] = null;
      for (var idxStr in keptSet) {
        var idx = +idxStr;
        var val = dsRaw[idx];
        if (val !== null && val !== undefined && !Number.isNaN(val)) {
          data[idx] = val * multiplier;
        }
      }
    }

    var visibleCommits = max - min + 1;
    var keptCommits = 0;
    for (var _u in keptSet) keptCommits++;
    chart.update("none");
    syncSliderFromRange(card, visibleCommits);
    syncDownsampleBadge(card, keptCommits, visibleCommits, anyDownsampled);
    // If the user moves into the virtual, not-yet-loaded part of the x-axis,
    // promote this chart's queued full-history fetch. Group-open warmup should
    // usually have it in flight already; this just puts direct interaction
    // ahead of background work that has not started.
    if (allowFullFetch && rangeTouchesUnloadedHistory(payload, min, max)) {
      ensureFullHistory(card, INTERACTION_FULL_PRIORITY);
    }
  }

  // -----------------------------------------------------------------------
  // Full-history warmup. Opening a group queues `?n=all` for every chart in
  // that group. Direct interaction with an unloaded virtual range promotes the
  // same queued entry instead of issuing a duplicate request.
  // -----------------------------------------------------------------------
  function ensureFullHistory(card, priority) {
    var canvas = card.querySelector("canvas");
    if (!canvas) return Promise.resolve();
    if (canvas.__bench_full_loaded) return Promise.resolve();
    if (canvas.__bench_full_fetch_entry) {
      if (priority) {
        canvas.__bench_full_fetch_entry.priority = Math.max(
          canvas.__bench_full_fetch_entry.priority,
          priority
        );
        drainFullHistoryQueue();
      }
      return canvas.__bench_full_fetch_pending || Promise.resolve();
    }
    var slug = card.getAttribute("data-chart-slug");
    if (!slug) return Promise.resolve();
    var entry = scheduleFullHistory(function () {
      var url = "/api/chart/" + encodeURIComponent(slug)
        + "?n=" + encodeURIComponent(FETCH_N);
      return fetch(url, { headers: { "accept": "application/json" } })
        .then(function (r) {
          if (r.status === 404) return null;
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (full) {
          if (!full) return;
          replaceChartPayload(card, full);
          canvas.__bench_full_loaded = true;
          canvas.__bench_inline_trimmed = false;
          showCardLoading(card, false);
          var group = card.closest(".group-details");
          if (!canvas.__bench_chart && (!group || groupIsOpen(group))) {
            fetchAndConstruct(card);
          }
        });
    }, priority || 0);
    canvas.__bench_full_fetch_entry = entry;
    canvas.__bench_full_fetch_pending = entry.promise.catch(function (err) {
      // Quiet — the latest-100 virtual payload is still usable. Surface to the
      // console for debugging.
      if (window && window.console) {
        window.console.warn("bench: full history fetch failed", err);
      }
    }).then(function () {
      canvas.__bench_full_fetch_entry = null;
      canvas.__bench_full_fetch_pending = null;
    });
    return canvas.__bench_full_fetch_pending;
  }

  // Swap the chart's labels + datasets to a freshly fetched, unbounded
  // payload while preserving the current x-range. The virtual latest-100
  // payload and the full payload share a full-history x-axis, so the chart
  // should not jump when the real older values arrive.
  function replaceChartPayload(card, payload) {
    var canvas = card.querySelector("canvas");
    var chart = canvas && canvas.__bench_chart;
    if (!canvas || !payload) return;
    payload = normalizeChartPayload(payload);
    canvas.__bench_payload = payload;
    canvas.__bench_payload_window = FETCH_N;
    if (!chart) return;
    // Re-pick the display unit against the now-wider window. The first
    // payload was the inlined slice (`LANDING_INLINE_N` commits); the
    // refetch may surface older commits with a different magnitude, and
    // we'd rather move the y-axis once at the refetch boundary than leave
    // the chart on a stale unit. The axis title is updated to match.
    canvas.__bench_display_unit = pickDisplayUnit(
      payload.unit_kind, collectAllValues(payload),
    );
    var yAxis = chart.options.scales && chart.options.scales.y;
    if (yAxis && yAxis.title) {
      yAxis.title.display = !!canvas.__bench_display_unit.axisLabel;
      yAxis.title.text = canvas.__bench_display_unit.axisLabel;
    }
    var newLabels = (payload.commits || []).map(labelForCommit);
    var newDatasets = buildDatasets(payload);
    // Re-apply per-card legend overrides + global filter to the new datasets,
    // matching the visibility state the user had before the refetch.
    var overrides = canvas.__bench_overrides || {};
    for (var i = 0; i < newDatasets.length; i++) {
      var ds = newDatasets[i];
      if (overrides[ds.label]) {
        // Honour any explicit legend toggle the user had made already.
        var prev = chart.data.datasets.find(function (p) {
          return p.label === ds.label;
        });
        if (prev) ds.hidden = !!prev.hidden;
      }
    }
    chart.data.labels = newLabels;
    chart.data.datasets = newDatasets;
    // Re-evaluate per-group + global filter on the swapped dataset so the
    // visibility state matches what was on screen before the refetch. Also
    // refresh the group's series chip row in case the wider window surfaces
    // a series that was absent from the inline payload.
    applyFiltersToChart(card);
    noteSeriesFromCard(card);
    var newMaxIdx = Math.max(0, newLabels.length - 1);
    var zoomLimits = chart.options.plugins
      && chart.options.plugins.zoom
      && chart.options.plugins.zoom.limits
      && chart.options.plugins.zoom.limits.x;
    if (zoomLimits) {
      zoomLimits.max = newMaxIdx;
    }
    syncSliderBounds(card, newLabels.length);
    var sx = chart.options.scales.x;
    var prevMin = Number.isFinite(sx.min) ? sx.min : 0;
    var prevMax = Number.isFinite(sx.max) ? sx.max : 0;
    if (Number.isFinite(prevMin) && Number.isFinite(prevMax)) {
      sx.min = Math.max(0, Math.min(newMaxIdx, prevMin));
      sx.max = Math.max(sx.min, Math.min(newMaxIdx, prevMax));
    } else {
      var prevVisible = Math.max(1, prevMax - prevMin + 1);
      sx.max = newMaxIdx;
      sx.min = Math.max(0, newMaxIdx - (prevVisible - 1));
    }
    rebuildVisibleAndUpdate(card, chart, sx.min, sx.max);
    if (canvas.__bench_strip_render) canvas.__bench_strip_render();
  }

  // Mirror the chart's current visible commit count onto the toolbar
  // slider. Called from `rebuildVisibleAndUpdate` so every path that
  // changes the visible range — toolbar slider drag, drag-pan,
  // drag-rectangle-zoom, wheel-pan, range-strip drag — keeps the
  // slider in sync. Programmatic value writes do not fire the slider's
  // `input` event, so this never re-enters `applyScope`.
  function syncSliderFromRange(card, visibleCommits) {
    var slider = card.querySelector('[data-role="scope-slider"]');
    if (!slider) return;
    var lo = parseInt(slider.min, 10) || 1;
    var hi = parseInt(slider.max, 10) || visibleCommits;
    slider.value = String(Math.max(lo, Math.min(hi, visibleCommits)));
  }

  // Show the badge when at least one series in the visible range was
  // downsampled. The numbers are commit counts: how many distinct
  // commits the chart is rendering, and how many are in the visible
  // range. Both come from the slider's mental model so "300 / 3000" in
  // the badge matches "showing the last 3000" on the slider.
  function syncDownsampleBadge(card, keptCommits, visibleCommits, anyDownsampled) {
    var badge = card.querySelector('[data-role="downsample-badge"]');
    if (!badge) return;
    if (!anyDownsampled || keptCommits >= visibleCommits) {
      badge.setAttribute("hidden", "");
      badge.textContent = "";
      return;
    }
    badge.removeAttribute("hidden");
    badge.textContent = "downsampled · " + keptCommits + " / " + visibleCommits;
    badge.setAttribute(
      "title",
      "Showing " + keptCommits + " of " + visibleCommits
        + " commits in view. Each series renders at most "
        + MAX_VISIBLE_POINTS + " points at a time; when more are in "
        + "view, we apply LTTB (Largest Triangle, Three Buckets), an "
        + "algorithm that picks representative points by maximising "
        + "the area of triangles formed with neighbouring buckets. "
        + "Visual peaks and valleys are preserved while the chart "
        + "stays responsive. Zoom in past " + MAX_VISIBLE_POINTS
        + " visible commits to see every raw measurement."
    );
  }

  // -----------------------------------------------------------------------
  // Per-card construction. The set of `canvas.__bench_*` fields planted
  // by this function (and read elsewhere) is documented at the top of
  // this file under "Canvas state contract".
  // -----------------------------------------------------------------------
  function constructChart(card) {
    var idx = card.getAttribute("data-chart-index");
    var canvas = card.querySelector('canvas[data-chart-index="' + idx + '"]');
    if (!canvas || typeof Chart === "undefined") return null;
    if (canvas.__bench_chart) return canvas.__bench_chart;

    var payload = normalizeChartPayload(canvas.__bench_payload || readInlinePayload(idx));
    if (!payload) return null;
    canvas.__bench_payload = payload;
    // Latest-100 payloads are normalized onto the full x-axis. `history`
    // tells us whether old indices are virtual placeholders or real data.
    if (canvas.__bench_full_loaded === undefined) {
      var history = canonicalHistory(payload);
      canvas.__bench_full_loaded = !!history.complete;
      canvas.__bench_inline_trimmed = !canvas.__bench_full_loaded;
    }

    var state = canvas.__bench_state
      || { y: "linear", scope: defaultScopeForCard(card) };
    canvas.__bench_state = state;
    // Series labels the user has explicitly toggled on this card. Once a
    // label lands here, the global filter no longer drives that series's
    // hidden-state on this card — only direct legend clicks do.
    if (!canvas.__bench_overrides) canvas.__bench_overrides = {};
    // Lock the display unit for the lifetime of this loaded payload. We
    // recompute only when `replaceChartPayload` swaps in a wider window
    // after a `?n=all` refetch — toggling a series via the global filter
    // never touches it. See `pickDisplayUnit` for the full rationale.
    canvas.__bench_display_unit = pickDisplayUnit(
      payload.unit_kind, collectAllValues(payload),
    );

    var labels = (payload.commits || []).map(labelForCommit);
    var datasets = buildDatasets(payload);
    var host = card.querySelector(".chart-tooltip-host");
    var range = visibleRange(labels.length, state.scope);
    var legendPosition = (window.matchMedia
      && window.matchMedia("(max-width: 768px)").matches) ? "top" : "bottom";

    // Throttled rebuild for pan/zoom. Both axes mutate scales.x.min/max
    // continuously during interaction, so we re-derive the rendered
    // points on every frame (capped to PAN_THROTTLE_MS) and refresh the
    // range strip to match. Single throttle so LTTB and the strip never
    // diverge.
    var throttledRebuild = throttle(function (chart) {
      var sx = chart.scales && chart.scales.x;
      if (!sx) return;
      rebuildVisibleAndUpdate(card, chart, sx.min, sx.max, true);
      if (canvas.__bench_strip_render) canvas.__bench_strip_render();
    }, PAN_THROTTLE_MS);

    var chart = new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      plugins: [crosshairPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        // Snap to the single nearest commit *that has rendered data*.
        // After LTTB downsampling most commit indices are null in
        // `dataset.data`; `mode: "index"` would happily pick one of
        // those null indices and produce an empty tooltip, while
        // `mode: "x"` would pick multiple closely-packed LTTB columns
        // at once and the tooltip would render duplicate rows for the
        // same series at different commits. `mode: "nearest"` returns
        // exactly one closest data point — its `dataIndex` is then
        // used by the external handler as the single hovered commit,
        // and the handler iterates `chart.data.datasets` itself to
        // build one row per series. `intersect: false` keeps it
        // active anywhere on the chart and, combined with
        // `pointer-events: none` on the tooltip host, is also the
        // flicker fix.
        interaction: { mode: "nearest", intersect: false, axis: "x" },
        onClick: function (event, _activeElements, chart) {
          var points = chart.getElementsAtEventForMode(
            event, "nearest", { intersect: false, axis: "x" }, true,
          );
          if (!points.length) return;
          var pIdx = points[0].index;
          var commits = (canvas.__bench_payload || {}).commits || [];
          var commit = commits[pIdx];
          if (!commit) return;
          var pr = parsePrNumber(commit.message);
          var url = pr
            ? "https://github.com/vortex-data/vortex/pull/" + pr
            : commit.url;
          if (url) window.open(url, "_blank", "noopener");
        },
        scales: {
          y: {
            type: state.y === "log" ? "logarithmic" : "linear",
            beginAtZero: state.y !== "log",
            // Axis title reflects the locked display unit. Empty string when
            // the kind is dimensionless (`ratio`, `count`) so a "1.2× speedup"
            // chart doesn't get an arbitrary "value" label and a "12 m" chart
            // doesn't get read as anything other than `12 ms` / `12 s` / etc.
            title: {
              display: !!canvas.__bench_display_unit.axisLabel,
              text: canvas.__bench_display_unit.axisLabel,
            },
          },
          x: {
            min: range.min,
            max: range.max,
            title: { display: false },
            // With a 5000-commit history rendering one tick per commit
            // is unreadable anyway. Cap it; Chart.js will pick a sensible
            // subset of label indices to draw.
            ticks: { maxTicksLimit: 12, autoSkip: true },
          },
        },
        plugins: {
          legend: {
            position: legendPosition,
            // Wrap the default toggle so we record the per-card override
            // and keep `dataset.hidden` in sync with the legend's
            // `_hiddenInLegend` flag — the global filter pass writes to
            // `dataset.hidden`, so they need to track each other or
            // subsequent global changes look stale.
            onClick: function (e, item, legend) {
              var ci = legend.chart;
              var ds = ci.data.datasets[item.datasetIndex];
              var label = ds && ds.label;
              if (label && ci.canvas && ci.canvas.__bench_overrides) {
                ci.canvas.__bench_overrides[label] = true;
              }
              var visible = ci.isDatasetVisible(item.datasetIndex);
              ci.setDatasetVisibility(item.datasetIndex, !visible);
              if (ds) ds.hidden = visible; // flipped: was visible → now hidden, etc.
              ci.update();
            },
          },
          tooltip: {
            enabled: false,
            external: externalTooltipHandler(canvas, host),
            // Row ordering is handled inside the external handler now —
            // we iterate `chart.data.datasets` ourselves rather than the
            // tooltip's `dataPoints`, so `itemSort` here would be dead
            // code.
          },
          // chartjs-plugin-zoom config — wheel-zoom is disabled because we
          // want wheel-pan instead (handled by the canvas wheel listener
          // below). Drag-pan and drag-rectangle-zoom are free.
          zoom: {
            zoom: {
              wheel: { enabled: false },
              drag: {
                enabled: true,
                backgroundColor: "rgba(37, 99, 235, 0.10)",
              },
              mode: "x",
              onZoom: function (ctx) { throttledRebuild(ctx.chart); },
            },
            pan: {
              enabled: true,
              mode: "x",
              modifierKey: null,
              onPan: function (ctx) { throttledRebuild(ctx.chart); },
            },
            limits: {
              x: { min: 0, max: Math.max(0, labels.length - 1), minRange: 4 },
            },
          },
        },
      },
    });

    canvas.__bench_chart = chart;
    canvas.__bench_rebuild = throttledRebuild;
    attachWheelPan(canvas, chart, throttledRebuild);
    syncSliderBounds(card, labels.length);
    // Initial render: the chart is constructed with empty (null) data;
    // populate it for the initial visible window. Strip is bound after the
    // rebuild so its first paint reflects the same range Chart.js shows.
    rebuildVisibleAndUpdate(card, chart, range.min, range.max);
    bindRangeStrip(card, chart);
    if (canvas.__bench_strip_render) canvas.__bench_strip_render();
    // `buildDatasets` seeded `hidden` from the global filter; reapply through
    // the layered helper so a per-group filter set before this card hydrated
    // also takes effect. Then surface this card's series labels to the
    // group's filter dropdown so the chip row picks them up.
    applyFiltersToChart(card);
    noteSeriesFromCard(card);
    return chart;
  }

  // -----------------------------------------------------------------------
  // Range scrollbar strip — the thin track below each canvas. Spans the full
  // commit history; the highlighted "window" matches the chart's currently
  // visible x-range and can be dragged or its edges resized to pan/zoom.
  // -----------------------------------------------------------------------
  function bindRangeStrip(card, chart) {
    var strip = card.querySelector('[data-role="range-strip"]');
    if (!strip || strip.__bench_bound) return;
    strip.__bench_bound = true;
    var win = strip.querySelector('[data-role="range-window"]');
    var leftHandle = strip.querySelector('[data-role="range-handle-left"]');
    var rightHandle = strip.querySelector('[data-role="range-handle-right"]');
    if (!win || !leftHandle || !rightHandle) return;

    var canvas = card.querySelector("canvas");

    function commitCount() {
      return (chart.data.labels || []).length;
    }

    function visibleBounds() {
      var n = commitCount();
      if (n <= 0) return { min: 0, max: 0 };
      var maxIdx = n - 1;
      var sx = chart.options.scales.x || {};
      var min = Number.isFinite(sx.min) ? sx.min : 0;
      var max = Number.isFinite(sx.max) ? sx.max : maxIdx;
      min = Math.max(0, Math.min(maxIdx, min));
      max = Math.max(min, Math.min(maxIdx, max));
      return { min: min, max: max };
    }

    function render() {
      var n = commitCount();
      if (n <= 0) {
        win.style.left = "0%";
        win.style.width = "100%";
        return;
      }
      var b = visibleBounds();
      var span = Math.max(1, n - 1);
      var leftPct = (b.min / span) * 100;
      var widthPct = ((b.max - b.min) / span) * 100;
      // A minimum visible width keeps the handles grabbable when zoomed in
      // tight on a single commit.
      if (widthPct < 1.5) widthPct = 1.5;
      if (leftPct + widthPct > 100) leftPct = 100 - widthPct;
      win.style.left = leftPct + "%";
      win.style.width = widthPct + "%";
    }

    function setRange(newMin, newMax) {
      var n = commitCount();
      if (n <= 0) return;
      var maxIdx = n - 1;
      var minRange = 1; // matches plugin `limits.x.minRange = 4` loosely; allow tighter via strip
      newMin = Math.max(0, Math.min(maxIdx - minRange, newMin));
      newMax = Math.max(newMin + minRange, Math.min(maxIdx, newMax));
      chart.options.scales.x.min = newMin;
      chart.options.scales.x.max = newMax;
      // Track scope on the canvas so the toolbar slider stays consistent
      // when the user later drags it.
      if (canvas && canvas.__bench_state) {
        canvas.__bench_state.scope = Math.round(newMax - newMin + 1);
      }
      // Re-derive what Chart.js renders against the new visible window.
      // `rebuildVisibleAndUpdate` calls `chart.update("none")`, applies
      // LTTB, and mirrors the new scope onto the toolbar slider, so the
      // strip-driven pan/resize stays in lockstep with both the data
      // density and the slider readout.
      rebuildVisibleAndUpdate(card, chart, newMin, newMax, true);
      render();
    }

    function pxToIndex(px, trackWidth) {
      var n = commitCount();
      if (n <= 1 || trackWidth <= 0) return 0;
      var pct = Math.max(0, Math.min(1, px / trackWidth));
      return pct * (n - 1);
    }

    var dragState = null;

    function onPointerDown(e) {
      if (e.button !== undefined && e.button !== 0) return;
      var role = e.target.getAttribute && e.target.getAttribute("data-role");
      var rect = strip.getBoundingClientRect();
      var trackWidth = rect.width;
      var b = visibleBounds();
      var idxAtCursor = pxToIndex(e.clientX - rect.left, trackWidth);

      var mode;
      if (role === "range-handle-left") mode = "resize-left";
      else if (role === "range-handle-right") mode = "resize-right";
      else if (role === "range-window") mode = "pan";
      else {
        // Click on bare track: jump the window so its centre lands at the
        // cursor, then begin a pan drag.
        var width = b.max - b.min;
        var newMin = idxAtCursor - width / 2;
        setRange(newMin, newMin + width);
        b = visibleBounds();
        mode = "pan";
      }
      dragState = {
        mode: mode,
        rect: rect,
        startX: e.clientX,
        startMin: b.min,
        startMax: b.max,
        pointerId: e.pointerId,
      };
      try { strip.setPointerCapture(e.pointerId); } catch (err) {}
      e.preventDefault();
      strip.classList.add("chart-range-strip--dragging");
    }

    function onPointerMove(e) {
      if (!dragState) return;
      var n = commitCount();
      if (n <= 1) return;
      var trackWidth = dragState.rect.width;
      var dxPx = e.clientX - dragState.startX;
      var dxIdx = (dxPx / Math.max(1, trackWidth)) * (n - 1);
      if (dragState.mode === "pan") {
        setRange(dragState.startMin + dxIdx, dragState.startMax + dxIdx);
      } else if (dragState.mode === "resize-left") {
        setRange(dragState.startMin + dxIdx, dragState.startMax);
      } else if (dragState.mode === "resize-right") {
        setRange(dragState.startMin, dragState.startMax + dxIdx);
      }
    }

    function onPointerUp(e) {
      if (!dragState) return;
      try { strip.releasePointerCapture(dragState.pointerId); } catch (err) {}
      dragState = null;
      strip.classList.remove("chart-range-strip--dragging");
    }

    strip.addEventListener("pointerdown", onPointerDown);
    strip.addEventListener("pointermove", onPointerMove);
    strip.addEventListener("pointerup", onPointerUp);
    strip.addEventListener("pointercancel", onPointerUp);

    // Expose the strip's render function so other code paths (toolbar
    // slider, wheel-pan, the throttled LTTB rebuild) can keep the strip
    // in lockstep without each having to know strip internals. The chart
    // options' `onPan` / `onZoom` callbacks call this via the throttled
    // rebuild rather than overriding them here, so LTTB and the strip
    // refresh as one unit.
    canvas.__bench_strip_render = render;
    render();
  }

  // Cap the toolbar slider's `max` to the chart's full x-axis length. For a
  // latest-100 virtual payload this is intentionally larger than the loaded
  // point count, so "Show all" can expose the unloaded older range while the
  // full-history fetch is warming.
  function syncSliderBounds(card, commitCount) {
    var slider = card.querySelector('[data-role="scope-slider"]');
    if (!slider) return;
    var max = Math.max(5, commitCount);
    slider.max = String(max);
    // Pick a step that gives ~200 stops across the slider so dragging
    // feels continuous regardless of history size.
    var step = Math.max(1, Math.round(max / 200));
    slider.step = String(step);
    var current = parseInt(slider.value, 10);
    if (!Number.isFinite(current) || current > max) {
      var def = defaultScopeForCard(card);
      var seed = def === "all" ? max : Math.min(def, max);
      slider.value = String(seed);
    }
  }

  // Wheel = horizontal pan. Chart.js zoom plugin doesn't support wheel-pan
  // out of the box (wheel is always zoom in its config), so we attach a
  // `wheel` listener that translates `deltaY`/`deltaX` into `chart.pan` and
  // re-runs the rebuild after panning.
  function attachWheelPan(canvas, chart, rebuild) {
    if (canvas.__bench_wheel_attached) return;
    canvas.__bench_wheel_attached = true;
    canvas.addEventListener("wheel", function (e) {
      // Treat horizontal-wheel-or-shift+wheel as horizontal pan; otherwise
      // also pan on plain vertical wheel so trackpad scroll-up/down moves
      // through commit history without needing modifier keys.
      var dx = (Math.abs(e.deltaX) > Math.abs(e.deltaY)) ? e.deltaX : e.deltaY;
      if (!dx) return;
      e.preventDefault();
      // Browser wheel-down reports a positive delta. In Chart.js pan space,
      // positive x moves the visible window toward older commits, while
      // negative x moves back toward newer commits.
      chart.pan({ x: dx * 0.5 }, undefined, "none");
      // `rebuild` recomputes LTTB on the new visible range AND, via the
      // throttled wrapper, also calls `canvas.__bench_strip_render`.
      rebuild(chart);
    }, { passive: false });
  }

  // -----------------------------------------------------------------------
  // Recompute helpers driven by the per-chart toolbar.
  // -----------------------------------------------------------------------
  // Invariant: when `currentRange` is supplied AND the chart is already
  // panned away from the right edge, a scope change preserves the visible
  // CENTER instead of snapping to the most recent N commits. With no
  // `currentRange` (initial render) or a view that already covers
  // everything / sits flush with the newest commit, anchor to the right —
  // the right default at first load and after "show all".
  function visibleRange(commitCount, scope, currentRange) {
    if (commitCount <= 0) return { min: undefined, max: undefined };
    var maxIdx = commitCount - 1;
    if (scope === "all" || !Number.isFinite(scope) || scope <= 0 || scope >= commitCount) {
      return { min: 0, max: maxIdx };
    }
    var width = scope;
    var rightAnchored = { min: Math.max(0, maxIdx - (width - 1)), max: maxIdx };
    if (!currentRange) return rightAnchored;
    var curMin = Number.isFinite(currentRange.min) ? currentRange.min : 0;
    var curMax = Number.isFinite(currentRange.max) ? currentRange.max : maxIdx;
    var coversAll = curMin <= 0 && curMax >= maxIdx;
    // Half-commit tolerance: pan/zoom can leave fractional drift even when
    // the user is effectively still flush with the newest commit.
    var atRightEdge = curMax >= maxIdx - 0.5;
    if (coversAll || atRightEdge) return rightAnchored;
    var center = (curMin + curMax) / 2;
    var halfWidth = (width - 1) / 2;
    var newMin = Math.round(center - halfWidth);
    var newMax = newMin + (width - 1);
    if (newMin < 0) {
      newMin = 0;
      newMax = width - 1;
    } else if (newMax > maxIdx) {
      newMax = maxIdx;
      newMin = maxIdx - (width - 1);
    }
    return { min: newMin, max: newMax };
  }

  function applyScope(card, scopeValue) {
    var canvas = card.querySelector("canvas");
    var chart = canvas && canvas.__bench_chart;
    if (!chart) return;
    var commits = chart.data.labels.length;
    var scope = scopeValue === "all" ? "all" : parseInt(scopeValue, 10);
    canvas.__bench_state.scope = scope;
    // Capture the chart's existing visible window BEFORE we overwrite it,
    // so `visibleRange` can preserve the center when the user has panned
    // away from the right edge.
    var sx = chart.options.scales.x;
    var currentRange = sx ? { min: sx.min, max: sx.max } : null;
    var range = visibleRange(commits, scope, currentRange);
    chart.options.scales.x.min = range.min;
    chart.options.scales.x.max = range.max;
    rebuildVisibleAndUpdate(card, chart, range.min, range.max, true);
    syncToolbarUi(card, "scope", String(scopeValue));
    if (canvas.__bench_strip_render) canvas.__bench_strip_render();
  }

  // `userInitiated` defaults to true. Once set, the chart is "sticky" — the
  // per-group Y apply pass skips it on subsequent group-level clicks,
  // honouring the user's explicit per-card choice. The per-group toolbar
  // passes `false` so it doesn't pollute the flag while broadcasting.
  function applyY(card, yValue, userInitiated) {
    var canvas = card.querySelector("canvas");
    var chart = canvas && canvas.__bench_chart;
    if (!chart) return;
    if (userInitiated !== false) {
      canvas.__bench_y_user_set = true;
    }
    canvas.__bench_state.y = yValue;
    chart.options.scales.y.type = yValue === "log" ? "logarithmic" : "linear";
    chart.options.scales.y.beginAtZero = yValue !== "log";
    chart.update("none");
    syncToolbarUi(card, "y", yValue);
  }

  function syncToolbarUi(card, group, value) {
    var attr = "data-" + group;
    card.querySelectorAll(".toolbar-btn[" + attr + "]").forEach(function (b) {
      b.classList.toggle("toolbar-btn--active", b.getAttribute(attr) === value);
    });
    if (group === "scope") {
      var slider = card.querySelector('[data-role="scope-slider"]');
      if (slider && /^\d+$/.test(value)) slider.value = value;
    }
  }

  function bindToolbar(card) {
    var toolbar = card.querySelector(".toolbar--card");
    if (!toolbar || toolbar.__bench_bound) return;
    toolbar.__bench_bound = true;

    toolbar.addEventListener("click", function (e) {
      var btn = e.target.closest(".toolbar-btn");
      if (!btn || !toolbar.contains(btn)) return;
      if (btn.hasAttribute("data-y")) applyY(card, btn.getAttribute("data-y"));
    });

    var slider = toolbar.querySelector('[data-role="scope-slider"]');
    if (slider) {
      // `input` (continuous), throttled so dragging stays at ~60fps even on
      // pages with dozens of charts. Last value still lands because
      // `throttle` preserves the trailing call.
      var throttled = throttle(function () {
        applyScope(card, slider.value);
      }, ZOOM_THROTTLE_MS);
      slider.addEventListener("input", throttled);
    }
  }

  // -----------------------------------------------------------------------
  // Group hydration. Every landing group renders chart shells with versioned
  // shard metadata. On intent or first open we fetch materialized latest-100
  // group shards through a small per-tab queue and construct each chart as
  // soon as its shard arrives.
  // -----------------------------------------------------------------------
  var hydrationActive = 0;
  var hydrationQueue = [];
  var fullHistoryActive = 0;
  var fullHistoryQueue = [];
  var groupOpenPriority = 0;

  function scheduleHydration(task, priority) {
    var entry = {
      task: task,
      priority: priority || 0,
      promise: null,
      resolve: null,
      reject: null,
    };
    entry.promise = new Promise(function (resolve, reject) {
      entry.resolve = resolve;
      entry.reject = reject;
    });
    hydrationQueue.push(entry);
    drainHydrationQueue();
    return entry;
  }

  function drainHydrationQueue() {
    while (hydrationActive < HYDRATION_CONCURRENCY && hydrationQueue.length) {
      hydrationQueue.sort(function (a, b) { return b.priority - a.priority; });
      var item = hydrationQueue.shift();
      hydrationActive++;
      Promise.resolve()
        .then(item.task)
        .then(
          function (value) {
            hydrationActive--;
            item.resolve(value);
            drainHydrationQueue();
          },
          function (err) {
            hydrationActive--;
            item.reject(err);
            drainHydrationQueue();
          }
        );
    }
  }

  function scheduleFullHistory(task, priority) {
    var entry = {
      task: task,
      priority: priority || 0,
      promise: null,
      resolve: null,
      reject: null,
    };
    entry.promise = new Promise(function (resolve, reject) {
      entry.resolve = resolve;
      entry.reject = reject;
    });
    fullHistoryQueue.push(entry);
    drainFullHistoryQueue();
    return entry;
  }

  function drainFullHistoryQueue() {
    while (fullHistoryActive < FULL_HISTORY_CONCURRENCY && fullHistoryQueue.length) {
      fullHistoryQueue.sort(function (a, b) { return b.priority - a.priority; });
      var item = fullHistoryQueue.shift();
      fullHistoryActive++;
      Promise.resolve()
        .then(item.task)
        .then(
          function (value) {
            fullHistoryActive--;
            item.resolve(value);
            drainFullHistoryQueue();
          },
          function (err) {
            fullHistoryActive--;
            item.reject(err);
            drainFullHistoryQueue();
          }
        );
    }
  }

  function priorityForGroupOpen(group) {
    groupOpenPriority += GROUP_OPEN_PRIORITY_STEP;
    group.__bench_group_priority = groupOpenPriority;
    return groupOpenPriority;
  }

  function fetchAndConstruct(card) {
    var canvas = card.querySelector("canvas");
    if (!canvas) return Promise.resolve();
    if (canvas.__bench_chart) return Promise.resolve();
    if (constructChart(card)) {
      bindToolbar(card);
    }
    return Promise.resolve();
  }

  function groupCards(group) {
    return Array.prototype.slice.call(
      group.querySelectorAll(".chart-card[data-chart-slug]")
    );
  }

  function cardHasPayloadAvailable(card) {
    var canvas = card.querySelector("canvas");
    if (!canvas) return true;
    if (canvas.__bench_payload) return true;
    var idx = card.getAttribute("data-chart-index");
    return idx != null && !!readInlinePayload(idx);
  }

  function groupShardCount(group) {
    var n = parseInt(group.getAttribute("data-group-shard-count") || "0", 10);
    return Number.isFinite(n) && n > 0 ? n : 0;
  }

  function groupShardPrefix(group) {
    return group.getAttribute("data-group-shard-prefix") || "";
  }

  function groupShardUrl(group, index) {
    var prefix = groupShardPrefix(group);
    return prefix ? prefix + encodeURIComponent(String(index)) : "";
  }

  function groupHydrationState(group) {
    if (!group.__bench_group_hydration) {
      group.__bench_group_hydration = {
        loaded: {},
        pending: {},
        entries: {},
        errors: {},
      };
    }
    return group.__bench_group_hydration;
  }

  function cardBySlug(group, slug) {
    var cards = groupCards(group);
    for (var i = 0; i < cards.length; i++) {
      if (cards[i].getAttribute("data-chart-slug") === slug) return cards[i];
    }
    return null;
  }

  function showCardLoading(card, on) {
    var existing = card.querySelector(".chart-loading");
    if (on) {
      if (existing) return;
      var el = document.createElement("div");
      el.className = "chart-loading";
      el.textContent = "loading…";
      card.appendChild(el);
    } else if (existing) {
      existing.remove();
    }
  }

  function showCardError(card, msg) {
    var existing = card.querySelector(".chart-error");
    if (existing) existing.remove();
    var el = document.createElement("div");
    el.className = "chart-error";
    el.textContent = msg;
    card.appendChild(el);
    setTimeout(function () { if (el.parentNode) el.remove(); }, 4000);
  }

  function applyGroupShard(group, shard) {
    if (!shard || !Array.isArray(shard.charts)) return;
    shard.charts.forEach(function (payload) {
      if (!payload || !payload.slug) return;
      var card = cardBySlug(group, payload.slug);
      var canvas = card && card.querySelector("canvas");
      if (!canvas) return;
      if (!canvas.__bench_full_loaded) {
        canvas.__bench_payload = normalizeChartPayload(payload);
        canvas.__bench_payload_window = CHART_FETCH_N;
      }
      showCardLoading(card, false);
      if (groupIsOpen(group)) fetchAndConstruct(card);
    });
  }

  function fetchGroupShard(group, index, priority) {
    var state = groupHydrationState(group);
    if (state.loaded[index]) return Promise.resolve();
    if (state.pending[index]) {
      if (state.entries[index] && priority) {
        state.entries[index].priority = Math.max(state.entries[index].priority, priority);
        drainHydrationQueue();
      }
      return state.pending[index];
    }
    var url = groupShardUrl(group, index);
    if (!url) return Promise.resolve();
    var entry = scheduleHydration(function () {
      return fetch(url, { headers: { "accept": "application/json" } })
        .then(function (r) {
          if (r.status === 404) throw new Error("not found");
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (payload) {
          state.loaded[index] = true;
          state.errors[index] = null;
          applyGroupShard(group, payload);
        });
    }, priority);
    state.entries[index] = entry;
    state.pending[index] = entry.promise.catch(function (err) {
      state.errors[index] = err;
      if (index === 0) {
        groupCards(group).forEach(function (card) {
          if (!cardHasPayloadAvailable(card)) {
            showCardLoading(card, false);
            showCardError(card, "failed to load: " + (err.message || "unknown error"));
          }
        });
      }
    }).then(function () {
      state.pending[index] = null;
      state.entries[index] = null;
    });
    return state.pending[index];
  }

  function groupIsOpen(group) {
    var details = group.querySelector("details.group-disclosure");
    return !details || details.open;
  }

  function queueRemainingGroupShards(group, priority) {
    var count = groupShardCount(group);
    for (var i = 1; i < count; i++) fetchGroupShard(group, i, priority || 0);
  }

  function queueGroupFullHistory(group, priority) {
    groupCards(group).forEach(function (card) {
      ensureFullHistory(card, priority || 0);
    });
  }

  function hydrateGroupShardZero(group, showLoading, priority) {
    if (showLoading) {
      groupCards(group).forEach(function (card) {
        if (!cardHasPayloadAvailable(card)) showCardLoading(card, true);
      });
    }
    return fetchGroupShard(group, 0, priority || (showLoading ? 1 : 0)).then(function () {
      if (showLoading) {
        groupCards(group).forEach(function (card) {
          if (cardHasPayloadAvailable(card)) showCardLoading(card, false);
        });
      }
    });
  }

  function hydrateOpenGroup(disclosure) {
    if (!disclosure || !disclosure.open) return;
    var group = disclosure.closest(".group-details");
    if (!group) return;
    var priority = priorityForGroupOpen(group);
    hydrateGroupShardZero(group, true, priority + 20).then(function () {
      queueRemainingGroupShards(group, priority + 10);
      queueGroupFullHistory(group, priority);
    });
  }

  function prefetchGroupOnIntent(group) {
    fetchGroupShard(group, 0, 0);
  }

  // -----------------------------------------------------------------------
  // Global filter bar wiring.
  //
  // Chips live in `.global-filter-bar`. Click a non-"all" chip to toggle
  // that engine/format in/out of the active set; click "all" to clear the
  // filter for that dimension. After every change we:
  //   1. Re-paint the chips.
  //   2. Walk every chart on the page and re-apply the filter (skipping
  //      series the user has explicitly overridden on that card).
  //   3. Sync the URL with `history.replaceState` so a refresh / share
  //      preserves the view.
  // -----------------------------------------------------------------------
  // Apply the layered filter on a single card. Layer order matches the
  // resolution rule documented at the top of the file:
  //   1. Per-card legend overrides (`canvas.__bench_overrides`) win.
  //   2. Per-group filter (`section.__bench_group_filter`) hides next.
  //   3. Global filter hides last.
  //   4. Otherwise show.
  // Used by every code path that mutates filter state (global chip clicks,
  // per-group chip clicks, post-construction seeding).
  function applyFiltersToChart(card) {
    var canvas = card.querySelector("canvas");
    var chart = canvas && canvas.__bench_chart;
    if (!chart) return;
    var overrides = canvas.__bench_overrides || {};
    var section = card.closest(".group-details");
    var groupFilter = section && section.__bench_group_filter;
    var datasets = chart.data.datasets || [];
    for (var i = 0; i < datasets.length; i++) {
      var ds = datasets[i];
      if (overrides[ds.label]) continue;
      var hidden = false;
      if (!seriesPassesGroupFilter(groupFilter, ds.label)) {
        hidden = true;
      } else if (!seriesPassesFilter(ds.benchMeta)) {
        hidden = true;
      }
      // Use the dataset.hidden field directly so the legend stays in sync;
      // setDatasetVisibility writes into a separate visibility map.
      ds.hidden = hidden;
    }
    chart.update("none");
  }

  function applyGlobalFilterEverywhere() {
    document.querySelectorAll(".chart-card[data-chart-index]").forEach(function (card) {
      applyFiltersToChart(card);
    });
  }

  function syncFilterChipsUi() {
    var bar = document.querySelector('[data-role="global-filter-bar"]');
    if (!bar) return;
    bar.querySelectorAll(".filter-chip").forEach(function (chip) {
      var dim = chip.getAttribute("data-filter");
      var value = chip.getAttribute("data-value");
      var key = dim === "engine" ? "engines" : "formats";
      var list = globalFilter[key];
      var active;
      if (value === "*") {
        // The "all" chip is a one-shot reset, never a "current state"
        // indicator — leave it inactive. Pressing it forces every other
        // chip in the row back to active.
        active = false;
      } else {
        active = list.indexOf(value) !== -1;
      }
      chip.classList.toggle("filter-chip--active", active);
      chip.setAttribute("aria-pressed", active ? "true" : "false");
    });
    syncFilterBadge();
  }

  // Show a badge on the trigger that counts how many chips are *off*
  // (i.e. how many things the global filter is hiding). Hidden when the
  // filter is fully open, so it's noise-free in the resting state.
  function syncFilterBadge() {
    var trigger = document.querySelector('[data-role="filter-trigger"]');
    if (!trigger) return;
    var hidden =
      Math.max(0, filterUniverse.engines.length - globalFilter.engines.length) +
      Math.max(0, filterUniverse.formats.length - globalFilter.formats.length);
    var badge = trigger.querySelector('[data-role="filter-badge"]');
    if (hidden === 0) {
      if (badge) badge.remove();
      return;
    }
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "filter-badge";
      badge.setAttribute("data-role", "filter-badge");
      trigger.appendChild(badge);
    }
    badge.textContent = String(hidden);
  }

  function syncFilterUrl() {
    if (!window.history || !window.history.replaceState) return;
    var url = new URL(window.location.href);
    // URL stays as an allowlist (`?engine=duckdb` = "show only duckdb"). We
    // emit the param only when the active set is a strict subset of the
    // universe; an all-active row leaves the URL clean.
    syncDimensionUrl(url, "engine", "engines");
    syncDimensionUrl(url, "format", "formats");
    window.history.replaceState(null, "", url.toString());
  }

  function syncDimensionUrl(url, paramName, key) {
    if (dimensionIsFiltered(key)) {
      url.searchParams.set(paramName, globalFilter[key].join(","));
    } else {
      url.searchParams.delete(paramName);
    }
  }

  // Toggle one chip independently. The "all" chip resets the dimension to
  // every-chip-active; specific chips just flip their own active state.
  function toggleFilterValue(dim, value) {
    var key = dim === "engine" ? "engines" : "formats";
    if (value === "*") {
      globalFilter[key] = filterUniverse[key].slice();
      return;
    }
    var list = globalFilter[key];
    var idx = list.indexOf(value);
    if (idx === -1) {
      list.push(value);
    } else {
      list.splice(idx, 1);
    }
  }

  // Toggle the dropdown panel open/closed. Click outside or press Escape
  // to close. The panel is anchored under the trigger button via CSS.
  function bindFilterDropdown() {
    var bar = document.querySelector('[data-role="global-filter-bar"]');
    if (!bar) return;
    var trigger = bar.querySelector('[data-role="filter-trigger"]');
    var panel = bar.querySelector('[data-role="filter-panel"]');
    if (!trigger || !panel) return;

    function setOpen(open) {
      if (open) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "");
      }
      trigger.setAttribute("aria-expanded", open ? "true" : "false");
      bar.classList.toggle("filter-dropdown--open", open);
    }
    trigger.addEventListener("click", function (e) {
      e.stopPropagation();
      var isOpen = !panel.hasAttribute("hidden");
      setOpen(!isOpen);
    });
    document.addEventListener("click", function (e) {
      if (!bar.contains(e.target)) setOpen(false);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });
  }

  function initGlobalFilterBar() {
    var bar = document.querySelector('[data-role="global-filter-bar"]');
    if (!bar) return;
    bar.addEventListener("click", function (e) {
      var chip = e.target.closest(".filter-chip");
      if (!chip || !bar.contains(chip)) return;
      var dim = chip.getAttribute("data-filter");
      var value = chip.getAttribute("data-value");
      if (!dim || !value) return;
      toggleFilterValue(dim, value);
      syncFilterChipsUi();
      applyGlobalFilterEverywhere();
      syncFilterUrl();
    });
    bindFilterDropdown();
    syncFilterChipsUi();
  }

  // -----------------------------------------------------------------------
  // Per-group toolbar wiring.
  //
  // Each `.group-details` section carries a `[data-role="group-toolbar"]`
  // with Y-axis buttons and a centered filter dropdown. State lives on the
  // section node:
  //   section.__bench_group_filter = { hiddenSeries: [<dataset.label>, ...] }
  //   section.__bench_group_y      = "linear" | "log" | null
  //   section.__bench_known_series = { <label>: { engine, format, ... } }
  //
  // Empty `hiddenSeries` and `null` Y mean "no group override; defer to the
  // next layer". Engine and format chips in the dropdown are macros: a click
  // computes every known series whose `engine`/`format` matches and bulk-
  // toggles their membership in `hiddenSeries`. The series chips are
  // populated lazily via `noteSeriesFromCard` as charts in the group hydrate.
  // -----------------------------------------------------------------------
  function ensureGroupFilter(section) {
    if (!section.__bench_group_filter) {
      section.__bench_group_filter = { hiddenSeries: [] };
    } else if (!section.__bench_group_filter.hiddenSeries) {
      section.__bench_group_filter.hiddenSeries = [];
    }
    return section.__bench_group_filter;
  }

  function ensureKnownSeries(section) {
    if (!section.__bench_known_series) {
      section.__bench_known_series = {};
    }
    return section.__bench_known_series;
  }

  // Pull every series label from the card's payload into the section's
  // running set. Returns true when at least one new label was added so the
  // caller knows whether to re-render the chip row.
  function harvestSeriesFromCanvas(section, canvas) {
    var payload = canvas && canvas.__bench_payload;
    var meta = payload && payload.series_meta;
    if (!meta) return false;
    var known = ensureKnownSeries(section);
    var added = false;
    Object.keys(meta).forEach(function (label) {
      if (!known[label]) {
        known[label] = meta[label] || {};
        added = true;
      }
    });
    return added;
  }

  // Render one button per known series into the dropdown's series row.
  // Wipes and rebuilds — the row is small (typically <10 chips) so this is
  // cheap and avoids tracking per-label DOM nodes. Visibility state is then
  // resynced via `syncGroupChipsUi`.
  function renderGroupSeriesChips(section) {
    var container = section.querySelector('[data-role="group-series-chips"]');
    if (!container) return;
    var known = ensureKnownSeries(section);
    var labels = Object.keys(known).sort();
    while (container.firstChild) container.removeChild(container.firstChild);
    labels.forEach(function (label) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "filter-chip";
      btn.setAttribute("data-group-filter", "series");
      btn.setAttribute("data-value", label);
      btn.textContent = label;
      container.appendChild(btn);
    });
  }

  // Called whenever a card's payload becomes available (constructChart,
  // replaceChartPayload). Folds new series labels into the section's
  // running set, refreshes the dropdown chip row when the set grew, and
  // re-syncs all chip + badge visuals against the current filter state.
  function noteSeriesFromCard(card) {
    var section = card.closest && card.closest(".group-details");
    if (!section) return;
    var canvas = card.querySelector("canvas");
    if (!canvas) return;
    if (harvestSeriesFromCanvas(section, canvas)) {
      renderGroupSeriesChips(section);
    }
    syncGroupChipsUi(section);
    syncGroupFilterBadge(section);
  }

  // Toggle a single series label in/out of the hidden set.
  function toggleGroupSeriesValue(section, label) {
    var filter = ensureGroupFilter(section);
    var idx = filter.hiddenSeries.indexOf(label);
    if (idx === -1) filter.hiddenSeries.push(label);
    else filter.hiddenSeries.splice(idx, 1);
  }

  // Apply an engine/format macro click. Find every known series whose meta
  // matches. If every match is currently visible, hide them all; otherwise
  // (any match already hidden) show them all. The result is that the macro
  // chip toggles between "all matching visible" and "all matching hidden",
  // which mirrors the chip's own active-state semantics.
  function applyMacroToHiddenSeries(section, dim, value) {
    var filter = ensureGroupFilter(section);
    var known = ensureKnownSeries(section);
    var matching = [];
    Object.keys(known).forEach(function (label) {
      if (known[label] && known[label][dim] === value) matching.push(label);
    });
    if (!matching.length) return;
    var allVisible = matching.every(function (l) {
      return filter.hiddenSeries.indexOf(l) === -1;
    });
    if (allVisible) {
      matching.forEach(function (l) {
        if (filter.hiddenSeries.indexOf(l) === -1) filter.hiddenSeries.push(l);
      });
    } else {
      filter.hiddenSeries = filter.hiddenSeries.filter(function (l) {
        return matching.indexOf(l) === -1;
      });
    }
  }

  function syncGroupChipsUi(section) {
    var filter = ensureGroupFilter(section);
    var known = ensureKnownSeries(section);
    section.querySelectorAll(
      '[data-role="group-toolbar"] .filter-chip[data-group-filter]',
    ).forEach(function (chip) {
      var dim = chip.getAttribute("data-group-filter");
      var value = chip.getAttribute("data-value");
      var active;
      if (value === "*") {
        // The "all" chip is a one-shot reset, never a "current state"
        // indicator — leave it inactive in every row.
        active = false;
      } else if (dim === "series") {
        active = filter.hiddenSeries.indexOf(value) === -1;
      } else if (dim === "engine" || dim === "format") {
        // Macro chip is active iff at least one known series matches this
        // dim AND every match is currently visible. When no series in the
        // group has this engine/format the chip is inert — show it inactive
        // so the dropdown doesn't falsely advertise irrelevant filters.
        var matching = Object.keys(known).filter(function (l) {
          return known[l] && known[l][dim] === value;
        });
        if (matching.length === 0) {
          active = false;
        } else {
          active = matching.every(function (l) {
            return filter.hiddenSeries.indexOf(l) === -1;
          });
        }
      } else {
        active = false;
      }
      chip.classList.toggle("filter-chip--active", active);
      chip.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  // Show a count of hidden series on the trigger button when at least one
  // is hidden; remove the badge cleanly when the filter is empty so the
  // resting state stays noise-free. Mirrors `syncFilterBadge` for the
  // global filter.
  function syncGroupFilterBadge(section) {
    var trigger = section.querySelector('[data-role="group-filter-trigger"]');
    if (!trigger) return;
    var filter = ensureGroupFilter(section);
    var hidden = filter.hiddenSeries.length;
    var badge = trigger.querySelector('[data-role="group-filter-badge"]');
    if (hidden === 0) {
      if (badge) badge.remove();
      return;
    }
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "filter-badge";
      badge.setAttribute("data-role", "group-filter-badge");
      trigger.appendChild(badge);
    }
    badge.textContent = String(hidden);
  }

  // Highlight whichever group-Y button matches the current state. `null`
  // (the resting default and the post-Reset state) is treated as "linear"
  // for the visual — matches each chart's own default — even though the
  // resolution rule still distinguishes "no override" from an explicit
  // user click for `applyGroupYTo`'s revert-to-linear semantics.
  function syncGroupYUi(section) {
    var y = section.__bench_group_y;
    var visual = y === "log" ? "log" : "linear";
    section.querySelectorAll(
      '[data-role="group-toolbar"] .toolbar-btn[data-group-y]',
    ).forEach(function (b) {
      var match = b.getAttribute("data-group-y") === visual;
      b.classList.toggle("toolbar-btn--active", match);
    });
  }

  // Re-evaluate every chart-card in the section under the unified filter
  // resolution. Per-group filter changes cascade through `applyFiltersToChart`
  // because that function reads the section's filter on every call.
  function applyGroupFilterTo(section) {
    section.querySelectorAll(".chart-card[data-chart-index]").forEach(function (card) {
      applyFiltersToChart(card);
    });
  }

  // Broadcast the per-group Y-axis setting. Skips cards with
  // `__bench_y_user_set` so the user's per-chart click stays sticky. When
  // the section's group Y is null (e.g. after Reset), revert non-overridden
  // cards to the default linear scale.
  function applyGroupYTo(section) {
    var y = section.__bench_group_y;
    var target = (y === "linear" || y === "log") ? y : "linear";
    section.querySelectorAll(".chart-card[data-chart-index]").forEach(function (card) {
      var canvas = card.querySelector("canvas");
      if (!canvas || !canvas.__bench_chart) return;
      if (canvas.__bench_y_user_set) return;
      applyY(card, target, false);
    });
  }

  // Open/close behaviour for the per-group filter dropdown. Mirrors
  // `bindFilterDropdown` for the global filter — click outside or press
  // Escape to close. The trigger calls `e.stopPropagation()` so the
  // document-level "click outside" listener doesn't immediately reclose
  // the panel on the same click that opened it.
  function bindGroupFilterDropdown(section) {
    var dropdown = section.querySelector('[data-role="group-filter-dropdown"]');
    if (!dropdown || dropdown.__bench_bound) return;
    dropdown.__bench_bound = true;
    var trigger = dropdown.querySelector('[data-role="group-filter-trigger"]');
    var panel = dropdown.querySelector('[data-role="group-filter-panel"]');
    if (!trigger || !panel) return;
    function setOpen(open) {
      if (open) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "");
      }
      trigger.setAttribute("aria-expanded", open ? "true" : "false");
      dropdown.classList.toggle("group-filter-dropdown--open", open);
    }
    trigger.addEventListener("click", function (e) {
      e.stopPropagation();
      var isOpen = !panel.hasAttribute("hidden");
      setOpen(!isOpen);
    });
    document.addEventListener("click", function (e) {
      if (!dropdown.contains(e.target)) setOpen(false);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });
  }

  function bindGroupToolbar(section) {
    var toolbar = section.querySelector('[data-role="group-toolbar"]');
    if (!toolbar || toolbar.__bench_bound) return;
    toolbar.__bench_bound = true;
    bindGroupFilterDropdown(section);
    toolbar.addEventListener("click", function (e) {
      var target = e.target;
      var resetBtn = target.closest && target.closest('[data-role="group-toolbar-reset"]');
      if (resetBtn && toolbar.contains(resetBtn)) {
        section.__bench_group_filter = { hiddenSeries: [] };
        section.__bench_group_y = null;
        syncGroupChipsUi(section);
        syncGroupYUi(section);
        syncGroupFilterBadge(section);
        applyGroupYTo(section);
        applyGroupFilterTo(section);
        return;
      }
      var yBtn = target.closest && target.closest('.toolbar-btn[data-group-y]');
      if (yBtn && toolbar.contains(yBtn)) {
        section.__bench_group_y = yBtn.getAttribute("data-group-y");
        syncGroupYUi(section);
        applyGroupYTo(section);
        return;
      }
      var chip = target.closest && target.closest('.filter-chip[data-group-filter]');
      if (chip && toolbar.contains(chip)) {
        var dim = chip.getAttribute("data-group-filter");
        var value = chip.getAttribute("data-value");
        if (!dim || !value) return;
        if (value === "*") {
          ensureGroupFilter(section).hiddenSeries = [];
        } else if (dim === "series") {
          toggleGroupSeriesValue(section, value);
        } else {
          applyMacroToHiddenSeries(section, dim, value);
        }
        syncGroupChipsUi(section);
        syncGroupFilterBadge(section);
        applyGroupFilterTo(section);
      }
    });
  }

  function initGroupToolbars() {
    document.querySelectorAll(".group-details").forEach(function (section) {
      bindGroupToolbar(section);
      syncGroupChipsUi(section);
      syncGroupYUi(section);
      syncGroupFilterBadge(section);
    });
  }

  // -----------------------------------------------------------------------
  // Header controls
  // -----------------------------------------------------------------------
  function effectiveTheme() {
    var forced = document.documentElement.getAttribute("data-theme");
    if (forced === "light" || forced === "dark") return forced;
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }
    return "light";
  }

  function setTheme(theme) {
    if (theme === "light" || theme === "dark") {
      document.documentElement.setAttribute("data-theme", theme);
      try { localStorage.setItem("bench-theme", theme); } catch (e) {}
    }
    updateThemeButtons();
  }

  function updateThemeButtons() {
    var next = effectiveTheme() === "light" ? "Dark" : "Light";
    var nextTheme = next.toLowerCase();
    document.querySelectorAll('[data-role="theme-toggle"]').forEach(function (btn) {
      var label = btn.querySelector(".theme-toggle-label");
      if (label) label.textContent = next;
      btn.setAttribute("data-next-theme", nextTheme);
      btn.setAttribute("aria-label", "Switch to " + nextTheme + " mode");
    });
  }

  function setAllGroups(open) {
    document.querySelectorAll("details.group-disclosure").forEach(function (disclosure) {
      disclosure.open = open;
      if (open) hydrateOpenGroup(disclosure);
    });
  }

  function initHeaderControls() {
    updateThemeButtons();
    document.querySelectorAll('[data-role="theme-toggle"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        setTheme(effectiveTheme() === "light" ? "dark" : "light");
      });
    });
    document.querySelectorAll('[data-action="expand-all"]').forEach(function (btn) {
      btn.addEventListener("click", function () { setAllGroups(true); });
    });
    document.querySelectorAll('[data-action="collapse-all"]').forEach(function (btn) {
      btn.addEventListener("click", function () { setAllGroups(false); });
    });
    bindMobileNav();
  }

  // Hamburger toggle for the mobile-only nav panel. The panel itself is
  // `.nav-controls`; CSS hides it at < 769px until `.nav-controls--open`
  // is planted on it. Mirrors the open/close-on-outside-click pattern used
  // by the global filter dropdown.
  function bindMobileNav() {
    var toggle = document.querySelector('[data-role="nav-mobile-toggle"]');
    var nav = document.querySelector('[data-role="nav-controls"]');
    if (!toggle || !nav) return;
    function setOpen(open) {
      nav.classList.toggle("nav-controls--open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }
    toggle.addEventListener("click", function (e) {
      e.stopPropagation();
      var isOpen = nav.classList.contains("nav-controls--open");
      setOpen(!isOpen);
    });
    document.addEventListener("click", function (e) {
      if (!nav.contains(e.target) && !toggle.contains(e.target)) {
        setOpen(false);
      }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });
  }

  // -----------------------------------------------------------------------
  // Page wiring
  // -----------------------------------------------------------------------
  function initOpenCharts() {
    // Charts that arrive with inline JSON (`<script id="chart-data-N">`):
    // construct them via IntersectionObserver as before so a long open page
    // doesn't pay for offscreen Chart.js cost up front.
    var cards = document.querySelectorAll(".chart-card[data-chart-index]");
    if (!cards.length) return;

    var construct = function (card) {
      // Skip cards whose group disclosure is closed — they'll be picked up
      // on toggle. Summaries live outside the disclosure so they remain
      // visible while the chart grid is collapsed.
      var group = card.closest(".group-details");
      var details = group ? group.querySelector("details.group-disclosure") : null;
      if (details && !details.open) return;
      if (constructChart(card)) bindToolbar(card);
    };

    if (typeof IntersectionObserver === "undefined") {
      cards.forEach(construct);
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            construct(entry.target);
            io.unobserve(entry.target);
          }
        });
      }, { rootMargin: "150px 0px" });
      cards.forEach(function (card) { io.observe(card); });
    }
  }

  function initDetailsToggle() {
    var groups = document.querySelectorAll("details.group-disclosure");
    groups.forEach(function (d) {
      d.addEventListener("toggle", function () {
        if (!d.open) return;
        hydrateOpenGroup(d);
      });
      if (d.open) hydrateOpenGroup(d);
    });
  }

  function initGroupIntentPrefetch() {
    document.querySelectorAll(".group-details").forEach(function (group) {
      if (group.__bench_intent_prefetch_bound) return;
      group.__bench_intent_prefetch_bound = true;
      var summary = group.querySelector(".group-summary");
      var target = summary || group;
      target.addEventListener("pointerenter", function () {
        prefetchGroupOnIntent(group);
      });
      target.addEventListener("focusin", function () {
        prefetchGroupOnIntent(group);
      });
    });
  }

  function init() {
    initHeaderControls();
    initGlobalFilterBar();
    initGroupToolbars();
    initDetailsToggle();
    initGroupIntentPrefetch();
    initOpenCharts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
