// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Hydrate the Chart.js line chart on /chart/:slug.
//
// The server embeds the chart payload as a JSON <script id="chart-data">
// block matching the /api/chart/:slug shape:
//   { display_name, unit, commits: [{sha,timestamp,message,url}], series: { name: [Number|null] } }
//
// On top of the chart we wire:
// * Tooltip rows sorted top-to-bottom by current value (Task D).
// * A short, sanitized commit summary in the tooltip — no full URL (Task D).
// * Click-to-PR: parse `(#NNNN)` from the commit's first line and open the
//   PR; fall back to the raw commit URL (Task D).
// * A horizontal range scrollbar beneath the canvas, two-way wired with
//   the chart's x-axis min/max (Task C).
// * Series visibility chips driven by the `?engine=` / `?format=` /
//   `?hidden=` URL params (Task E).
(function () {
  "use strict";

  var palette = [
    "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed",
    "#0891b2", "#ca8a04", "#db2777", "#65a30d", "#475569",
  ];

  function colorFor(i) {
    return palette[i % palette.length];
  }

  function shortSha(sha) {
    return typeof sha === "string" ? sha.slice(0, 7) : String(sha);
  }

  function firstLine(s) {
    if (typeof s !== "string") return "";
    var nl = s.indexOf("\n");
    return nl === -1 ? s : s.slice(0, nl);
  }

  function truncate(s, n) {
    if (typeof s !== "string") return "";
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  // Match Vortex's squash-merge convention: PR number is parenthesized at
  // the end of the subject line, e.g. "fix: bla (#7642)".
  function parsePrNumber(message) {
    if (typeof message !== "string") return null;
    var m = firstLine(message).match(/\(#(\d+)\)\s*$/);
    return m ? parseInt(m[1], 10) : null;
  }

  function readPayload() {
    var node = document.getElementById("chart-data");
    if (!node) {
      throw new Error("missing #chart-data element");
    }
    return JSON.parse(node.textContent);
  }

  // Series labels follow `engine:format` (query/compression-time charts) or
  // a plain `format` (compression-size, random-access). The filter machinery
  // tests engine and format independently so a partial label still matches.
  function seriesMatchesFilter(label, engineFilter, formatFilter) {
    if (!engineFilter && !formatFilter) return true;
    var parts = label.split(":");
    var engine = parts.length > 1 ? parts[0] : null;
    var format = parts.length > 1 ? parts.slice(1).join(":") : parts[0];
    if (engineFilter && engine && engine !== engineFilter) return false;
    if (formatFilter && format && format !== formatFilter) return false;
    return true;
  }

  function readUrlParams() {
    var u = new URL(window.location.href);
    var hidden = (u.searchParams.get("hidden") || "")
      .split(",")
      .map(function (s) { return s.trim(); })
      .filter(Boolean);
    return {
      engine: u.searchParams.get("engine") || "",
      format: u.searchParams.get("format") || "",
      hidden: new Set(hidden),
    };
  }

  function writeHiddenParam(hiddenSet) {
    var u = new URL(window.location.href);
    if (hiddenSet.size === 0) {
      u.searchParams.delete("hidden");
    } else {
      var arr = [];
      hiddenSet.forEach(function (v) { arr.push(v); });
      arr.sort();
      u.searchParams.set("hidden", arr.join(","));
    }
    window.history.replaceState(null, "", u.toString());
  }

  function buildDatasets(series, names, filterState) {
    return names.map(function (name, i) {
      var hiddenByFilter = !seriesMatchesFilter(name, filterState.engine, filterState.format);
      var hiddenExplicit = filterState.hidden.has(name);
      return {
        label: name,
        data: series[name],
        borderColor: colorFor(i),
        backgroundColor: colorFor(i),
        spanGaps: true,
        tension: 0.1,
        pointRadius: 3,
        pointHoverRadius: 5,
        hidden: hiddenByFilter || hiddenExplicit,
      };
    });
  }

  // ---- Range scrubber (Task C) -------------------------------------------
  //
  // A 16px horizontal strip placed beneath the chart canvas. Renders a
  // highlighted "window" rectangle reflecting the chart's current x-axis
  // (min, max) clamped to (0, n-1). Drag the window body to pan; drag
  // either edge to grow or shrink; click outside the window to recenter
  // it on the click point. Hooks into Chart.js via the `afterUpdate`
  // plugin event so programmatic zoom updates also redraw the strip.
  function attachRangeScrubber(chart, total) {
    var wrap = chart.canvas.closest(".chart-wrap");
    if (!wrap) return;
    var host = document.createElement("div");
    host.className = "chart-scrubber";
    var win = document.createElement("div");
    win.className = "chart-scrubber-window";
    var leftHandle = document.createElement("div");
    leftHandle.className = "chart-scrubber-handle chart-scrubber-handle-left";
    var rightHandle = document.createElement("div");
    rightHandle.className = "chart-scrubber-handle chart-scrubber-handle-right";
    win.appendChild(leftHandle);
    win.appendChild(rightHandle);
    host.appendChild(win);

    if (wrap.parentNode) {
      wrap.parentNode.insertBefore(host, wrap.nextSibling);
    }

    function indexFromPx(px) {
      var rect = host.getBoundingClientRect();
      if (rect.width <= 0 || total <= 1) return 0;
      var ratio = Math.max(0, Math.min(1, px / rect.width));
      return ratio * (total - 1);
    }

    function paintFromChart() {
      if (total <= 0) {
        win.style.left = "0%";
        win.style.width = "100%";
        return;
      }
      var scale = chart.scales.x;
      var lo = scale && typeof scale.min === "number" ? scale.min : 0;
      var hi = scale && typeof scale.max === "number" ? scale.max : total - 1;
      lo = Math.max(0, Math.min(total - 1, lo));
      hi = Math.max(lo, Math.min(total - 1, hi));
      var span = total > 1 ? total - 1 : 1;
      var pctL = (lo / span) * 100;
      var pctR = (hi / span) * 100;
      win.style.left = pctL + "%";
      win.style.width = Math.max(0.5, pctR - pctL) + "%";
    }

    function applyToChart(lo, hi) {
      lo = Math.max(0, Math.min(total - 1, lo));
      hi = Math.max(lo, Math.min(total - 1, hi));
      chart.options.scales.x.min = lo;
      chart.options.scales.x.max = hi;
      chart.update("none");
    }

    var dragging = null; // { mode: "move" | "left" | "right", offsetIdx, anchorLo, anchorHi }
    function onPointerDown(e) {
      var rect = host.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var idx = indexFromPx(x);
      var scale = chart.scales.x;
      var lo = scale && typeof scale.min === "number" ? scale.min : 0;
      var hi = scale && typeof scale.max === "number" ? scale.max : total - 1;
      var target = e.target;
      var mode;
      if (target === leftHandle) {
        mode = "left";
      } else if (target === rightHandle) {
        mode = "right";
      } else if (target === win) {
        mode = "move";
      } else {
        // Outside the window: recenter the window on the click point.
        var width = hi - lo;
        var newLo = Math.max(0, Math.min(total - 1 - width, idx - width / 2));
        applyToChart(newLo, newLo + width);
        paintFromChart();
        return;
      }
      dragging = {
        mode: mode,
        offsetIdx: idx - lo,
        anchorLo: lo,
        anchorHi: hi,
      };
      try { host.setPointerCapture(e.pointerId); } catch (_) { /* ok */ }
      e.preventDefault();
    }

    function onPointerMove(e) {
      if (!dragging) return;
      var rect = host.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var idx = indexFromPx(x);
      if (dragging.mode === "move") {
        var width = dragging.anchorHi - dragging.anchorLo;
        var newLo = idx - dragging.offsetIdx;
        newLo = Math.max(0, Math.min(total - 1 - width, newLo));
        applyToChart(newLo, newLo + width);
      } else if (dragging.mode === "left") {
        var newLoL = Math.min(idx, dragging.anchorHi);
        applyToChart(newLoL, dragging.anchorHi);
      } else if (dragging.mode === "right") {
        var newHiR = Math.max(idx, dragging.anchorLo);
        applyToChart(dragging.anchorLo, newHiR);
      }
      paintFromChart();
    }

    function onPointerUp(e) {
      if (!dragging) return;
      dragging = null;
      try { host.releasePointerCapture(e.pointerId); } catch (_) { /* ok */ }
    }

    host.addEventListener("pointerdown", onPointerDown);
    host.addEventListener("pointermove", onPointerMove);
    host.addEventListener("pointerup", onPointerUp);
    host.addEventListener("pointercancel", onPointerUp);

    // Throttle paint to animation frames so chart-driven updates feel smooth.
    var paintQueued = false;
    function schedulePaint() {
      if (paintQueued) return;
      paintQueued = true;
      requestAnimationFrame(function () {
        paintQueued = false;
        paintFromChart();
      });
    }

    chart.options.plugins = chart.options.plugins || {};
    return {
      schedulePaint: schedulePaint,
      paintFromChart: paintFromChart,
    };
  }

  // ---- Filter chips (Task E light version) -------------------------------
  //
  // Render two rows of toggle chips above the chart card: one row of
  // engine values, one of formats, derived from the chart's series labels.
  // Toggling a chip writes into `?hidden=` and updates the chart.
  function attachFilterChips(chart, names, filterState) {
    var wrap = chart.canvas.closest(".chart-wrap");
    if (!wrap || wrap.parentNode == null) return;

    var engines = new Set();
    var formats = new Set();
    names.forEach(function (n) {
      var parts = n.split(":");
      if (parts.length > 1) {
        engines.add(parts[0]);
        formats.add(parts.slice(1).join(":"));
      } else {
        formats.add(parts[0]);
      }
    });

    var bar = document.createElement("div");
    bar.className = "filter-bar";

    function chipRow(label, values, kind) {
      if (values.size === 0) return null;
      var row = document.createElement("div");
      row.className = "filter-row";
      var lab = document.createElement("span");
      lab.className = "filter-label";
      lab.textContent = label;
      row.appendChild(lab);
      Array.from(values).sort().forEach(function (v) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "filter-chip";
        chip.dataset.kind = kind;
        chip.dataset.value = v;
        chip.textContent = v;
        chip.setAttribute("aria-pressed", "true");
        chip.addEventListener("click", function () {
          toggleChip(chip);
        });
        row.appendChild(chip);
      });
      return row;
    }

    var er = chipRow("engine", engines, "engine");
    if (er) bar.appendChild(er);
    var fr = chipRow("format", formats, "format");
    if (fr) bar.appendChild(fr);

    if (bar.children.length === 0) return;
    wrap.parentNode.insertBefore(bar, wrap);

    // Initial pressed state from the URL. We also flag chips whose value
    // has every series hidden (e.g. via `?hidden=`) as not-pressed so the
    // chips visually agree with the chart on page load.
    Array.from(bar.querySelectorAll(".filter-chip")).forEach(function (chip) {
      var kind = chip.dataset.kind;
      var value = chip.dataset.value;
      var pressed = true;
      if (kind === "engine" && filterState.engine && filterState.engine !== value) pressed = false;
      if (kind === "format" && filterState.format && filterState.format !== value) pressed = false;
      if (pressed) {
        // If every matching dataset is already hidden (likely via ?hidden=),
        // surface that visually so the chip toggle is meaningful.
        var allHidden = true;
        var anyMatch = false;
        chart.data.datasets.forEach(function (ds, i) {
          var parts = ds.label.split(":");
          var engine = parts.length > 1 ? parts[0] : null;
          var format = parts.length > 1 ? parts.slice(1).join(":") : parts[0];
          var matches = (kind === "engine" && engine === value) ||
                        (kind === "format" && format === value);
          if (matches) {
            anyMatch = true;
            if (chart.isDatasetVisible(i)) allHidden = false;
          }
        });
        if (anyMatch && allHidden) pressed = false;
      }
      chip.setAttribute("aria-pressed", pressed ? "true" : "false");
    });

    function toggleChip(chip) {
      var pressed = chip.getAttribute("aria-pressed") === "true";
      chip.setAttribute("aria-pressed", pressed ? "false" : "true");
      syncSeriesVisibilityFromChips();
    }

    function activeOf(kind) {
      var active = new Set();
      Array.from(bar.querySelectorAll('.filter-chip[data-kind="' + kind + '"]')).forEach(function (c) {
        if (c.getAttribute("aria-pressed") === "true") active.add(c.dataset.value);
      });
      return active;
    }

    function syncSeriesVisibilityFromChips() {
      var activeEngines = activeOf("engine");
      var activeFormats = activeOf("format");
      chart.data.datasets.forEach(function (ds, i) {
        var name = ds.label;
        var parts = name.split(":");
        var engine = parts.length > 1 ? parts[0] : null;
        var format = parts.length > 1 ? parts.slice(1).join(":") : parts[0];
        var visible = true;
        if (engine && activeEngines.size > 0 && !activeEngines.has(engine)) visible = false;
        if (format && activeFormats.size > 0 && !activeFormats.has(format)) visible = false;
        chart.setDatasetVisibility(i, visible);
      });
      chart.update();

      // Persist the inverse (which series are hidden by the user) into the URL.
      var hidden = new Set();
      chart.data.datasets.forEach(function (ds, i) {
        if (!chart.isDatasetVisible(i)) hidden.add(ds.label);
      });
      writeHiddenParam(hidden);
    }
  }

  function init() {
    var canvas = document.getElementById("chart");
    if (!canvas || typeof Chart === "undefined") {
      return;
    }
    var payload = readPayload();
    var commits = payload.commits || [];
    var labels = commits.map(function (c) { return shortSha(c.sha); });
    var series = payload.series || {};
    var names = Object.keys(series).sort();
    var filterState = readUrlParams();
    var datasets = buildDatasets(series, names, filterState);

    var chart = new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: payload.unit || "" },
          },
          x: {
            title: { display: true, text: "commit" },
            min: 0,
            max: commits.length > 0 ? commits.length - 1 : undefined,
          },
        },
        onClick: function (_evt, elements, c) {
          var pts = c.getElementsAtEventForMode(_evt, "nearest", { intersect: false }, true);
          if (!pts || pts.length === 0) return;
          var idx = pts[0].index;
          var commit = commits[idx];
          if (!commit) return;
          var pr = parsePrNumber(commit.message);
          var url = pr
            ? "https://github.com/vortex-data/vortex/pull/" + pr
            : commit.url;
          if (url) {
            window.open(url, "_blank", "noopener,noreferrer");
          }
        },
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            // Top-to-bottom order matches the on-screen order at the
            // hovered x: highest y at the top.
            itemSort: function (a, b) {
              var av = (a.parsed && typeof a.parsed.y === "number") ? a.parsed.y : -Infinity;
              var bv = (b.parsed && typeof b.parsed.y === "number") ? b.parsed.y : -Infinity;
              return bv - av;
            },
            callbacks: {
              title: function (items) {
                if (!items.length) return "";
                var idx = items[0].dataIndex;
                var c = commits[idx] || {};
                var msg = truncate(firstLine(c.message || ""), 80);
                return shortSha(c.sha) + (msg ? "  " + msg : "");
              },
            },
          },
        },
      },
    });

    var scrubber = attachRangeScrubber(chart, commits.length);
    if (scrubber) {
      // Re-paint the scrubber whenever the chart's scale changes, even
      // if the change came from outside the scrubber (toolbar buttons,
      // future zoom plugins, programmatic update).
      var orig = chart.update.bind(chart);
      chart.update = function (mode) {
        var r = orig(mode);
        scrubber.schedulePaint();
        return r;
      };
      scrubber.paintFromChart();
    }
    attachFilterChips(chart, names, filterState);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
