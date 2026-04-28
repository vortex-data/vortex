// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Hydrate Chart.js charts on /, /chart/:slug, and /group/:slug, plus the
// lazy-fetch-on-toggle behaviour for closed `<details>` groups.
//
// Per-chart UX:
//   - Each `.chart-card` carries `data-chart-slug`. The card *owns* its own
//     toolbar (`.toolbar--card`) — there is no page-level toolbar.
//   - Each chart fetches up to 1000 commits once. The toolbar's slider sets
//     `chart.options.scales.x.min/max` to reveal a window of that fetched
//     slice; we never refetch on a scope change.
//   - The slider is throttled to ~16ms (one frame at 60fps) per v2's
//     `CONFIG.ZOOM_THROTTLE_DELAY` so dragging the slider feels continuous.
//   - Mouse wheel pans horizontally (chartjs-plugin-zoom does not expose
//     pan-on-wheel, so a manual `wheel` listener calls `chart.pan(...)`).
//   - Drag-pan + drag-rectangle-zoom are wired through the plugin.
//   - A custom inline plugin draws a vertical crosshair at the hovered
//     commit; the external tooltip is offset and `pointer-events: none`
//     to fix the flicker described in the per-chart UX rebuild brief.
(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // Constants — match v2 (`origin/ct/vfvb:benchmarks-website/config.js`).
  // -----------------------------------------------------------------------
  var ZOOM_THROTTLE_MS = 16; // one frame at ~60fps for slider drag
  var FETCH_N = 1000; // matches `PER_CHART_FETCH_N` server-side
  var DEFAULT_VISIBLE = 100; // initial visible window (last 100 of fetched)

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

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNumber(v, unit) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (unit === "ns") {
      var abs = Math.abs(v);
      if (abs >= 1e9) return (v / 1e9).toFixed(2) + " s";
      if (abs >= 1e6) return (v / 1e6).toFixed(2) + " ms";
      if (abs >= 1e3) return (v / 1e3).toFixed(2) + " µs";
      return v.toFixed(0) + " ns";
    }
    if (unit === "bytes") {
      var a = Math.abs(v);
      if (a >= 1024 * 1024 * 1024) return (v / (1024 * 1024 * 1024)).toFixed(2) + " GiB";
      if (a >= 1024 * 1024) return (v / (1024 * 1024)).toFixed(2) + " MiB";
      if (a >= 1024) return (v / 1024).toFixed(2) + " KiB";
      return v.toFixed(0) + " B";
    }
    return v.toString();
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
  // Flicker fix: the tooltip host is **always** `pointer-events: none`. The
  // previous implementation flipped it to `auto` when visible; the cursor
  // would land on the tooltip, fire mouseout on the canvas, the tooltip
  // would hide, the cursor would re-enter the canvas, and the cycle would
  // repeat at event-loop frequency. The cost of `pointer-events: none` is
  // that the github-link in the tooltip footer is no longer clickable, but
  // the chart-card title already links to the permalink.
  // -----------------------------------------------------------------------
  function externalTooltipHandler(canvas, host) {
    return function (context) {
      var tt = context.tooltip;
      if (!host) return;
      if (tt.opacity === 0) {
        host.style.opacity = "0";
        return;
      }

      var payload = canvas.__bench_payload || { commits: [], unit: "" };
      var idx = tt.dataPoints && tt.dataPoints[0]
        ? tt.dataPoints[0].dataIndex
        : -1;
      var commit = (payload.commits || [])[idx] || {};
      var unit = payload.unit || "";

      var rows = (tt.dataPoints || []).map(function (dp) {
        var ds = dp.dataset || {};
        var raw = (ds.rawData || [])[idx];
        var prevIdx = idx - 1;
        var prevRaw = null;
        while (prevIdx >= 0) {
          var pv = (ds.rawData || [])[prevIdx];
          if (pv !== null && pv !== undefined && !Number.isNaN(pv)) { prevRaw = pv; break; }
          prevIdx--;
        }
        var deltaHtml = "";
        if (prevRaw !== null && raw !== null && raw !== undefined && prevRaw !== 0) {
          var pct = ((raw - prevRaw) / prevRaw) * 100;
          var cls = pct > 0 ? "tt-delta tt-delta--worse"
                  : pct < 0 ? "tt-delta tt-delta--better" : "tt-delta";
          var sign = pct > 0 ? "+" : "";
          deltaHtml = '<span class="' + cls + '">' + sign + pct.toFixed(1) + "%</span>";
        }
        return '<div class="tt-row">'
          + '<span class="tt-swatch" style="background:' + ds.borderColor + '"></span>'
          + '<span class="tt-label">' + escapeHtml(ds.label) + '</span>'
          + '<span class="tt-value">' + escapeHtml(formatNumber(raw, unit)) + '</span>'
          + deltaHtml
          + "</div>";
      }).join("");

      var titleHtml = '<div class="tt-title">'
        + escapeHtml(shortSha(commit.sha)) + ' · '
        + escapeHtml(shortDate(commit.timestamp))
        + "</div>";

      var msg = commit.message ? truncate(commit.message, 120) : "";
      var footerHtml = (msg || commit.url)
        ? '<div class="tt-footer">'
          + (msg ? '<div class="tt-msg">' + escapeHtml(msg) + "</div>" : "")
          + (commit.url ? '<div class="tt-msg">'
              + escapeHtml(commit.url.replace(/^https?:\/\//, "")) + "</div>" : "")
          + "</div>"
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

  function buildDatasets(payload) {
    var raw = payload.series || {};
    var names = Object.keys(raw).sort();
    var values = names.map(function (name) {
      return Array.isArray(raw[name]) ? raw[name].slice() : [];
    });

    return names.map(function (name, i) {
      return {
        label: name,
        data: values[i],
        rawData: raw[name],
        borderColor: colorFor(i),
        backgroundColor: colorFor(i) + "20",
        borderWidth: 1.5,
        spanGaps: true,
        tension: 0,
        pointRadius: 2,
        pointHoverRadius: 5,
        pointHitRadius: 8,
        pointStyle: "cross",
      };
    });
  }

  // -----------------------------------------------------------------------
  // Per-card construction. State lives on the canvas:
  //   canvas.__bench_chart   — Chart.js instance
  //   canvas.__bench_payload — last-fetched ChartResponse
  //   canvas.__bench_state   — { y, scope } (per-chart toolbar state)
  // -----------------------------------------------------------------------
  function constructChart(card) {
    var idx = card.getAttribute("data-chart-index");
    var canvas = card.querySelector('canvas[data-chart-index="' + idx + '"]');
    if (!canvas || typeof Chart === "undefined") return null;
    if (canvas.__bench_chart) return canvas.__bench_chart;

    var payload = canvas.__bench_payload || readInlinePayload(idx);
    if (!payload) return null;
    canvas.__bench_payload = payload;

    var state = canvas.__bench_state || { y: "linear", scope: DEFAULT_VISIBLE };
    canvas.__bench_state = state;

    var labels = (payload.commits || []).map(function (c) { return shortSha(c.sha); });
    var datasets = buildDatasets(payload);
    var host = card.querySelector(".chart-tooltip-host");
    var range = visibleRange(labels.length, state.scope);
    var legendPosition = (window.matchMedia
      && window.matchMedia("(max-width: 768px)").matches) ? "top" : "bottom";

    var chart = new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      plugins: [crosshairPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        // Snap-to-x-index, no vertical-intersection requirement: a stable
        // hover anywhere over the chart, with the crosshair plugin painting
        // the column. Combined with `pointer-events: none` on the tooltip
        // host, this is the flicker fix.
        interaction: { mode: "index", intersect: false, axis: "x" },
        scales: {
          y: {
            type: state.y === "log" ? "logarithmic" : "linear",
            beginAtZero: state.y !== "log",
            title: { display: true, text: payload.unit || "" },
          },
          x: {
            min: range.min,
            max: range.max,
            title: { display: false },
          },
        },
        plugins: {
          legend: { position: legendPosition },
          tooltip: {
            enabled: false,
            external: externalTooltipHandler(canvas, host),
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
            },
            pan: {
              enabled: true,
              mode: "x",
              modifierKey: null,
            },
            limits: {
              x: { min: 0, max: Math.max(0, labels.length - 1), minRange: 4 },
            },
          },
        },
      },
    });

    canvas.__bench_chart = chart;
    attachWheelPan(canvas, chart);
    return chart;
  }

  // Wheel = horizontal pan. Chart.js zoom plugin doesn't support wheel-pan
  // out of the box (wheel is always zoom in its config), so we attach a
  // `wheel` listener that translates `deltaY`/`deltaX` into `chart.pan`.
  function attachWheelPan(canvas, chart) {
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
    }, { passive: false });
  }

  // -----------------------------------------------------------------------
  // Recompute helpers driven by the per-chart toolbar.
  // -----------------------------------------------------------------------
  function visibleRange(commitCount, scope) {
    if (commitCount <= 0) return { min: undefined, max: undefined };
    var maxIdx = commitCount - 1;
    if (scope === "all" || !Number.isFinite(scope) || scope <= 0 || scope >= commitCount) {
      return { min: 0, max: maxIdx };
    }
    return { min: Math.max(0, maxIdx - (scope - 1)), max: maxIdx };
  }

  function applyScope(card, scopeValue) {
    var canvas = card.querySelector("canvas");
    var chart = canvas && canvas.__bench_chart;
    if (!chart) return;
    var commits = chart.data.labels.length;
    var scope = scopeValue === "all" ? "all" : parseInt(scopeValue, 10);
    canvas.__bench_state.scope = scope;
    var range = visibleRange(commits, scope);
    chart.options.scales.x.min = range.min;
    chart.options.scales.x.max = range.max;
    chart.update("none");
    syncToolbarUi(card, "scope", String(scopeValue));
  }

  function applyY(card, yValue) {
    var canvas = card.querySelector("canvas");
    var chart = canvas && canvas.__bench_chart;
    if (!chart) return;
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
  // Lazy fetch on `<details>` toggle for closed-by-default groups.
  // -----------------------------------------------------------------------
  function fetchAndConstruct(card) {
    var canvas = card.querySelector("canvas");
    if (!canvas) return Promise.resolve();
    if (canvas.__bench_chart) return Promise.resolve();
    if (canvas.__bench_payload) {
      constructChart(card);
      bindToolbar(card);
      return Promise.resolve();
    }
    var slug = card.getAttribute("data-chart-slug");
    if (!slug) return Promise.resolve();
    showCardLoading(card, true);
    return fetch("/api/chart/" + encodeURIComponent(slug) + "?n=" + FETCH_N, {
      headers: { "accept": "application/json" },
    })
      .then(function (r) {
        if (r.status === 404) return null; // empty chart, leave the shell
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (payload) {
        if (!payload) return;
        canvas.__bench_payload = payload;
        constructChart(card);
        bindToolbar(card);
      })
      .catch(function (err) {
        showCardError(card, "failed to load: " + (err && err.message ? err.message : err));
      })
      .then(function () { showCardLoading(card, false); });
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

  function hydrateOpenGroup(disclosure) {
    if (!disclosure || !disclosure.open) return;
    var group = disclosure.closest(".group-details");
    if (!group) return;
    group.querySelectorAll(".chart-card[data-chart-slug]").forEach(function (card) {
      fetchAndConstruct(card);
    });
  }

  function setAllGroups(open) {
    document.querySelectorAll("details.group-disclosure").forEach(function (disclosure) {
      var wasOpen = disclosure.open;
      disclosure.open = open;
      if (open && wasOpen) hydrateOpenGroup(disclosure);
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
    });
  }

  function init() {
    initHeaderControls();
    initDetailsToggle();
    initOpenCharts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
