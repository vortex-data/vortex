// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Hydrate Chart.js charts on /chart/:slug and /group/:slug, plus the
// landing-page client-side filter.
//
// Each chart's payload is embedded inline as
//   <script id="chart-data-N" type="application/json">{ChartResponse}</script>
// paired with a <canvas data-chart-index="N"> via the index attribute.
// Construction is deferred until the canvas crosses an IntersectionObserver
// threshold so a 22-chart group doesn't pay for offscreen charts up front.
//
// URL state (n, y, mode, hidden) is the source of truth. Server emits
// scope/Y/mode toolbar links that navigate via plain <a>; client-side
// legend toggles rewrite ?hidden=... via history.replaceState so a
// permalink reproduces the view.
(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // Palette + helpers
  // -----------------------------------------------------------------------
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

  function shortDate(ts) {
    if (typeof ts !== "string") return "";
    // commits.timestamp arrives as either ISO 8601 or DuckDB's `YYYY-MM-DD HH:MM:SS`.
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
      // Pick a friendlier unit when the magnitude warrants it.
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

  // -----------------------------------------------------------------------
  // URL state
  // -----------------------------------------------------------------------
  function parseUrl() {
    var p = new URLSearchParams(window.location.search);
    return {
      n: p.get("n") || "",
      y: p.get("y") === "log" ? "log" : "linear",
      mode: p.get("mode") === "rel" ? "rel" : "abs",
      hidden: parseHiddenParam(p.get("hidden")),
    };
  }

  function parseHiddenParam(s) {
    if (!s) return Object.create(null);
    var out = Object.create(null);
    s.split(",").forEach(function (k) {
      if (k) out[k] = true;
    });
    return out;
  }

  function serializeHidden(set) {
    var keys = Object.keys(set).filter(function (k) { return set[k]; });
    keys.sort();
    return keys.join(",");
  }

  function rewriteHiddenInUrl(set) {
    var p = new URLSearchParams(window.location.search);
    var v = serializeHidden(set);
    if (v) {
      p.set("hidden", v);
    } else {
      p.delete("hidden");
    }
    var qs = p.toString();
    var url = window.location.pathname + (qs ? "?" + qs : "") + window.location.hash;
    window.history.replaceState(null, "", url);
  }

  // -----------------------------------------------------------------------
  // Payload + dataset construction
  // -----------------------------------------------------------------------
  function readPayload(scriptNode) {
    return JSON.parse(scriptNode.textContent);
  }

  function buildDatasets(payload, urlState) {
    var raw = payload.series || {};
    var names = Object.keys(raw).sort();
    var values = names.map(function (name) {
      return Array.isArray(raw[name]) ? raw[name].slice() : [];
    });

    if (urlState.mode === "rel") {
      values = values.map(function (arr) {
        var baseline = null;
        for (var i = 0; i < arr.length; i++) {
          if (arr[i] !== null && arr[i] !== undefined && !Number.isNaN(arr[i])) {
            baseline = arr[i];
            break;
          }
        }
        if (!baseline) return arr.map(function () { return null; });
        return arr.map(function (v) {
          if (v === null || v === undefined || Number.isNaN(v)) return null;
          return (v / baseline) * 100;
        });
      });
    }

    return names.map(function (name, i) {
      return {
        label: name,
        data: values[i],
        rawData: raw[name],
        borderColor: colorFor(i),
        backgroundColor: colorFor(i),
        spanGaps: true,
        tension: 0.1,
        pointRadius: 2,
        pointHoverRadius: 5,
        hidden: !!urlState.hidden[name],
      };
    });
  }

  // -----------------------------------------------------------------------
  // Tooltip
  // -----------------------------------------------------------------------
  function externalTooltipHandler(payload, host) {
    return function (context) {
      var tooltipModel = context.tooltip;
      if (!host) return;
      if (tooltipModel.opacity === 0) {
        host.style.opacity = "0";
        host.style.pointerEvents = "none";
        return;
      }

      var idx = tooltipModel.dataPoints && tooltipModel.dataPoints[0]
        ? tooltipModel.dataPoints[0].dataIndex
        : -1;
      var commit = (payload.commits || [])[idx] || {};
      var unit = payload.unit || "";

      var rows = (tooltipModel.dataPoints || []).map(function (dp) {
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
      var footerHtml = "";
      if (msg || commit.url) {
        footerHtml = '<div class="tt-footer">'
          + (msg ? '<div class="tt-msg">' + escapeHtml(msg) + "</div>" : "")
          + (commit.url
              ? '<a class="tt-link" href="' + escapeHtml(commit.url)
                + '" target="_blank" rel="noopener">view on github →</a>'
              : "")
          + "</div>";
      }

      host.innerHTML = titleHtml + '<div class="tt-rows">' + rows + "</div>" + footerHtml;

      var canvasRect = context.chart.canvas.getBoundingClientRect();
      var hostRect = host.parentNode.getBoundingClientRect();
      var x = canvasRect.left - hostRect.left + tooltipModel.caretX;
      var y = canvasRect.top - hostRect.top + tooltipModel.caretY;
      host.style.opacity = "1";
      host.style.pointerEvents = "auto";
      host.style.left = x + "px";
      host.style.top = y + "px";
    };
  }

  // -----------------------------------------------------------------------
  // Single-chart construction
  // -----------------------------------------------------------------------
  function constructChart(card, urlState) {
    var idx = card.getAttribute("data-chart-index");
    var script = document.getElementById("chart-data-" + idx);
    var canvas = card.querySelector('canvas[data-chart-index="' + idx + '"]');
    if (!script || !canvas || typeof Chart === "undefined") return null;
    if (canvas.__bench_chart) return canvas.__bench_chart;

    var payload;
    try {
      payload = readPayload(script);
    } catch (e) {
      return null;
    }

    var labels = (payload.commits || []).map(function (c) { return shortSha(c.sha); });
    var datasets = buildDatasets(payload, urlState);
    var unit = payload.unit || "";
    var host = card.querySelector(".chart-tooltip-host");

    var yTitle = unit;
    if (urlState.mode === "rel") yTitle = "% of baseline";
    // Mobile gets the legend above the chart so the chart doesn't get pushed
    // off-screen by a tall legend on narrow viewports.
    var legendPosition = (window.matchMedia
      && window.matchMedia("(max-width: 768px)").matches) ? "top" : "bottom";

    var chart = new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: {
            type: urlState.y === "log" ? "logarithmic" : "linear",
            beginAtZero: urlState.y !== "log" && urlState.mode !== "rel",
            title: { display: !!yTitle, text: yTitle },
          },
          x: { title: { display: false } },
        },
        plugins: {
          legend: {
            position: legendPosition,
            onClick: function (e, item, legend) {
              // Default toggle behaviour, then mirror into URL.
              var ci = legend.chart;
              var meta = ci.getDatasetMeta(item.datasetIndex);
              meta.hidden = meta.hidden === null ? !ci.data.datasets[item.datasetIndex].hidden : null;
              ci.update();
              var hiddenSet = parseHiddenParam(new URLSearchParams(window.location.search).get("hidden"));
              var label = item.text;
              if (meta.hidden) hiddenSet[label] = true; else delete hiddenSet[label];
              rewriteHiddenInUrl(hiddenSet);
            },
          },
          tooltip: {
            enabled: false,
            external: externalTooltipHandler(payload, host),
          },
        },
      },
    });
    canvas.__bench_chart = chart;
    return chart;
  }

  // -----------------------------------------------------------------------
  // Page wiring
  // -----------------------------------------------------------------------
  function initCharts() {
    var urlState = parseUrl();
    var cards = document.querySelectorAll(".chart-card[data-chart-index]");
    if (!cards.length) return;

    if (typeof IntersectionObserver === "undefined") {
      cards.forEach(function (card) { constructChart(card, urlState); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            constructChart(entry.target, urlState);
            io.unobserve(entry.target);
          }
        });
      }, { rootMargin: "150px 0px" });
      cards.forEach(function (card) { io.observe(card); });
    }

    // Tap-elsewhere closes any open external tooltip.
    document.addEventListener("click", function (e) {
      var hosts = document.querySelectorAll(".chart-tooltip-host");
      hosts.forEach(function (host) {
        if (!host.contains(e.target)) {
          host.style.opacity = "0";
          host.style.pointerEvents = "none";
        }
      });
    });

    initSlider();
  }

  function initSlider() {
    var slider = document.getElementById("scope-slider");
    var label = document.getElementById("scope-slider-label");
    if (!slider) return;
    slider.addEventListener("input", function () {
      if (label) label.textContent = slider.value;
    });
    slider.addEventListener("change", function () {
      var p = new URLSearchParams(window.location.search);
      p.set("n", slider.value);
      window.location.search = p.toString();
    });
  }

  function initLandingFilter() {
    var input = document.getElementById("group-search");
    if (!input) return;
    var groups = document.querySelectorAll("section.group[data-group-name]");
    input.addEventListener("input", function () {
      var q = input.value.toLowerCase();
      groups.forEach(function (g) {
        var name = (g.getAttribute("data-group-name") || "").toLowerCase();
        g.style.display = !q || name.indexOf(q) !== -1 ? "" : "none";
      });
    });
  }

  function init() {
    initLandingFilter();
    initCharts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
