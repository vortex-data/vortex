// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Hydrate Chart.js charts on /, /chart/:slug, and /group/:slug, plus the
// landing-page client-side filter.
//
// Each chart's initial payload is embedded inline as
//   <script id="chart-data-N" type="application/json">{ChartResponse}</script>
// paired with a <canvas data-chart-index="N"> via the index attribute. The
// chart-card carries `data-chart-slug` so the toolbar can refetch a single
// card from `/api/chart/{slug}?n=...` without a page reload.
//
// URL state (n, y, mode, hidden) is the source of truth and the URL stays in
// sync via `history.replaceState`. Toolbar clicks are handled in JS:
//   - `n` → refetch every chart on the page, swap data, chart.update("none").
//   - `y` → swap `chart.options.scales.y` in place; no fetch.
//   - `mode` → recompute datasets client-side; no fetch.
//   - legend toggle → mirror into `?hidden=...` like before.
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

  // `|` cannot appear in our series labels (which are
  // "engine:format"-shaped today), unlike `,`/`/` which could plausibly
  // sneak in via dataset variants. URLSearchParams handles `|` as-is.
  var HIDDEN_DELIM = "|";

  function parseHiddenParam(s) {
    if (!s) return Object.create(null);
    var out = Object.create(null);
    s.split(HIDDEN_DELIM).forEach(function (k) {
      if (k) out[k] = true;
    });
    return out;
  }

  function serializeHidden(set) {
    var keys = Object.keys(set).filter(function (k) { return set[k]; });
    keys.sort();
    return keys.join(HIDDEN_DELIM);
  }

  // Default value the server treats as "use the route's default scope". When
  // the URL has no `n` we want to leave the param off so the server can
  // re-pick its own default (50 on `/`, 100 on `/chart` and `/group`).
  function applyUrlState(state) {
    var p = new URLSearchParams(window.location.search);
    if (state.n) p.set("n", state.n); else p.delete("n");
    if (state.y && state.y !== "linear") p.set("y", state.y); else p.delete("y");
    if (state.mode && state.mode !== "abs") p.set("mode", state.mode); else p.delete("mode");
    var h = serializeHidden(state.hidden || {});
    if (h) p.set("hidden", h); else p.delete("hidden");
    var qs = p.toString();
    var url = window.location.pathname + (qs ? "?" + qs : "") + window.location.hash;
    window.history.replaceState(null, "", url);
  }

  function rewriteHiddenInUrl(set) {
    var state = parseUrl();
    state.hidden = set;
    applyUrlState(state);
  }

  // -----------------------------------------------------------------------
  // Payload + dataset construction
  // -----------------------------------------------------------------------
  function readInlinePayload(idx) {
    var script = document.getElementById("chart-data-" + idx);
    if (!script) return null;
    try {
      return JSON.parse(script.textContent);
    } catch (e) {
      return null;
    }
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

  function yAxisTitle(payload, urlState) {
    return urlState.mode === "rel" ? "% of baseline" : (payload.unit || "");
  }

  // -----------------------------------------------------------------------
  // Tooltip
  // -----------------------------------------------------------------------
  function externalTooltipHandler(canvas, host) {
    return function (context) {
      var tooltipModel = context.tooltip;
      if (!host) return;
      if (tooltipModel.opacity === 0) {
        host.style.opacity = "0";
        host.style.pointerEvents = "none";
        return;
      }

      // Always read the current payload from the canvas: a refetch may have
      // replaced it under us since this handler was installed.
      var payload = canvas.__bench_payload || { commits: [], unit: "" };

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
  // Single-chart construction + in-place rebuild
  // -----------------------------------------------------------------------
  function constructChart(card, urlState) {
    var idx = card.getAttribute("data-chart-index");
    var canvas = card.querySelector('canvas[data-chart-index="' + idx + '"]');
    if (!canvas || typeof Chart === "undefined") return null;
    if (canvas.__bench_chart) return canvas.__bench_chart;

    // Prefer a payload that arrived via fetch (refetch landed before the
    // canvas scrolled into view); else fall back to the inline JSON.
    var payload = canvas.__bench_payload || readInlinePayload(idx);
    if (!payload) return null;
    canvas.__bench_payload = payload;

    var labels = (payload.commits || []).map(function (c) { return shortSha(c.sha); });
    var datasets = buildDatasets(payload, urlState);
    var host = card.querySelector(".chart-tooltip-host");

    // Mobile gets the legend above the chart so the chart doesn't get pushed
    // off-screen by a tall legend on narrow viewports.
    var legendPosition = (window.matchMedia
      && window.matchMedia("(max-width: 768px)").matches) ? "top" : "bottom";

    var yTitle = yAxisTitle(payload, urlState);
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
            external: externalTooltipHandler(canvas, host),
          },
        },
      },
    });
    canvas.__bench_chart = chart;
    return chart;
  }

  // Re-skin a chart from its current payload + url state. No fetch.
  function rebuildChart(card, urlState) {
    var idx = card.getAttribute("data-chart-index");
    var canvas = card.querySelector('canvas[data-chart-index="' + idx + '"]');
    if (!canvas) return;
    var chart = canvas.__bench_chart;
    var payload = canvas.__bench_payload;
    if (!chart || !payload) return;

    chart.data.labels = (payload.commits || []).map(function (c) { return shortSha(c.sha); });
    chart.data.datasets = buildDatasets(payload, urlState);
    chart.options.scales.y.type = urlState.y === "log" ? "logarithmic" : "linear";
    chart.options.scales.y.beginAtZero = urlState.y !== "log" && urlState.mode !== "rel";
    var t = yAxisTitle(payload, urlState);
    chart.options.scales.y.title.display = !!t;
    chart.options.scales.y.title.text = t;
    chart.update("none");
  }

  // -----------------------------------------------------------------------
  // Loading + error overlays per card
  // -----------------------------------------------------------------------
  function setCardLoading(card, on) {
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
  // Refetching when the commit window changes
  // -----------------------------------------------------------------------
  function refetchAll(urlState) {
    var cards = document.querySelectorAll(".chart-card[data-chart-slug]");
    if (!cards.length) return Promise.resolve();
    var n = urlState.n || "";
    var qs = n ? "?n=" + encodeURIComponent(n) : "";

    var jobs = [];
    cards.forEach(function (card) {
      var slug = card.getAttribute("data-chart-slug");
      var canvas = card.querySelector("canvas");
      if (!slug || !canvas) return;
      var prevPayload = canvas.__bench_payload;
      setCardLoading(card, true);
      var p = fetch("/api/chart/" + encodeURIComponent(slug) + qs, {
        headers: { "accept": "application/json" },
      })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (payload) {
          canvas.__bench_payload = payload;
          if (canvas.__bench_chart) {
            rebuildChart(card, urlState);
          }
          // Else: chart not constructed yet; the IntersectionObserver path
          // will read the new payload when the canvas eventually scrolls in.
        })
        .catch(function (err) {
          if (prevPayload) canvas.__bench_payload = prevPayload;
          showCardError(card, "failed to load: " + (err && err.message ? err.message : err));
        })
        .then(function () { setCardLoading(card, false); });
      jobs.push(p);
    });
    return Promise.all(jobs);
  }

  // -----------------------------------------------------------------------
  // Toolbar wiring
  // -----------------------------------------------------------------------
  function updateToolbarActive(group, value) {
    var attr = "data-" + group;
    var btns = document.querySelectorAll(".toolbar-btn[" + attr + "]");
    btns.forEach(function (b) {
      var match = b.getAttribute(attr) === value;
      b.classList.toggle("toolbar-btn--active", match);
    });
  }

  function updateSubtitle(urlState, defaultN) {
    var sub = document.querySelector(".page-header .subtitle");
    if (!sub) return;
    var base = sub.getAttribute("data-base") || sub.textContent.split(" · ")[0];
    sub.setAttribute("data-base", base);
    var bits = [base];
    var n = urlState.n || String(defaultN || "");
    if (n === "all") bits.push("all commits");
    else if (n) bits.push("last " + n + " commits");
    if (urlState.y === "log") bits.push("log");
    if (urlState.mode === "rel") bits.push("rel");
    sub.textContent = bits.join(" · ");
  }

  function updateSliderUi(value) {
    var slider = document.getElementById("scope-slider");
    var label = document.getElementById("scope-slider-label");
    if (slider && /^\d+$/.test(value)) slider.value = value;
    if (label) label.textContent = value;
  }

  function applyScope(value, defaultN) {
    var state = parseUrl();
    state.n = value;
    applyUrlState(state);
    updateToolbarActive("scope", value);
    updateSliderUi(value);
    updateSubtitle(state, defaultN);
    rewriteCardLinks();
    refetchAll(state);
  }

  function applyY(value, defaultN) {
    var state = parseUrl();
    state.y = value;
    applyUrlState(state);
    updateToolbarActive("y", value);
    updateSubtitle(state, defaultN);
    rewriteCardLinks();
    document.querySelectorAll(".chart-card[data-chart-index]").forEach(function (card) {
      rebuildChart(card, state);
    });
  }

  function applyMode(value, defaultN) {
    var state = parseUrl();
    state.mode = value;
    applyUrlState(state);
    updateToolbarActive("mode", value);
    updateSubtitle(state, defaultN);
    rewriteCardLinks();
    document.querySelectorAll(".chart-card[data-chart-index]").forEach(function (card) {
      rebuildChart(card, state);
    });
  }

  // The chart-card title links carry the toolbar state in their query string
  // so a click out to a permalink preserves the current view. After every
  // toolbar change we rewrite them.
  function rewriteCardLinks() {
    var p = new URLSearchParams(window.location.search);
    var qs = p.toString();
    var suffix = qs ? "?" + qs : "";
    document.querySelectorAll(".chart-card-title a[data-permalink]").forEach(function (a) {
      a.setAttribute("href", a.getAttribute("data-permalink") + suffix);
    });
  }

  function initToolbar(defaultN) {
    var toolbar = document.querySelector(".toolbar");
    if (!toolbar) return;

    toolbar.addEventListener("click", function (e) {
      var btn = e.target.closest(".toolbar-btn");
      if (!btn || !toolbar.contains(btn)) return;
      // Hijack the link; we update state in place.
      e.preventDefault();
      if (btn.hasAttribute("data-scope")) {
        applyScope(btn.getAttribute("data-scope"), defaultN);
      } else if (btn.hasAttribute("data-y")) {
        applyY(btn.getAttribute("data-y"), defaultN);
      } else if (btn.hasAttribute("data-mode")) {
        applyMode(btn.getAttribute("data-mode"), defaultN);
      }
    });

    var slider = document.getElementById("scope-slider");
    var label = document.getElementById("scope-slider-label");
    if (slider) {
      slider.addEventListener("input", function () {
        if (label) label.textContent = slider.value;
      });
      slider.addEventListener("change", function () {
        applyScope(slider.value, defaultN);
      });
    }
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
            constructChart(entry.target, parseUrl());
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
    var main = document.querySelector("main");
    var defaultN = main && main.getAttribute("data-default-n");
    initLandingFilter();
    initCharts();
    initToolbar(defaultN);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
