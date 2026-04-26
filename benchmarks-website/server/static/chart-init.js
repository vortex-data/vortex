// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Hydrate the Chart.js line chart on /chart/:slug.
//
// The server embeds the chart payload as a JSON <script id="chart-data">
// block matching the /api/chart/:slug shape:
//   { display_name, unit, commits: [{sha,timestamp,message,url}], series: { name: [Number|null] } }
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

  function readPayload() {
    var node = document.getElementById("chart-data");
    if (!node) {
      throw new Error("missing #chart-data element");
    }
    return JSON.parse(node.textContent);
  }

  function buildDatasets(series) {
    var names = Object.keys(series).sort();
    return names.map(function (name, i) {
      return {
        label: name,
        data: series[name],
        borderColor: colorFor(i),
        backgroundColor: colorFor(i),
        spanGaps: true,
        tension: 0.1,
        pointRadius: 3,
        pointHoverRadius: 5,
      };
    });
  }

  function init() {
    var canvas = document.getElementById("chart");
    if (!canvas || typeof Chart === "undefined") {
      return;
    }
    var payload = readPayload();
    var labels = (payload.commits || []).map(function (c) {
      return shortSha(c.sha);
    });
    var datasets = buildDatasets(payload.series || {});
    new Chart(canvas, {
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
          x: { title: { display: true, text: "commit" } },
        },
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            callbacks: {
              title: function (items) {
                if (!items.length) return "";
                var idx = items[0].dataIndex;
                var c = (payload.commits || [])[idx] || {};
                var msg = c.message ? " — " + c.message : "";
                return shortSha(c.sha) + msg;
              },
            },
          },
        },
      },
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
