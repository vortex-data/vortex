"use strict";

// Configuration constants
const CONFIG = {
  MOBILE_BREAKPOINT: 768,
  MOBILE_MAX_DATA_POINTS: 100,
  DEFAULT_VISIBLE_COMMITS: 50,
  DEBOUNCE_DELAY: 50,
  MOBILE_DEBOUNCE_DELAY: 200,
  THROTTLE_SCROLL: 100,
  SEARCH_DEBOUNCE: 300,
  CHART_OBSERVER_MARGIN: "50px",
  SCROLL_OFFSET_PADDING: 20,
  ZOOM_SPEED: 0.1,
  MIN_VISIBLE_COMMITS: 10,
  COMPRESS_THROUGHPUT_MAX: 1024,
  DECOMPRESS_THROUGHPUT_MAX: 8192,
  ANIMATION_DURATION: 1000,
  LINK_FEEDBACK_DURATION: 2000,
  BACK_TO_TOP_THRESHOLD: 200,
  SCROLL_ACTIVE_THRESHOLD: 100,
  URL_INIT_DELAY: 100,
  RESIZE_DEBOUNCE: 250,
};

// Color mappings for series
const SERIES_COLOR_MAP = {
  "datafusion:arrow": "#7a27b1",
  "datafusion:parquet": "#ef7f1d",
  "datafusion:vortex": "#19a508",
  "duckdb:parquet": "#985113",
  "duckdb:vortex": "#0e5e04",
  "duckdb:duckdb": "#87752e",
};

// Brand colors
const VORTEX_COLORS = {
  primary: "#5971FD", // Vortex Blue
  accent: "#CEE562", // Vortex Green
  pink: "#EEB3E1", // Vortex Pink
  black: "#101010", // Vortex Black
  gray: "#666666", // Secondary gray
};

// Fallback color palette
const FALLBACK_PALETTE = [
  VORTEX_COLORS.primary,
  VORTEX_COLORS.accent,
  VORTEX_COLORS.pink,
  "#FF8C42", // Orange
  "#B8336A", // Deep pink
  "#726DA8", // Purple
  "#2D936C", // Teal
  "#E9B44C", // Gold
];

// Benchmark descriptions
const BENCHMARK_DESCRIPTIONS = {
  "Random Access":
    "Tests performance of selecting arbitrary row indices from a file on NVMe storage",
  Compression:
    "Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet files (with zstd page compression)",
  "Compression Size":
    "Compares compressed file sizes and compression ratios across different encoding strategies, helping evaluate the space efficiency trade-offs between Vortex and Parquet formats",
  "TPC-H (NVMe)":
    "TPC-H benchmark queries executed on local NVMe storage, testing analytical query performance",
  "TPC-H (S3)":
    "TPC-H benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance and the impact of network latency on analytical workloads",
  Clickbench:
    "ClickHouse's analytical benchmark suite testing real-world query patterns on web analytics data, run against NVMe storage",
};

// Category tags mapping
const CATEGORY_TAGS = {
  "Random Access": ["Read/Write"],
  Compression: ["Read/Write"],
  "Compression Size": ["Read/Write"],
  Clickbench: ["Queries (NVMe)"],
  "TPC-H (NVMe) (SF=1)": ["Queries (NVMe)", "TPC-H (SF=1)"],
  "TPC-H (S3) (SF=1)": ["Queries (S3)", "TPC-H (SF=1)"],
  "TPC-H (NVMe) (SF=10)": ["Queries (NVMe)", "TPC-H (SF=10)"],
  "TPC-H (S3) (SF=10)": ["Queries (S3)", "TPC-H (SF=10)"],
  "TPC-H (NVMe) (SF=100)": ["Queries (NVMe)", "TPC-H (SF=100)"],
  "TPC-H (S3) (SF=100)": ["Queries (S3)", "TPC-H (SF=100)"],
  "TPC-H (NVMe) (SF=1000)": ["Queries (NVMe)", "TPC-H (SF=1000)"],
  "TPC-H (S3) (SF=1000)": ["Queries (S3)", "TPC-H (SF=1000)"],
};

// Scale factor descriptions
const SCALE_FACTOR_DESCRIPTIONS = {
  1: "SF=1 (~1GB of data)",
  10: "SF=10 (~10GB of data)",
  100: "SF=100 (~100GB of data)",
  1000: "SF=1000 (~1TB of data)",
};

// Query name transformations
const QUERY_NAME_MAP = {
  "VORTEX:RAW SIZE": "VORTEX COMPRESSION RATIO",
  "VORTEX:PARQUET-ZSTD SIZE": "VORTEX:PARQUET-ZSTD SIZE RATIO",
};

// Engine labels
const ENGINE_LABELS = {
  all: "All",
  duckdb: "DuckDB",
  datafusion: "DataFusion",
  vortex: "Vortex",
  parquet: "Parquet",
};

// Group definitions
const BENCHMARK_GROUPS = [
  "Random Access",
  "Compression",
  "Compression Size",
  "Clickbench",
  "TPC-H (NVMe) (SF=1)",
  "TPC-H (S3) (SF=1)",
  "TPC-H (NVMe) (SF=10)",
  "TPC-H (S3) (SF=10)",
  "TPC-H (NVMe) (SF=100)",
  "TPC-H (S3) (SF=100)",
  "TPC-H (NVMe) (SF=1000)",
  "TPC-H (S3) (SF=1000)",
];

// Main module
window.initAndRender = (function () {
  // State management
  const state = {
    currentView: "grid",
    expandedSections: new Set(),
    activeCategory: "all",
    activeTag: "all",
    activeEngine: "all",
    searchTerm: "",
    charts: [],
    chartInstances: new Map(),
    pendingZoomUpdates: new Map(),
    lastWindowWidth: window.innerWidth,
    isResizing: false,
  };

  // DOM element cache
  const domElements = {};
  let chartObserver = null;
  let debouncedSyncZoom = null;

  // Utility functions
  const utils = {
    throttle(func, limit) {
      let inThrottle;
      return function (...args) {
        if (!inThrottle) {
          func.apply(this, args);
          inThrottle = true;
          setTimeout(() => (inThrottle = false), limit);
        }
      };
    },

    debounce(func, wait) {
      let timeout;
      return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
      };
    },

    isMobile() {
      return window.innerWidth <= CONFIG.MOBILE_BREAKPOINT;
    },

    getDebounceDelay() {
      return utils.isMobile()
        ? CONFIG.MOBILE_DEBOUNCE_DELAY
        : CONFIG.DEBOUNCE_DELAY;
    },

    stringToColor(str) {
      if (SERIES_COLOR_MAP[str]) {
        return SERIES_COLOR_MAP[str];
      }

      const hash = new Hashes.MD5().hex(str);
      const index = parseInt(hash.slice(0, 2), 16) % FALLBACK_PALETTE.length;
      return FALLBACK_PALETTE[index];
    },

    batchDOMUpdates(updates) {
      requestAnimationFrame(() => {
        updates.forEach((update) => update());
      });
    },
  };

  // Data processing module
  const dataProcessor = {
    parseCommits(commitMetadata) {
      const commits = [];
      Object.values(commitMetadata)
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
        .forEach((commit, index) => {
          commit.sortedIndex = index;
          commits.push(commit);
        });
      return commits;
    },

    createMissingCommit(commitId) {
      return {
        author: { email: "daniel.zidan.king@gmail.com", name: "Dan King" },
        committer: { email: "noreply@github.com", name: "GitHub" },
        id: commitId,
        message: "!! This commit is missing from commits.json !!",
        timestamp: "1970-01-01T00:00:00Z",
        tree_id: null,
        url: `https://github.com/vortex-data/vortex/commit/${commitId}`,
      };
    },

    determineGroupId(benchmark) {
      const { name, dataset, storage } = benchmark;

      if (dataset?.tpch) {
        const scaleFactor = dataset.tpch.scale_factor;
        const isNvme = storage === undefined || storage === "nvme";
        return this.getTpchGroupId(scaleFactor, isNvme);
      }

      if (dataset?.clickbench) return "Clickbench";
      if (name.startsWith("random-access/")) return "Random Access";
      if (name.includes("compress time/")) return "Compression";
      if (name.startsWith("vortex size/")) return "Compression Size";
      if (
        name.startsWith("vortex:raw size/") ||
        name.startsWith("vortex:parquet-zstd size/")
      ) {
        return "Compression Size";
      }
      if (name.startsWith("tpch_q")) {
        const isNvme = storage === undefined || storage === "nvme";
        return isNvme ? "TPC-H (NVMe) (SF=1)" : "TPC-H (S3) (SF=1)";
      }
      if (name.startsWith("clickbench")) return "Clickbench";

      return null;
    },

    getTpchGroupId(scaleFactor, isNvme) {
      const sf = Number(scaleFactor);
      const storage = isNvme ? "NVMe" : "S3";

      switch (sf) {
        case 1:
          return `TPC-H (${storage}) (SF=1)`;
        case 10:
          return `TPC-H (${storage}) (SF=10)`;
        case 100:
          return `TPC-H (${storage}) (SF=100)`;
        case 1000:
          return `TPC-H (${storage}) (SF=1000)`;
        default:
          console.warn("Unknown scale factor:", scaleFactor);
          return null;
      }
    },

    normalizeSeriesName(name, seriesName) {
      let normalizedName = seriesName;
      let normalizedQuery = name;

      if (
        seriesName.endsWith(" throughput") ||
        seriesName.endsWith("throughput")
      ) {
        const suffix = seriesName.endsWith(" throughput")
          ? " throughput"
          : "throughput";
        normalizedName = seriesName.slice(0, seriesName.length - suffix.length);
        normalizedQuery = name.replace("time", "throughput");
      }

      return { name: normalizedQuery, seriesName: normalizedName };
    },

    formatQueryName(query) {
      let prettyQ = query.replace(/_/g, " ").toUpperCase();
      prettyQ = QUERY_NAME_MAP[prettyQ] || prettyQ;
      prettyQ = prettyQ.replace(/^TPCH\s/, "TPC-H ");
      return prettyQ;
    },

    convertValue(value, unit) {
      const isNanos = unit === "ns/iter" || unit === "ns";
      const isBytes = unit === "bytes";
      const isThroughput = unit === "bytes/ns";

      if (isNanos) return value / 1_000_000;
      if (isBytes) return value / 1_048_576;
      if (isThroughput) return (value * 1_000_000_000) / 1_048_576;
      return value;
    },

    getUnit(unit) {
      const isNanos = unit === "ns/iter" || unit === "ns";
      const isBytes = unit === "bytes";
      const isThroughput = unit === "bytes/ns";

      if (isNanos) return "ms/iter";
      if (isBytes) return "MiB";
      if (isThroughput) return "MiB/s";
      return unit;
    },

    downloadAndGroupData(data, commitMetadata, seriesRenameFn) {
      const commits = this.parseCommits(commitMetadata);
      const groups = this.initializeGroups();
      const uncategorizableNames = new Set();
      const missingCommits = new Set();

      for (const benchmark of data) {
        this.processBenchmark(
          benchmark,
          commitMetadata,
          commits,
          groups,
          seriesRenameFn,
          missingCommits,
          uncategorizableNames
        );
      }

      this.sortGroups(groups);

      if (missingCommits.size > 0) {
        console.warn(
          "These commits were missing from commits.json so the commit message is missing and the datetime is set to 1970-01-01T00:00:00Z",
          missingCommits
        );
      }
      if (uncategorizableNames.size > 0) {
        console.warn(
          "Could not categorize benchmarks with these names, they will not be shown:",
          uncategorizableNames
        );
      }

      return Object.keys(groups).map((name) => ({
        name,
        dataSet: groups[name],
      }));
    },

    initializeGroups() {
      const groups = {};
      BENCHMARK_GROUPS.forEach((name) => {
        groups[name] = new Map();
      });
      return groups;
    },

    processBenchmark(
      benchmark,
      commitMetadata,
      commits,
      groups,
      seriesRenameFn,
      missingCommits,
      uncategorizableNames
    ) {
      // Ensure commit metadata
      if (!benchmark.commit) {
        benchmark.commit = commitMetadata[benchmark.commit_id];
        if (!benchmark.commit) {
          missingCommits.add(benchmark.commit_id);
          benchmark.commit = commitMetadata[benchmark.commit_id] =
            this.createMissingCommit(benchmark.commit_id);
        }
      }

      // Determine group
      const groupId = this.determineGroupId(benchmark);
      if (!groupId) {
        uncategorizableNames.add(benchmark.name);
        return;
      }

      const group = groups[groupId];
      if (!group) {
        console.warn("Cannot find group element in group:", groupId);
        return;
      }

      // Process benchmark data
      let [query, seriesName] = benchmark.name.split("/");
      const normalized = this.normalizeSeriesName(query, seriesName);
      query = normalized.name;
      seriesName = normalized.seriesName;

      // Apply series renaming
      seriesName = this.applySeriesRenaming(
        seriesName,
        groupId,
        seriesRenameFn
      );

      // Format query name
      const prettyQ = this.formatQueryName(query);
      if (prettyQ.includes("PARQUET-UNC")) return;

      // Set units
      let unit = benchmark.unit;
      if (!unit && benchmark.name.startsWith("vortex size/")) {
        unit = "bytes";
      } else if (
        !unit &&
        (benchmark.name.startsWith("vortex:raw size/") ||
          benchmark.name.startsWith("vortex:parquet-zstd size/"))
      ) {
        unit = "ratio";
      }

      // Calculate sort position
      const sortPosition =
        query.slice(0, 4) === "tpch"
          ? parseInt(prettyQ.split(" ")[1].substring(1), 10)
          : 0;

      // Add to group
      this.addToGroup(
        group,
        prettyQ,
        seriesName,
        benchmark,
        unit,
        sortPosition,
        commits
      );
    },

    applySeriesRenaming(seriesName, groupId, seriesRenameFn) {
      if (!seriesRenameFn) return seriesName;

      const renamer = seriesRenameFn.find(([name]) => name === groupId);
      if (renamer?.[1]?.renamedDatasets) {
        const renameDict = renamer[1].renamedDatasets;
        return renameDict[seriesName] || seriesName;
      }
      return seriesName;
    },

    addToGroup(
      group,
      queryName,
      seriesName,
      benchmark,
      unit,
      sortPosition,
      commits
    ) {
      let arr = group.get(queryName);
      if (!arr) {
        group.set(queryName, {
          sort_position: sortPosition,
          commits,
          unit: this.getUnit(unit),
          series: new Map(),
        });
        arr = group.get(queryName);
      }

      let series = arr.series.get(seriesName);
      if (!series) {
        arr.series.set(seriesName, new Array(commits.length).fill(null));
        series = arr.series.get(seriesName);
      }

      series[benchmark.commit.sortedIndex] = {
        range: "this was the range",
        value: this.convertValue(benchmark.value, unit),
      };
    },

    sortGroups(groups) {
      const sortByPositionThenName = (a, b) => {
        const positionCompare = a[1].sort_position - b[1].sort_position;
        return positionCompare !== 0
          ? positionCompare
          : a[0].localeCompare(b[0]);
      };

      Object.entries(groups).forEach(([name, charts]) => {
        groups[name] = new Map(
          [...charts.entries()].sort(sortByPositionThenName)
        );
      });
    },
  };

  // Chart management module
  const chartManager = {
    createChartContainer(name, benchName, index) {
      const container = document.createElement("div");
      container.className = "chart-container fade-in";
      container.setAttribute("data-benchmark", name);
      container.setAttribute("data-chart", benchName);

      const header = document.createElement("div");
      header.className = "chart-header";

      const title = document.createElement("h3");
      title.className = "chart-title";
      title.textContent = benchName;

      const actions = document.createElement("div");
      actions.className = "chart-actions";

      const fullscreenBtn = document.createElement("button");
      fullscreenBtn.className = "chart-action-btn";
      fullscreenBtn.textContent = "Fullscreen";
      fullscreenBtn.onclick = () =>
        chartManager.openModal(name, benchName, index);

      actions.appendChild(fullscreenBtn);
      header.appendChild(title);
      header.appendChild(actions);
      container.appendChild(header);

      const canvas = document.createElement("canvas");
      canvas.id = `chart-${name}-${index}`;
      container.appendChild(canvas);

      return { container, canvas };
    },

    renderChart(
      parent,
      name,
      benchName,
      dataset,
      hiddenDatasets,
      removedDatasets,
      renamedDatasets,
      index
    ) {
      const { container, canvas } = this.createChartContainer(
        name,
        benchName,
        index
      );
      parent.appendChild(container);

      // Store chart configuration for lazy loading
      const chartConfig = {
        canvas,
        name,
        benchName,
        dataset,
        hiddenDatasets,
        removedDatasets,
        renamedDatasets,
        index,
      };

      // On mobile or when IntersectionObserver is available, use lazy loading
      const isMobile = utils.isMobile();
      if (isMobile && chartObserver) {
        container.chartData = chartConfig;
        chartObserver.observe(container);
        return null; // Don't create chart immediately
      }

      // Otherwise create chart immediately
      return this.createChartInstance(chartConfig);
    },

    createChartInstance(config) {
      const {
        canvas,
        name,
        benchName,
        dataset,
        hiddenDatasets,
        removedDatasets,
        renamedDatasets,
        index,
      } = config;

      // On mobile, limit data points
      const isMobile = utils.isMobile();
      const maxDataPoints = isMobile
        ? CONFIG.MOBILE_MAX_DATA_POINTS
        : dataset.commits.length;
      const startIndex = Math.max(0, dataset.commits.length - maxDataPoints);

      const limitedCommits = dataset.commits.slice(startIndex);
      const data = {
        labels: limitedCommits.map((commit) => commit.id.slice(0, 7)),
        datasets: Array.from(dataset.series)
          .filter(([name, benches]) => {
            return removedDatasets === undefined || !removedDatasets.has(name);
          })
          .map(([name, benches]) => {
            const renamedName =
              renamedDatasets === undefined
                ? name
                : renamedDatasets[name] || name;
            const color = utils.stringToColor(renamedName);
            const limitedData = benches.slice(startIndex);
            return {
              label: renamedName,
              data: limitedData.map((b) => (b ? b.value : null)),
              borderColor: color,
              backgroundColor: color + "60", // Add alpha for #rrggbbaa
              hidden: hiddenDatasets !== undefined && hiddenDatasets.has(name),
            };
          }),
      };

      const options = this.createChartOptions(
        name,
        benchName,
        dataset,
        limitedCommits,
        isMobile,
        index
      );

      const chart = new Chart(canvas, {
        type: "line",
        data: data,
        options: options,
      });

      const chartKey = `${name}-${index}`;
      state.chartInstances.set(chartKey, { chart, data, options });

      return chart;
    },

    createChartOptions(
      categoryName,
      benchName,
      dataset,
      limitedCommits,
      isMobile,
      index
    ) {
      const yAxisScale = this.createYAxisScale(benchName, dataset);

      return {
        responsive: true,
        maintainAspectRatio: false,
        aspectRatio: isMobile ? 1.5 : 2,
        spanGaps: true,
        pointStyle: isMobile ? false : "crossRot",
        resizeDelay: 0, // Disable resize delay
        elements: {
          line: {
            borderWidth: 1,
            tension: 0,
          },
          point: {
            radius: isMobile ? 0 : 3,
          },
        },
        animation: {
          duration: isMobile ? 0 : CONFIG.ANIMATION_DURATION,
        },
        scales: {
          x: {
            title: {
              display: true,
              text: benchName,
              padding: { bottom: 50 },
            },
            min: isMobile
              ? 0 // Start from the beginning of the sliced data
              : Math.max(
                  0,
                  dataset.commits.length - CONFIG.DEFAULT_VISIBLE_COMMITS
                ),
            max: isMobile
              ? limitedCommits.length - 1 // Use the length of the sliced data
              : undefined,
          },
          y: yAxisScale,
        },
        plugins: this.createPlugins(
          categoryName,
          isMobile,
          limitedCommits,
          index
        ),
        onClick: this.createClickHandler(limitedCommits),
      };
    },

    createYAxisScale(benchName, dataset) {
      const scale = {
        title: {
          display: true,
          text: dataset.commits.length > 0 ? dataset.unit : "",
        },
        suggestedMin: 0,
        beginAtZero: true, // Force chart to start at 0
      };

      if (
        benchName.includes("COMPRESS") &&
        benchName.includes("THROUGHPUT") &&
        dataset.unit === "MiB/s"
      ) {
        scale.suggestedMax = CONFIG.COMPRESS_THROUGHPUT_MAX;
        scale.max = CONFIG.COMPRESS_THROUGHPUT_MAX;
      }

      if (
        benchName.includes("DECOMPRESS") &&
        benchName.includes("THROUGHPUT") &&
        dataset.unit === "MiB/s"
      ) {
        scale.suggestedMax = CONFIG.DECOMPRESS_THROUGHPUT_MAX;
        scale.max = CONFIG.DECOMPRESS_THROUGHPUT_MAX;
      }

      return scale;
    },

    createPlugins(categoryName, isMobile, limitedCommits, index) {
      return {
        zoom: {
          zoom: {
            wheel: {
              enabled: !isMobile,
              speed: CONFIG.ZOOM_SPEED,
              modifierKey: null,
            },
            mode: "x",
            drag: {
              enabled: !isMobile,
              backgroundColor: "rgba(89, 113, 253, 0.1)",
            },
            onZoom: !isMobile
              ? ({ chart }) => {
                  zoomSync.synchronizeZoomForCategory(
                    categoryName,
                    chart,
                    index
                  );
                }
              : undefined,
          },
          pan: {
            enabled: !isMobile,
            mode: "x",
            modifierKey: null,
            onPan: !isMobile
              ? ({ chart }) => {
                  zoomSync.synchronizeZoomForCategory(
                    categoryName,
                    chart,
                    index,
                    false
                  );
                }
              : undefined,
          },
          limits: {
            x: {
              min: 0,
              max: limitedCommits.length - 1,
              minRange: Math.min(
                CONFIG.MIN_VISIBLE_COMMITS,
                limitedCommits.length
              ),
            },
          },
        },
        legend: {
          display: true,
          onClick: this.createLegendClickHandler(),
        },
        tooltip: {
          callbacks: {
            afterLabel: this.createTooltipCallback(limitedCommits),
          },
        },
      };
    },

    createClickHandler(limitedCommits) {
      return (event, elements) => {
        if (elements.length > 0) {
          const index = elements[0].index;
          const commit = limitedCommits[index];
          if (commit?.url) {
            window.open(commit.url, "_blank");
          }
        }
      };
    },

    createLegendClickHandler() {
      return function (e, legendItem) {
        const index = legendItem.datasetIndex;
        const chart = this.chart;
        const dataset = chart.data.datasets[index];
        dataset.hidden = !dataset.hidden;
        chart.update();
      };
    },

    createTooltipCallback(limitedCommits) {
      return (context) => {
        const dataIndex = context.dataIndex;
        const commit = limitedCommits[dataIndex];
        if (!commit) return [];

        return [
          "",
          commit.message.split("\n")[0],
          `${commit.author.name} - ${new Date(
            commit.timestamp
          ).toLocaleDateString()}`,
        ];
      };
    },

    openModal(benchmarkName, chartName, index) {
      const modal = domElements.chartModal;
      const modalCanvas = document.getElementById("modal-chart");

      const chartKey = `${benchmarkName}-${index}`;
      const originalChart = state.chartInstances.get(chartKey);
      if (!originalChart) return;

      const modalChart = new Chart(modalCanvas, {
        type: "line",
        data: JSON.parse(JSON.stringify(originalChart.data)),
        options: {
          ...originalChart.options,
          maintainAspectRatio: false,
          responsive: true,
        },
      });

      modal.classList.add("active");
      modal.modalChart = modalChart;
    },

    closeModal() {
      const modal = domElements.chartModal;
      if (modal.modalChart) {
        modal.modalChart.destroy();
        modal.modalChart = null;
      }
      modal.classList.remove("active");
    },

    cleanupCharts() {
      state.chartInstances.forEach((chartData) => {
        if (chartData?.chart) {
          chartData.chart.destroy();
        }
      });
      state.chartInstances.clear();
      state.charts = [];
    },

    updateChartsForResize() {
      // Prevent multiple simultaneous resize operations
      if (state.isResizing) return;
      state.isResizing = true;

      const currentIsMobile = utils.isMobile();
      const wasDesktop = state.lastWindowWidth > CONFIG.MOBILE_BREAKPOINT;
      const isDesktop = window.innerWidth > CONFIG.MOBILE_BREAKPOINT;
      const crossedThreshold =
        (wasDesktop && !isDesktop) || (!wasDesktop && isDesktop);

      // Update window width immediately
      state.lastWindowWidth = window.innerWidth;

      if (!crossedThreshold) {
        // Simple resize - just update all charts
        requestAnimationFrame(() => {
          state.chartInstances.forEach((chartData) => {
            if (chartData?.chart) {
              chartData.chart.resize();
              chartData.chart.update("none");
            }
          });
          state.isResizing = false;
        });
        return;
      }

      // For threshold crossing, update chart options
      requestAnimationFrame(() => {
        // Update all charts
        state.chartInstances.forEach((chartData, key) => {
          if (chartData?.chart) {
            const chart = chartData.chart;
            const totalCommits = chart.data.labels.length;

            // Store current state - deep clone y-axis to preserve all properties
            const currentXMin = chart.options.scales.x.min;
            const currentXMax = chart.options.scales.x.max;

            // Store actual scale values before any changes
            const actualYMin = chart.scales.y?.min;
            const actualYMax = chart.scales.y?.max;

            const currentYScale = JSON.parse(
              JSON.stringify(chart.options.scales.y)
            );

            // Determine new x-axis bounds
            let newXMin, newXMax;
            if (currentIsMobile) {
              // Show all data points on mobile (which is already limited)
              newXMax = totalCommits - 1;
              newXMin = 0; // Start from beginning of limited data
            } else {
              // Going to desktop - restore previous or use defaults
              const wasShowingAllMobileData =
                currentXMin === 0 && currentXMax === totalCommits - 1;
              if (wasShowingAllMobileData) {
                newXMin = Math.max(
                  0,
                  totalCommits - CONFIG.DEFAULT_VISIBLE_COMMITS
                );
                newXMax = totalCommits - 1;
              } else {
                newXMin = currentXMin;
                newXMax = currentXMax;
              }
            }

            // Update all options directly
            chart.options.animation.duration = 0;
            chart.options.aspectRatio = currentIsMobile ? 1.5 : 2;
            chart.options.pointStyle = currentIsMobile ? false : "crossRot";
            chart.options.elements.point.radius = currentIsMobile ? 0 : 3;

            // Update zoom settings
            if (chart.options.plugins.zoom) {
              const zoomEnabled = !currentIsMobile;
              chart.options.plugins.zoom.zoom.wheel.enabled = zoomEnabled;
              chart.options.plugins.zoom.zoom.pinch.enabled = zoomEnabled;
              chart.options.plugins.zoom.zoom.drag.enabled = false;
              chart.options.plugins.zoom.pan.enabled = zoomEnabled;
            }

            // Update x-axis only, preserve y-axis completely
            chart.options.scales.x.min = newXMin;
            chart.options.scales.x.max = newXMax;

            // Ensure y-axis is preserved with all its properties
            Object.keys(currentYScale).forEach((key) => {
              chart.options.scales.y[key] = currentYScale[key];
            });

            // Force preserve critical y-axis properties
            if (currentYScale.min !== undefined)
              chart.options.scales.y.min = currentYScale.min;
            if (currentYScale.max !== undefined)
              chart.options.scales.y.max = currentYScale.max;
            if (currentYScale.suggestedMin !== undefined)
              chart.options.scales.y.suggestedMin = currentYScale.suggestedMin;
            if (currentYScale.suggestedMax !== undefined)
              chart.options.scales.y.suggestedMax = currentYScale.suggestedMax;

            // If no explicit min/max in options, use the actual computed values
            if (
              chart.options.scales.y.min === undefined &&
              actualYMin !== undefined
            ) {
              chart.options.scales.y.min = actualYMin;
            }
            if (
              chart.options.scales.y.max === undefined &&
              actualYMax !== undefined
            ) {
              chart.options.scales.y.max = actualYMax;
            }

            // Single update per chart
            chart.update("none");
          }
        });

        // Reset animation duration after update
        setTimeout(() => {
          state.chartInstances.forEach((chartData) => {
            if (chartData?.chart) {
              chartData.chart.options.animation.duration = currentIsMobile
                ? 0
                : CONFIG.ANIMATION_DURATION;
            }
          });
        }, 100);

        // Recreate debounced sync zoom with new delay
        if (crossedThreshold) {
          debouncedSyncZoom = utils.debounce((categoryName) => {
            const update = state.pendingZoomUpdates.get(categoryName);
            if (!update) return;

            const { min, max, sourceIndex } = update;

            const categorySection = document.querySelector(
              `[data-category="${categoryName}"]`
            );
            if (!categorySection) return;

            const chartContainers =
              categorySection.querySelectorAll(".chart-container");

            requestAnimationFrame(() => {
              chartContainers.forEach((container, index) => {
                if (index === sourceIndex) return;

                const chartKey = `${categoryName}-${index}`;
                const chartData = state.chartInstances.get(chartKey);

                if (chartData?.chart) {
                  const chart = chartData.chart;
                  chart.options.scales.x.min = min;
                  chart.options.scales.x.max = max;
                  chart.update("none");
                }
              });
            });

            state.pendingZoomUpdates.delete(categoryName);
          }, utils.getDebounceDelay());
        }

        state.isResizing = false;
      });
    },
  };

  // Zoom synchronization module
  const zoomSync = {
    synchronizeZoomForCategory(
      categoryName,
      sourceChart,
      sourceIndex,
      isZoom = true
    ) {
      // Get the current zoom state from the source chart
      const xScale = sourceChart.scales.x;
      let min = xScale.min;
      let max = xScale.max;

      const isCurrentlyMobile = utils.isMobile();

      // Always anchor to the most recent commit when zooming (not on mobile)
      if (isZoom && !isCurrentlyMobile) {
        const totalCommits = sourceChart.data.labels.length;
        const currentRange = max - min;

        // Always keep the most recent commit visible
        max = totalCommits - 1;
        min = Math.max(0, max - currentRange);
      }

      // Store the update for this category
      state.pendingZoomUpdates.set(categoryName, { min, max, sourceIndex });

      // Debounce the actual sync operation
      debouncedSyncZoom(categoryName);
    },

    resetZoomForCategory(categoryName) {
      const section = document.querySelector(
        `[data-category="${categoryName}"]`
      );
      if (!section) return;

      const isCurrentlyMobile = utils.isMobile();
      const containers = section.querySelectorAll(".chart-container");

      containers.forEach((container, index) => {
        const chartKey = `${categoryName}-${index}`;
        const chartData = state.chartInstances.get(chartKey);

        if (chartData?.chart) {
          const chart = chartData.chart;
          const totalCommits = chart.data.labels.length;
          const minIndex = isCurrentlyMobile
            ? 0
            : Math.max(0, totalCommits - CONFIG.DEFAULT_VISIBLE_COMMITS);

          chart.options.scales.x.min = minIndex;
          chart.options.scales.x.max = totalCommits - 1;
          chart.update("none");
        }
      });
    },
  };

  // UI module
  const ui = {
    getTpchDescription(categoryName) {
      const scaleFactorMatch = categoryName.match(/SF=(\d+)/);
      const scaleFactor = scaleFactorMatch ? scaleFactorMatch[1] : null;
      const scaleFactorInfo =
        SCALE_FACTOR_DESCRIPTIONS[scaleFactor] || "various scale factors";

      if (categoryName.includes("NVMe")) {
        return `TPC-H benchmark queries executed on local NVMe storage, testing analytical query performance at ${scaleFactorInfo}`;
      } else if (categoryName.includes("S3")) {
        return `TPC-H benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance and the impact of network latency on analytical workloads at ${scaleFactorInfo}`;
      }
      return "";
    },

    getDescription(categoryName) {
      if (categoryName.startsWith("TPC-H")) {
        return this.getTpchDescription(categoryName);
      }
      const baseCategory = categoryName.split(" (")[0];
      return BENCHMARK_DESCRIPTIONS[baseCategory] || "";
    },

    createBenchmarkSection(name, benchSet, groupFilterSettings = {}) {
      const { keptCharts, hiddenDatasets, removedDatasets, renamedDatasets } =
        groupFilterSettings;

      const section = document.createElement("div");
      section.className = "benchmark-set";
      section.setAttribute("data-category", name);

      // Add header
      const header = this.createSectionHeader(name, benchSet, keptCharts);
      section.appendChild(header);

      // Add description
      const description = this.getDescription(name);
      if (description) {
        const descElem = document.createElement("div");
        descElem.className = "benchmark-description";
        descElem.textContent = description;
        section.appendChild(descElem);
      }

      // Add controls
      const controls = this.createSectionControls(name);
      if (controls) {
        section.appendChild(controls);
      }

      // Add charts container
      const chartsContainer = document.createElement("div");
      chartsContainer.className = "benchmark-graphs";
      section.appendChild(chartsContainer);

      // Expand by default
      state.expandedSections.add(name);

      return { section, chartsContainer };
    },

    createSectionHeader(name, benchSet, keptCharts) {
      const h1id = name.replace(/\s+/g, "_");

      const header = document.createElement("div");
      header.className = "benchmark-header";
      header.onclick = () => this.toggleSection(name);

      const titleWrapper = document.createElement("div");
      titleWrapper.className = "title-wrapper";

      const title = document.createElement("h1");
      title.id = h1id;
      title.className = "benchmark-title";
      title.innerHTML = `<span class="collapse-icon">▼</span> ${name}`;

      const linkBtn = document.createElement("button");
      linkBtn.className = "group-link-btn";
      linkBtn.setAttribute("aria-label", "Copy link to this section");
      linkBtn.innerHTML = "🔗";
      linkBtn.onclick = (e) => {
        e.stopPropagation();
        this.linkToGroup(name);
      };

      titleWrapper.appendChild(title);
      titleWrapper.appendChild(linkBtn);

      const meta = document.createElement("div");
      meta.className = "benchmark-meta";
      const chartCount = keptCharts ? keptCharts.length : benchSet?.size || 0;
      meta.textContent = `${chartCount} charts`;

      header.appendChild(titleWrapper);
      header.appendChild(meta);

      return header;
    },

    createSectionControls(name) {
      const tags = CATEGORY_TAGS[name] || [];
      const isQueryGroup = tags.some((tag) => tag.includes("Queries"));

      const container = document.createElement("div");
      container.className = "engine-filter-container";

      if (isQueryGroup) {
        const label = document.createElement("span");
        label.className = "engine-filter-label";
        label.textContent = "Show: ";
        container.appendChild(label);

        Object.entries(ENGINE_LABELS).forEach(([engine, label]) => {
          const btn = document.createElement("button");
          btn.className =
            "engine-filter-btn" +
            (engine === state.activeEngine ? " active" : "");
          btn.textContent = label;
          btn.setAttribute("data-engine", engine);
          btn.setAttribute("data-category", name);
          btn.onclick = () => this.filterEngine(name, engine);
          container.appendChild(btn);
        });

        const separator = document.createElement("span");
        separator.className = "filter-separator";
        separator.textContent = "|";
        container.appendChild(separator);
      }

      const resetBtn = document.createElement("button");
      resetBtn.className = "reset-zoom-btn";
      resetBtn.textContent = "Reset X-Axis";
      resetBtn.setAttribute("data-category", name);
      resetBtn.onclick = () => zoomSync.resetZoomForCategory(name);
      container.appendChild(resetBtn);

      return container;
    },

    toggleSection(name) {
      const section = document.querySelector(`[data-category="${name}"]`);
      if (!section) return;

      if (state.expandedSections.has(name)) {
        state.expandedSections.delete(name);
        section.classList.add("collapsed");
      } else {
        state.expandedSections.add(name);
        section.classList.remove("collapsed");
      }
    },

    linkToGroup(name) {
      urlManager.updateParams({ group: name });

      const targetSection = document.querySelector(`[data-category="${name}"]`);

      navigator.clipboard.writeText(window.location.href).then(() => {
        if (targetSection) {
          const linkBtn = targetSection.querySelector(".group-link-btn");
          if (linkBtn) {
            const originalText = linkBtn.innerHTML;
            linkBtn.innerHTML = "✓";
            linkBtn.classList.add("copied");
            setTimeout(() => {
              linkBtn.innerHTML = originalText;
              linkBtn.classList.remove("copied");
            }, CONFIG.LINK_FEEDBACK_DURATION);
          }
        }
      });
    },

    filterEngine(categoryName, engine) {
      state.activeEngine = engine;
      urlManager.updateParams({ engine });

      // Update all engine filter buttons
      document
        .querySelectorAll(".engine-filter-container")
        .forEach((container) => {
          container.querySelectorAll(".engine-filter-btn").forEach((btn) => {
            btn.classList.toggle(
              "active",
              btn.getAttribute("data-engine") === engine
            );
          });
        });

      // Apply filter to charts
      this.applyEngineFilter(engine);
    },

    applyEngineFilter(engine) {
      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const category = section.getAttribute("data-category");
        const tags = CATEGORY_TAGS[category] || [];
        const isQueryGroup = tags.some((tag) => tag.includes("Queries"));

        if (isQueryGroup) {
          const containers = section.querySelectorAll(".chart-container");
          containers.forEach((container, index) => {
            const chartKey = `${category}-${index}`;
            const chartData = state.chartInstances.get(chartKey);

            if (chartData?.chart) {
              this.updateChartVisibility(chartData.chart, engine);
            }
          });
        }
      });
    },

    updateChartVisibility(chart, engine) {
      const updates = [];

      chart.data.datasets.forEach((dataset, index) => {
        const label = dataset.label.toLowerCase();
        const shouldShow = engine === "all" || label.includes(engine);

        if (chart.isDatasetVisible(index) !== shouldShow) {
          updates.push({ index, visible: shouldShow });
        }
      });

      if (updates.length > 0) {
        updates.forEach(({ index, visible }) => {
          chart.setDatasetVisibility(index, visible);
        });
        chart.update("none");
      }
    },

    setView(view) {
      state.currentView = view;
      document.querySelectorAll(".benchmark-graphs").forEach((graphs) => {
        graphs.classList.toggle("list-view", view === "list");
      });

      document.querySelectorAll(".view-btn").forEach((btn) => {
        btn.classList.remove("active");
      });
      document.getElementById(`${view}-view`).classList.add("active");
    },
  };

  // URL management module
  const urlManager = {
    getParams() {
      const params = new URLSearchParams(window.location.search);
      return {
        tag: params.get("tag") || "all",
        engine: params.get("engine") || "all",
        expanded: params.get("expanded") || "true",
        group: params.get("group") || null,
      };
    },

    updateParams(updates) {
      const params = new URLSearchParams(window.location.search);

      Object.entries(updates).forEach(([key, value]) => {
        if (
          value &&
          value !== "all" &&
          !(key === "expanded" && value === "true")
        ) {
          params.set(key, value);
        } else {
          params.delete(key);
        }
      });

      const newURL =
        window.location.pathname +
        (params.toString() ? "?" + params.toString() : "");
      window.history.replaceState({}, "", newURL);
    },

    initializeFromParams() {
      const params = this.getParams();

      state.activeTag = params.tag;
      state.activeEngine = params.engine;

      const categoryFilter = domElements.categoryFilter;
      if (categoryFilter) {
        categoryFilter.value = params.tag;
        filterManager.filterByTag(params.tag);
      }

      if (params.engine !== "all") {
        ui.filterEngine(null, params.engine);
      }

      if (params.group) {
        setTimeout(() => {
          navigationManager.focusOnGroup(params.group);
        }, CONFIG.URL_INIT_DELAY);
      } else if (params.expanded === "false") {
        navigationManager.collapseAll();
      }
    },
  };

  // Filter management module
  const filterManager = {
    filterByTag(tag) {
      state.activeTag = tag;
      urlManager.updateParams({ tag });

      // Filter sections
      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const category = section.getAttribute("data-category");
        const tags = CATEGORY_TAGS[category] || [];
        section.style.display =
          tag === "all" || tags.includes(tag) ? "block" : "none";
      });

      // Filter navigation
      document.querySelectorAll(".toc-list li").forEach((navItem) => {
        const link = navItem.querySelector("a");
        if (link) {
          const targetId = link.getAttribute("href").substring(1);
          const targetSection = document.getElementById(targetId);

          if (targetSection?.closest(".benchmark-set")) {
            const category = targetSection
              .closest(".benchmark-set")
              .getAttribute("data-category");
            const tags = CATEGORY_TAGS[category] || [];
            navItem.style.display =
              tag === "all" || tags.includes(tag) ? "block" : "none";
          }
        }
      });

      // Update clear filter button
      const clearBtn = domElements.clearFilter;
      if (clearBtn) {
        clearBtn.style.display = tag === "all" ? "none" : "block";
        clearBtn.textContent = tag === "all" ? "" : `Clear Filter: ${tag}`;
      }
    },

    filterBySearch(term) {
      state.searchTerm = term.toLowerCase();

      document.querySelectorAll(".chart-container").forEach((chart) => {
        const benchmarkName = chart
          .getAttribute("data-benchmark")
          .toLowerCase();
        const chartName = chart.getAttribute("data-chart").toLowerCase();
        const matches =
          benchmarkName.includes(state.searchTerm) ||
          chartName.includes(state.searchTerm);
        chart.style.display = matches ? "block" : "none";
      });
    },
  };

  // Navigation management module
  const navigationManager = {
    expandAll() {
      const sections = document.querySelectorAll(".benchmark-set");
      const updates = [];

      sections.forEach((section) => {
        const category = section.getAttribute("data-category");
        state.expandedSections.add(category);
        if (section.classList.contains("collapsed")) {
          updates.push(() => section.classList.remove("collapsed"));
        }
      });

      utils.batchDOMUpdates(updates);
      urlManager.updateParams({ expanded: "true" });
    },

    collapseAll() {
      const sections = document.querySelectorAll(".benchmark-set");
      const updates = [];

      sections.forEach((section) => {
        const category = section.getAttribute("data-category");
        state.expandedSections.delete(category);
        if (!section.classList.contains("collapsed")) {
          updates.push(() => section.classList.add("collapsed"));
        }
      });

      utils.batchDOMUpdates(updates);
      urlManager.updateParams({ expanded: "false" });
    },

    focusOnGroup(groupName) {
      // Collapse all first
      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const category = section.getAttribute("data-category");
        state.expandedSections.delete(category);
        section.classList.add("collapsed");
      });

      // Expand target
      const targetSection = document.querySelector(
        `[data-category="${groupName}"]`
      );
      if (targetSection) {
        state.expandedSections.add(groupName);
        targetSection.classList.remove("collapsed");

        // Scroll to section
        const targetId = targetSection.querySelector(".benchmark-title").id;
        const targetElement = document.getElementById(targetId);
        const headerHeight =
          document.querySelector(".sticky-header").offsetHeight;
        const elementPosition =
          targetElement.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition =
          elementPosition - headerHeight - CONFIG.SCROLL_OFFSET_PADDING;

        window.scrollTo({
          top: offsetPosition,
          behavior: "smooth",
        });

        this.updateActiveNavItem(targetId);
      }
    },

    updateActiveNavItem(id) {
      document.querySelectorAll(".toc-list a").forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
      });
    },

    handleScroll() {
      const scrollY = window.scrollY;
      domElements.backToTop.classList.toggle(
        "visible",
        scrollY > CONFIG.BACK_TO_TOP_THRESHOLD
      );

      // Update active nav item
      const sections = document.querySelectorAll(".benchmark-set");
      let current = "";

      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= CONFIG.SCROLL_ACTIVE_THRESHOLD) {
          current = section.querySelector(".benchmark-title").id;
        }
      });

      if (current) {
        this.updateActiveNavItem(current);
      }
    },
  };

  // Initialization module
  const initializer = {
    async loadData() {
      const [dataResponse, commitsResponse] = await Promise.all([
        this.fetchGzippedData(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz"
        ),
        fetch(
          "https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json"
        ).then((r) => r.text()),
      ]);

      const data = this.parseJsonl(dataResponse);
      const commitsArray = this.parseJsonl(commitsResponse);

      const commits = {};
      commitsArray.forEach((commit) => {
        commits[commit.id] = commit;
      });

      return { data, commits };
    },

    async fetchGzippedData(url) {
      const response = await fetch(url);
      const decompressedStream = response.body.pipeThrough(
        new DecompressionStream("gzip")
      );
      const reader = decompressedStream.getReader();
      const decoder = new TextDecoder();
      let result = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
      }

      result += decoder.decode();
      return result;
    },

    parseJsonl(jsonl) {
      return jsonl
        .split("\n")
        .filter((line) => line.trim().length !== 0)
        .map((line) => JSON.parse(line));
    },

    initializeControls() {
      // Cache DOM elements
      const elementIds = [
        "menu-toggle",
        "sidebar",
        "sidebar-close",
        "expand-all",
        "collapse-all",
        "grid-view",
        "list-view",
        "category-filter",
        "clear-filter",
        "search-filter",
        "back-to-top",
        "modal-close",
        "chart-modal",
        "main",
        "toc",
      ];

      elementIds.forEach((id) => {
        const camelCaseId = id.replace(/-(.)/g, (match, char) =>
          char.toUpperCase()
        );
        domElements[camelCaseId] = document.getElementById(id);
      });

      // Initialize chart observer for lazy loading
      if ("IntersectionObserver" in window) {
        chartObserver = new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              if (entry.isIntersecting) {
                const container = entry.target;
                if (!container.hasAttribute("data-chart-loaded")) {
                  container.setAttribute("data-chart-loaded", "true");
                  const chartData = container.chartData;
                  if (chartData) {
                    chartManager.createChartInstance(chartData);
                  }
                }
              }
            });
          },
          {
            rootMargin: CONFIG.CHART_OBSERVER_MARGIN,
          }
        );
      }

      // Initialize debounced zoom sync
      debouncedSyncZoom = utils.debounce((categoryName) => {
        const update = state.pendingZoomUpdates.get(categoryName);
        if (!update) return;

        const { min, max, sourceIndex } = update;

        // Find all charts in this category
        const categorySection = document.querySelector(
          `[data-category="${categoryName}"]`
        );
        if (!categorySection) return;

        const chartContainers =
          categorySection.querySelectorAll(".chart-container");

        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
          chartContainers.forEach((container, index) => {
            // Skip the source chart
            if (index === sourceIndex) return;

            const chartKey = `${categoryName}-${index}`;
            const chartData = state.chartInstances.get(chartKey);

            if (chartData?.chart) {
              // Apply the same zoom to this chart
              const chart = chartData.chart;
              chart.options.scales.x.min = min;
              chart.options.scales.x.max = max;
              chart.update("none");
            }
          });
        });

        // Clear the pending update
        state.pendingZoomUpdates.delete(categoryName);
      }, utils.getDebounceDelay());

      // Set up event listeners
      this.setupEventListeners();
    },

    setupEventListeners() {
      // Mobile menu
      domElements.menuToggle.addEventListener("click", () => {
        domElements.sidebar.classList.toggle("active");
      });

      domElements.sidebarClose.addEventListener("click", () => {
        domElements.sidebar.classList.remove("active");
      });

      // Expand/Collapse
      domElements.expandAll.addEventListener("click", () =>
        navigationManager.expandAll()
      );
      domElements.collapseAll.addEventListener("click", () =>
        navigationManager.collapseAll()
      );

      // View controls
      domElements.gridView.addEventListener("click", () => ui.setView("grid"));
      domElements.listView.addEventListener("click", () => ui.setView("list"));

      // Filters
      domElements.categoryFilter.addEventListener("change", (e) => {
        filterManager.filterByTag(e.target.value);
      });

      domElements.clearFilter.addEventListener("click", () => {
        domElements.categoryFilter.value = "all";
        filterManager.filterByTag("all");
        urlManager.updateParams({ tag: "all" });
      });

      const debouncedSearch = utils.debounce(
        (term) => filterManager.filterBySearch(term),
        CONFIG.SEARCH_DEBOUNCE
      );
      domElements.searchFilter.addEventListener("input", (e) => {
        debouncedSearch(e.target.value);
      });

      // Scroll handling
      const throttledScroll = utils.throttle(
        () => navigationManager.handleScroll(),
        CONFIG.THROTTLE_SCROLL
      );
      window.addEventListener("scroll", throttledScroll);

      domElements.backToTop.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });

      // Modal
      domElements.modalClose.addEventListener("click", () =>
        chartManager.closeModal()
      );
      domElements.chartModal.addEventListener("click", (e) => {
        if (e.target.id === "chart-modal") {
          chartManager.closeModal();
        }
      });

      // Outside click for sidebar
      document.addEventListener("click", (e) => {
        if (
          !domElements.sidebar.contains(e.target) &&
          !domElements.menuToggle.contains(e.target)
        ) {
          domElements.sidebar.classList.remove("active");
        }
      });

      // Window resize handler
      const debouncedResize = utils.debounce(() => {
        chartManager.updateChartsForResize();
      }, CONFIG.RESIZE_DEBOUNCE);

      window.addEventListener("resize", debouncedResize);
    },
  };

  // Render benchmark set function
  function renderBenchmarkSet(
    name,
    benchSet,
    main,
    toc,
    groupFilterSettings = {}
  ) {
    const { section, chartsContainer } = ui.createBenchmarkSection(
      name,
      benchSet,
      groupFilterSettings
    );
    main.appendChild(section);

    // Create TOC entry
    const tocLi = document.createElement("li");
    const tocLink = document.createElement("a");
    const h1id = name.replace(/\s+/g, "_");
    tocLink.href = "#" + h1id;
    tocLink.innerHTML = name;
    tocLink.onclick = (e) => {
      e.preventDefault();
      const targetElement = document.getElementById(h1id);
      const headerHeight =
        document.querySelector(".sticky-header").offsetHeight;
      const elementPosition =
        targetElement.getBoundingClientRect().top + window.pageYOffset;
      const offsetPosition =
        elementPosition - headerHeight - CONFIG.SCROLL_OFFSET_PADDING;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth",
      });

      navigationManager.updateActiveNavItem(h1id);
    };
    tocLi.appendChild(tocLink);
    toc.appendChild(tocLi);

    // Render charts
    let chartIndex = 0;
    const { keptCharts, hiddenDatasets, removedDatasets, renamedDatasets } =
      groupFilterSettings;

    if (keptCharts === undefined) {
      if (benchSet !== undefined) {
        for (const [benchName, benches] of benchSet.entries()) {
          state.charts.push(
            chartManager.renderChart(
              chartsContainer,
              name,
              benchName,
              benches,
              hiddenDatasets,
              removedDatasets,
              renamedDatasets,
              chartIndex++
            )
          );
        }
      }
    } else {
      for (const benchName of keptCharts) {
        const benches = benchSet.get(benchName);
        if (benches) {
          state.charts.push(
            chartManager.renderChart(
              chartsContainer,
              name,
              benchName,
              benches,
              hiddenDatasets,
              removedDatasets,
              renamedDatasets,
              chartIndex++
            )
          );
        }
      }
    }
  }

  // Main initialization
  return async function initAndRender(keptGroups) {
    try {
      const { data, commits } = await initializer.loadData();
      const grouped = dataProcessor.downloadAndGroupData(
        data,
        commits,
        keptGroups
      );

      const main = domElements.main || document.getElementById("main");
      const toc = domElements.toc || document.getElementById("toc");

      // Clear loading indicator
      main.innerHTML = "";

      // Render all charts
      if (keptGroups === undefined) {
        for (const { name, dataSet } of grouped) {
          renderBenchmarkSet(name, dataSet, main, toc);
        }
      } else {
        const dataSetsMap = new Map(
          grouped.map(({ name, dataSet }) => [name, dataSet])
        );
        for (const [name, groupFilterSettings] of keptGroups) {
          const dataSet = dataSetsMap.get(name);
          renderBenchmarkSet(name, dataSet, main, toc, groupFilterSettings);
        }
      }

      initializer.initializeControls();
      urlManager.initializeFromParams();
    } catch (error) {
      console.error("Failed to load benchmark data:", error);
      const main = domElements.main || document.getElementById("main");
      main.innerHTML = `
        <div class="loading-indicator">
          <p style="color: red;">Failed to load benchmark data. Please try refreshing the page.</p>
          <p>${error.message}</p>
        </div>
      `;
    }
  };
})();
