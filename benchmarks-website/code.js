"use strict";

// Import modules
import { CONFIG, BENCHMARK_DESCRIPTIONS, CATEGORY_TAGS, SCALE_FACTOR_DESCRIPTIONS, ENGINE_LABELS } from './config.js';
import { utils } from './utils.js';
import { chartManager } from './chart-manager.js';
import { scoring } from './scoring.js';
import { zoomSync } from './zoom-sync.js';

// API configuration - use relative URLs when served from server.js
const API_BASE = '';  // Empty for same-origin requests

// Main module
window.initAndRender = (function () {
  // Constants
  const DEFAULT_COMMIT_RANGE = 100; // Show last 100 commits by default

  // State management
  const state = {
    currentView: "grid",
    expandedSections: new Set(),
    activeCategory: "all",
    activeTag: "all",
    activeEngines: new Set(["all"]),
    searchTerm: "",
    charts: [],
    chartInstances: new Map(),
    pendingZoomUpdates: new Map(),
    lastWindowWidth: window.innerWidth,
    isResizing: false,
    metadata: null,  // Store metadata from API
    loadedCharts: new Map(),  // Track which charts have been loaded
    chartDataCache: new Map(),  // Cache chart data
    groupFilterSettings: new Map(),  // Store filter settings per group
    defaultStartCommitIndex: 0,  // Will be set after metadata loads
  };

  // DOM element cache
  const domElements = {};
  let chartObserver = null;

  // Helper to convert timestamp to milliseconds (handles ISO string or number)
  const toTimestampMs = (timestamp) => {
    if (!timestamp) return null;
    if (typeof timestamp === 'number') return timestamp;
    return new Date(timestamp).getTime();
  };

  // API client module
  const api = {
    async fetchMetadata() {
      const response = await fetch(`${API_BASE}/api/metadata`);
      if (!response.ok) throw new Error(`Failed to fetch metadata: ${response.status}`);
      return response.json();
    },

    async fetchChartData(groupName, chartName, startTimestamp = null, endTimestamp = null) {
      let url = `${API_BASE}/api/data/${encodeURIComponent(groupName)}/${encodeURIComponent(chartName)}`;
      const params = new URLSearchParams();
      // Convert timestamps to ms if they're ISO strings
      if (startTimestamp) params.set('start', toTimestampMs(startTimestamp));
      if (endTimestamp) params.set('end', toTimestampMs(endTimestamp));
      if (params.toString()) url += '?' + params.toString();

      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to fetch chart data: ${response.status}`);
      return response.json();
    }
  };

  // UI module
  const ui = {
    getTpchDescription(categoryName) {
      const scaleFactorMatch = categoryName.match(/SF=(\d+)/);
      const scaleFactor = scaleFactorMatch ? scaleFactorMatch[1] : null;
      const scaleFactorInfo = SCALE_FACTOR_DESCRIPTIONS[scaleFactor] || "various scale factors";

      if (categoryName.includes("NVMe")) {
        return `TPC-H benchmark queries executed on local NVMe storage, testing analytical query performance at ${scaleFactorInfo}`;
      } else if (categoryName.includes("S3")) {
        return `TPC-H benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance and the impact of network latency on analytical workloads at ${scaleFactorInfo}`;
      }
      return "";
    },

    getTpcdsDescription(categoryName) {
      const scaleFactorMatch = categoryName.match(/SF=(\d+)/);
      const scaleFactor = scaleFactorMatch ? scaleFactorMatch[1] : null;
      const scaleFactorInfo = SCALE_FACTOR_DESCRIPTIONS[scaleFactor] || "various scale factors";

      if (categoryName.includes("NVMe")) {
        return `TPC-DS benchmark queries executed on local NVMe storage, testing complex analytical query performance with a retail sales dataset at ${scaleFactorInfo}`;
      } else if (categoryName.includes("S3")) {
        return `TPC-DS benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance for complex retail analytics workloads at ${scaleFactorInfo}`;
      }
      return "";
    },

    getDescription(categoryName) {
      if (categoryName.startsWith("TPC-H")) {
        return this.getTpchDescription(categoryName);
      } else if (categoryName.startsWith("TPC-DS")) {
        return this.getTpcdsDescription(categoryName);
      }
      const baseCategory = categoryName.split(" (")[0];
      return BENCHMARK_DESCRIPTIONS[baseCategory] || "";
    },

    createBenchmarkSectionFromMetadata(name, groupMetadata, groupFilterSettings = {}) {
      const { keptCharts } = groupFilterSettings;

      const section = document.createElement("div");
      section.className = "benchmark-set";
      section.setAttribute("data-category", name);

      // Check if this benchmark group has any data
      const hasData = groupMetadata.hasData;
      if (!hasData) {
        section.classList.add("no-data");
      }

      // Create wrapper for sticky header
      const stickyWrapper = document.createElement("div");
      stickyWrapper.className = "sticky-header-wrapper";

      const stickyContainer = document.createElement("div");
      stickyContainer.className = "sticky-header-container";

      // Add header
      const header = this.createSectionHeaderFromMetadata(name, groupMetadata, keptCharts);
      stickyContainer.appendChild(header);

      // Add controls
      const controls = this.createSectionControls(name);
      if (controls) {
        stickyContainer.appendChild(controls);
      }

      stickyWrapper.appendChild(stickyContainer);
      section.appendChild(stickyWrapper);

      // Render summary from metadata (always visible, even when collapsed)
      if (groupMetadata.summary) {
        const summaryElement = this.renderSummaryFromMetadata(name, groupMetadata.summary);
        if (summaryElement) {
          section.appendChild(summaryElement);
        }
      }

      // Add summary placeholder for additional scoring data (populated when charts load)
      const summaryPlaceholder = document.createElement("div");
      summaryPlaceholder.className = "benchmark-summary-placeholder";
      summaryPlaceholder.setAttribute("data-group", name);
      section.appendChild(summaryPlaceholder);

      // Add charts container
      const chartsContainer = document.createElement("div");
      chartsContainer.className = "benchmark-graphs";
      chartsContainer.setAttribute("data-group", name);

      // Add chart placeholders based on metadata
      const charts = keptCharts || groupMetadata.charts.map(c => c.name);
      if (charts.length === 1) {
        chartsContainer.classList.add("single-chart");
      }

      // Create placeholder for each chart
      charts.forEach((chartName, index) => {
        const chartMeta = groupMetadata.charts.find(c => c.name === chartName);
        if (chartMeta) {
          const placeholder = this.createChartPlaceholder(name, chartMeta, index);
          chartsContainer.appendChild(placeholder);
        }
      });

      section.appendChild(chartsContainer);

      // Collapse by default
      section.classList.add("collapsed");

      return { section, chartsContainer };
    },

    createChartPlaceholder(groupName, chartMeta, index) {
      const container = document.createElement("div");
      container.className = "chart-container chart-placeholder fade-in";
      container.setAttribute("data-benchmark", groupName);
      container.setAttribute("data-chart", chartMeta.name);
      container.setAttribute("data-chart-index", index);

      const header = document.createElement("div");
      header.className = "chart-header";

      const title = document.createElement("h3");
      title.className = "chart-title";
      title.textContent = chartManager.remapNames(chartMeta.name);

      const actions = document.createElement("div");
      actions.className = "chart-actions";

      // Create zoom controls
      const zoomControls = this.createZoomControls(groupName, index);
      actions.appendChild(zoomControls);

      const fullscreenBtn = document.createElement("button");
      fullscreenBtn.className = "chart-action-btn";
      fullscreenBtn.textContent = "Fullscreen";
      fullscreenBtn.onclick = () => chartManager.openModal(groupName, chartMeta.name, index);

      actions.appendChild(fullscreenBtn);
      header.appendChild(title);
      header.appendChild(actions);
      container.appendChild(header);

      // Add loading placeholder for canvas area
      const canvasPlaceholder = document.createElement("div");
      canvasPlaceholder.className = "chart-canvas-placeholder";
      canvasPlaceholder.innerHTML = '<div class="chart-loading-spinner"></div>';
      container.appendChild(canvasPlaceholder);

      return container;
    },

    createZoomControls(groupName, index) {
      const zoomControls = document.createElement("div");
      zoomControls.className = "chart-zoom-controls";

      const chartKey = `${groupName}-${index}`;
      const createControlBtn = (text, title, clickHandler, dataAction) => {
        const btn = document.createElement("button");
        btn.className = "chart-zoom-btn";
        btn.textContent = text;
        btn.title = title;
        btn.onclick = clickHandler;
        btn.setAttribute("data-chart-key", chartKey);
        btn.setAttribute("data-action", dataAction);
        return btn;
      };

      const goToStartBtn = createControlBtn("|«", "Go to oldest", () =>
        chartManager.goToStart(groupName, index), "go-start");
      const panLeftBtn = createControlBtn("«", "Pan left", () =>
        chartManager.panChart(groupName, index, -0.5), "pan-left");
      const panRightBtn = createControlBtn("»", "Pan right", () =>
        chartManager.panChart(groupName, index, 0.5), "pan-right");
      const goToEndBtn = createControlBtn("»|", "Go to latest", () =>
        chartManager.goToEnd(groupName, index), "go-end");
      const zoomInBtn = createControlBtn("+", "Zoom in", () =>
        chartManager.zoomChart(groupName, index, 0.5), "zoom-in");
      const zoomOutBtn = createControlBtn("−", "Zoom out", () =>
        chartManager.zoomChart(groupName, index, 2), "zoom-out");

      zoomControls.appendChild(goToStartBtn);
      zoomControls.appendChild(panLeftBtn);
      zoomControls.appendChild(zoomInBtn);
      zoomControls.appendChild(zoomOutBtn);
      zoomControls.appendChild(panRightBtn);
      zoomControls.appendChild(goToEndBtn);

      return zoomControls;
    },

    createSectionHeaderFromMetadata(name, groupMetadata, keptCharts) {
      const h1id = name.replace(/\s+/g, "_");

      const header = document.createElement("div");
      header.className = "benchmark-header";

      if (groupMetadata.hasData) {
        header.onclick = (e) => {
          if (!e.target.closest(".info-icon")) {
            this.toggleSection(name);
          }
        };
      }

      const titleWrapper = document.createElement("div");
      titleWrapper.className = "title-wrapper";

      const title = document.createElement("h1");
      title.id = h1id;
      title.className = "benchmark-title";
      title.innerHTML = `<span class="collapse-icon">▼</span> ${name}`;

      const secondaryInfo = document.createElement("div");
      secondaryInfo.className = "benchmark-secondary-info";

      const linkBtn = document.createElement("button");
      linkBtn.className = "group-link-btn";
      linkBtn.setAttribute("aria-label", "Copy link to this section");
      linkBtn.innerHTML = "🔗";
      linkBtn.onclick = (e) => {
        e.stopPropagation();
        this.linkToGroup(name);
      };
      secondaryInfo.appendChild(linkBtn);

      const description = this.getDescription(name);
      if (description) {
        const infoIcon = document.createElement("div");
        infoIcon.className = "info-icon";
        infoIcon.innerHTML = "ⓘ";
        infoIcon.setAttribute("data-tooltip", description);
        secondaryInfo.appendChild(infoIcon);
      }

      const meta = document.createElement("div");
      meta.className = "benchmark-meta";
      const chartCount = keptCharts ? keptCharts.length : groupMetadata.totalCharts;
      meta.textContent = groupMetadata.hasData ? `${chartCount} charts` : "No data available";
      secondaryInfo.appendChild(meta);

      titleWrapper.appendChild(title);
      titleWrapper.appendChild(secondaryInfo);
      header.appendChild(titleWrapper);

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

        Object.entries(ENGINE_LABELS).forEach(([engine, labelText]) => {
          const btn = document.createElement("button");
          btn.className = "engine-filter-btn" + (state.activeEngines.has(engine) ? " active" : "");
          btn.textContent = labelText;
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
      resetBtn.onclick = () => window.zoomSync.resetZoomForCategory(name, state, utils, CONFIG);
      container.appendChild(resetBtn);

      return container;
    },

    async toggleSection(name) {
      const section = document.querySelector(`[data-category="${name}"]`);
      if (!section) return;

      if (section.classList.contains("no-data")) return;

      if (state.expandedSections.has(name)) {
        state.expandedSections.delete(name);
        section.classList.add("collapsed");
      } else {
        state.expandedSections.add(name);
        section.classList.remove("collapsed");

        // Load charts for this section if not already loaded
        await chartLoader.loadChartsForSection(name);
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
      if (engine === "all") {
        state.activeEngines.clear();
        state.activeEngines.add("all");
      } else {
        if (state.activeEngines.has("all")) {
          state.activeEngines.clear();
        }

        if (state.activeEngines.has(engine)) {
          state.activeEngines.delete(engine);
          if (state.activeEngines.size === 0) {
            state.activeEngines.add("all");
          }
        } else {
          state.activeEngines.add(engine);
        }
      }

      const engineParam = state.activeEngines.has("all")
        ? "all"
        : Array.from(state.activeEngines).join(",");
      urlManager.updateParams({ engine: engineParam });

      document.querySelectorAll(".engine-filter-container").forEach((container) => {
        container.querySelectorAll(".engine-filter-btn").forEach((btn) => {
          const btnEngine = btn.getAttribute("data-engine");
          btn.classList.toggle("active", state.activeEngines.has(btnEngine));
        });
      });

      this.applyEngineFilter();
    },

    applyEngineFilter() {
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
              this.updateChartVisibility(chartData.chart);
            }
          });
        }
      });
    },

    updateChartVisibility(chart) {
      const updates = [];

      chart.data.datasets.forEach((dataset, index) => {
        const label = dataset.label.toLowerCase();
        const shouldShow =
          state.activeEngines.has("all") ||
          Array.from(state.activeEngines).some((engine) => label.includes(engine));

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

    // Render summary from pre-calculated metadata
    renderSummaryFromMetadata(groupName, summary) {
      if (!summary) return null;

      const formatTime = (ms) => {
        if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
        if (ms < 1000) return `${ms.toFixed(1)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
      };

      const summaryDiv = document.createElement("div");
      summaryDiv.className = "benchmark-scores-summary";

      const title = document.createElement("h3");
      title.className = "scores-title";
      title.textContent = summary.title;
      summaryDiv.appendChild(title);

      const scoresList = document.createElement("div");
      scoresList.className = "scores-list";

      if (summary.type === 'randomAccess' && summary.rankings) {
        summary.rankings.forEach((item, index) => {
          const scoreItem = document.createElement("div");
          scoreItem.className = "score-item";
          scoreItem.innerHTML = `
            <span class="score-rank">#${index + 1}</span>
            <span class="score-series">${item.name}</span>
            <span class="score-metrics">
              <span class="score-value">${formatTime(item.time)}</span>
              <span class="score-runtime">${item.ratio.toFixed(2)}x</span>
            </span>
          `;
          scoresList.appendChild(scoreItem);
        });
      } else if (summary.type === 'compression') {
        if (summary.compressRatio !== null) {
          const compressItem = document.createElement("div");
          compressItem.className = "score-item";
          compressItem.innerHTML = `
            <span class="score-rank">⚡</span>
            <span class="score-series">Write Speed (Compression)</span>
            <span class="score-metrics">
              <span class="score-value">${summary.compressRatio.toFixed(2)}x</span>
            </span>
          `;
          scoresList.appendChild(compressItem);
        }
        if (summary.decompressRatio !== null) {
          const decompressItem = document.createElement("div");
          decompressItem.className = "score-item";
          decompressItem.innerHTML = `
            <span class="score-rank">📤</span>
            <span class="score-series">Scan Speed (Decompression)</span>
            <span class="score-metrics">
              <span class="score-value">${summary.decompressRatio.toFixed(2)}x</span>
            </span>
          `;
          scoresList.appendChild(decompressItem);
        }
      } else if (summary.type === 'compressionSize') {
        const minItem = document.createElement("div");
        minItem.className = "score-item";
        minItem.innerHTML = `
          <span class="score-rank">⬇️</span>
          <span class="score-series">Min Size Ratio</span>
          <span class="score-metrics">
            <span class="score-value">${summary.minRatio.toFixed(2)}x</span>
          </span>
        `;
        scoresList.appendChild(minItem);

        const meanItem = document.createElement("div");
        meanItem.className = "score-item";
        meanItem.innerHTML = `
          <span class="score-rank">📊</span>
          <span class="score-series">Mean Size Ratio</span>
          <span class="score-metrics">
            <span class="score-value">${summary.meanRatio.toFixed(2)}x</span>
          </span>
        `;
        scoresList.appendChild(meanItem);

        const maxItem = document.createElement("div");
        maxItem.className = "score-item";
        maxItem.innerHTML = `
          <span class="score-rank">⬆️</span>
          <span class="score-series">Max Size Ratio</span>
          <span class="score-metrics">
            <span class="score-value">${summary.maxRatio.toFixed(2)}x</span>
          </span>
        `;
        scoresList.appendChild(maxItem);
      } else if (summary.type === 'queryBenchmark' && summary.rankings) {
        summary.rankings.forEach((item, index) => {
          const scoreItem = document.createElement("div");
          scoreItem.className = "score-item";
          scoreItem.innerHTML = `
            <span class="score-rank">#${index + 1}</span>
            <span class="score-series">${item.name}</span>
            <span class="score-metrics">
              <span class="score-value">${item.score.toFixed(2)}x</span>
              <span class="score-runtime">${formatTime(item.totalRuntime)}</span>
            </span>
          `;
          scoresList.appendChild(scoreItem);
        });
      }

      summaryDiv.appendChild(scoresList);

      const explanation = document.createElement("div");
      explanation.className = "scores-explanation";
      explanation.textContent = summary.explanation;
      summaryDiv.appendChild(explanation);

      return summaryDiv;
    },
  };

  // Chart loader module - handles lazy loading of chart data
  const chartLoader = {
    loadingGroups: new Set(),

    async loadChartsForSection(groupName) {
      // Check if already loading or loaded
      if (this.loadingGroups.has(groupName)) return;
      if (state.loadedCharts.has(groupName)) return;

      this.loadingGroups.add(groupName);

      const section = document.querySelector(`[data-category="${groupName}"]`);
      if (!section) {
        this.loadingGroups.delete(groupName);
        return;
      }

      const groupMetadata = state.metadata.groups[groupName];
      if (!groupMetadata || !groupMetadata.hasData) {
        this.loadingGroups.delete(groupName);
        return;
      }

      const filterSettings = state.groupFilterSettings.get(groupName) || {};
      const { keptCharts, hiddenDatasets, removedDatasets, renamedDatasets } = filterSettings;

      const chartsContainer = section.querySelector('.benchmark-graphs');
      if (!chartsContainer) {
        this.loadingGroups.delete(groupName);
        return;
      }

      // Load each chart
      const chartNames = keptCharts || groupMetadata.charts.map(c => c.name);
      const loadedData = new Map();

      // Determine default range (last DEFAULT_COMMIT_RANGE commits) using timestamps
      const startTimestamp = state.defaultStartCommitIndex > 0
        ? state.metadata.commits[state.defaultStartCommitIndex]?.timestamp
        : null;

      // Load chart data in parallel
      const loadPromises = chartNames.map(async (chartName) => {
        try {
          const data = await api.fetchChartData(groupName, chartName, startTimestamp, null);
          loadedData.set(chartName, data);
        } catch (err) {
          console.error(`Failed to load chart ${groupName}/${chartName}:`, err);
        }
      });

      await Promise.all(loadPromises);

      // Summaries are now rendered from metadata, no need to recalculate

      // Replace placeholders with actual charts
      const placeholders = chartsContainer.querySelectorAll('.chart-placeholder');
      placeholders.forEach((placeholder, index) => {
        const chartName = placeholder.getAttribute('data-chart');
        const chartData = loadedData.get(chartName);

        if (chartData) {
          // Replace placeholder with real chart
          this.replaceWithChart(placeholder, groupName, chartName, chartData, index, filterSettings);
        }
      });

      state.loadedCharts.set(groupName, loadedData);
      this.loadingGroups.delete(groupName);

      // Update zoom sync cache for this category
      window.zoomSync.updateCacheForCategory(groupName);
    },

    convertToBenchSet(loadedData) {
      const benchSet = new Map();
      for (const [chartName, chartData] of loadedData.entries()) {
        const series = new Map();
        for (const [seriesName, seriesData] of Object.entries(chartData.series)) {
          // API now returns raw values, wrap in objects for scoring functions
          series.set(seriesName, seriesData.map(d => d !== null ? { value: d } : null));
        }
        benchSet.set(chartName, {
          unit: chartData.unit,
          commits: chartData.commits,
          series: series
        });
      }
      return benchSet;
    },

    async populateSummary(section, groupName, benchSet) {
      const summaryPlaceholder = section.querySelector('.benchmark-summary-placeholder');
      if (!summaryPlaceholder) return;

      // Add scoring summaries based on benchmark type
      if (scoring.isQueryBenchmark(groupName) && benchSet.size > 0) {
        const scores = scoring.calculateClickBenchScore(benchSet);
        const scoreSummary = scoring.formatScoresSummary(scores);
        if (scoreSummary) {
          summaryPlaceholder.appendChild(scoreSummary);
        }
      }

      if (scoring.isRandomAccessBenchmark(groupName) && benchSet.size > 0) {
        const metrics = scoring.calculateRandomAccessMetrics(benchSet);
        const metricsSummary = scoring.formatRandomAccessSummary(metrics);
        if (metricsSummary) {
          summaryPlaceholder.appendChild(metricsSummary);
        }
      }

      if (scoring.isCompressionBenchmark(groupName) && benchSet.size > 0) {
        const metrics = scoring.calculateCompressionMetrics(benchSet);
        const metricsSummary = scoring.formatCompressionSummary(metrics);
        if (metricsSummary) {
          summaryPlaceholder.appendChild(metricsSummary);
        }
      }

      if (scoring.isCompressionSizeBenchmark(groupName) && benchSet.size > 0) {
        const metrics = scoring.calculateCompressionSizeMetrics(benchSet);
        const metricsSummary = scoring.formatCompressionSizeSummary(metrics);
        if (metricsSummary) {
          summaryPlaceholder.appendChild(metricsSummary);
        }
      }
    },

    replaceWithChart(placeholder, groupName, chartName, chartData, index, filterSettings) {
      const { hiddenDatasets, removedDatasets, renamedDatasets } = filterSettings;

      // Create canvas element
      const canvas = document.createElement("canvas");
      canvas.id = `chart-${groupName}-${index}`;

      // Remove placeholder content and add canvas
      const canvasPlaceholder = placeholder.querySelector('.chart-canvas-placeholder');
      if (canvasPlaceholder) {
        canvasPlaceholder.replaceWith(canvas);
      }

      placeholder.classList.remove('chart-placeholder');

      // Format date helper
      const formatDate = (timestamp) => {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        return `${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
      };

      // Build dataset for Chart.js
      const labels = chartData.commits.map(c => formatDate(c.timestamp));
      const datasets = [];

      for (const [seriesName, seriesData] of Object.entries(chartData.series)) {
        if (removedDatasets && removedDatasets.has(seriesName)) continue;

        const displayName = renamedDatasets?.[seriesName] || seriesName;
        const color = utils.stringToColor(displayName);

        // seriesData from API contains raw values (numbers or null)
        datasets.push({
          label: displayName,
          data: seriesData,
          borderColor: color,
          backgroundColor: color + "60",
          hidden: (hiddenDatasets && hiddenDatasets.has(seriesName)) ||
                  seriesName.toLowerCase().startsWith("wide table cols")
        });
      }

      const isMobile = utils.isMobile();
      const options = chartManager.createChartOptions(
        groupName,
        chartName,
        { commits: chartData.commits, unit: chartData.unit },
        chartData.commits,
        isMobile,
        index
      );

      const chart = new Chart(canvas, {
        type: "line",
        data: { labels, datasets },
        options: options
      });

      const chartKey = `${groupName}-${index}`;
      state.chartInstances.set(chartKey, {
        chart,
        data: { labels, datasets },
        options,
        chartName,
        groupName,
        originalData: chartData,  // Store original response for range info
        filterSettings
      });

      // Update navigation button states for initial load
      if (chartManager?.updateNavigationButtons) {
        chartManager.updateNavigationButtons(chartKey);
      }
    },

    // Fetch new data when zoom/pan changes the visible range
    async refreshChartData(chartKey, startIndex, endIndex) {
      const chartInstance = state.chartInstances.get(chartKey);
      if (!chartInstance) return;

      const { chart, groupName, chartName, filterSettings, originalData } = chartInstance;
      if (!chart || !originalData) return;

      // Get timestamps for the range
      const allCommits = state.metadata.commits;
      const startTimestamp = allCommits[startIndex]?.timestamp || null;
      const endTimestamp = allCommits[endIndex]?.timestamp || null;

      try {
        const newData = await api.fetchChartData(groupName, chartName, startTimestamp, endTimestamp);

        // Update chart with new data
        const { hiddenDatasets, removedDatasets, renamedDatasets } = filterSettings || {};

        // Format date helper
        const formatDate = (timestamp) => {
          if (!timestamp) return '';
          const date = new Date(timestamp);
          const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
          return `${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
        };

        const newLabels = newData.commits.map(c => formatDate(c.timestamp));
        const newDatasets = [];

        for (const [seriesName, seriesData] of Object.entries(newData.series)) {
          if (removedDatasets && removedDatasets.has(seriesName)) continue;

          const displayName = renamedDatasets?.[seriesName] || seriesName;
          const color = utils.stringToColor(displayName);

          newDatasets.push({
            label: displayName,
            data: seriesData,
            borderColor: color,
            backgroundColor: color + "60",
            hidden: (hiddenDatasets && hiddenDatasets.has(seriesName)) ||
                    seriesName.toLowerCase().startsWith("wide table cols")
          });
        }

        // Preserve hidden state from current chart
        const currentHiddenState = new Map();
        chart.data.datasets.forEach((ds, idx) => {
          currentHiddenState.set(ds.label, !chart.isDatasetVisible(idx));
        });

        // Apply hidden state to new datasets
        newDatasets.forEach(ds => {
          if (currentHiddenState.has(ds.label)) {
            ds.hidden = currentHiddenState.get(ds.label);
          }
        });

        // Update chart
        chart.data.labels = newLabels;
        chart.data.datasets = newDatasets;

        // Reset zoom limits based on new data length
        if (chart.options.plugins.zoom?.limits?.x) {
          chart.options.plugins.zoom.limits.x.max = newLabels.length - 1;
        }

        // Update x-axis to show all new data
        chart.options.scales.x.min = 0;
        chart.options.scales.x.max = newLabels.length - 1;

        // Update x2 (date) axis labels and range
        chart.update('none');

        // Update stored data
        chartInstance.originalData = newData;

        console.log(`Refreshed chart ${chartKey} with ${newLabels.length} points (${newData.downsampleLevel})`);

        // Update navigation button states after data refresh
        if (chartManager?.updateNavigationButtons) {
          chartManager.updateNavigationButtons(chartKey);
        }
      } catch (err) {
        console.error(`Failed to refresh chart ${chartKey}:`, err);
      }
    }
  };

  // URL management module
  const urlManager = {
    getParams() {
      const params = new URLSearchParams(window.location.search);
      return {
        tag: params.get("tag") || "all",
        engine: params.get("engine") || "all",
        expanded: params.get("expanded") || "false",
        group: params.get("group") || null,
      };
    },

    updateParams(updates) {
      const params = new URLSearchParams(window.location.search);

      Object.entries(updates).forEach(([key, value]) => {
        if (value && value !== "all" && !(key === "expanded" && value === "false")) {
          params.set(key, value);
        } else {
          params.delete(key);
        }
      });

      const newURL = window.location.pathname + (params.toString() ? "?" + params.toString() : "");
      window.history.replaceState({}, "", newURL);
    },

    initializeFromParams() {
      const params = this.getParams();

      state.activeTag = params.tag;

      if (params.engine && params.engine !== "all") {
        const engines = params.engine.split(",");
        state.activeEngines.clear();
        engines.forEach((engine) => state.activeEngines.add(engine.trim()));
      }

      const categoryFilter = domElements.categoryFilter;
      if (categoryFilter) {
        categoryFilter.value = params.tag;
        filterManager.filterByTag(params.tag);
      }

      if (params.engine !== "all") {
        ui.applyEngineFilter();
      }

      if (params.group) {
        setTimeout(() => {
          navigationManager.focusOnGroup(params.group);
        }, CONFIG.URL_INIT_DELAY);
      } else if (params.expanded === "true") {
        navigationManager.expandAll();
      }
    },
  };

  // Filter management module
  const filterManager = {
    filterByTag(tag) {
      state.activeTag = tag;
      urlManager.updateParams({ tag });

      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const category = section.getAttribute("data-category");
        const tags = CATEGORY_TAGS[category] || [];
        section.style.display = tag === "all" || tags.includes(tag) ? "block" : "none";
      });

      document.querySelectorAll(".toc-list li").forEach((navItem) => {
        const link = navItem.querySelector("a");
        if (link) {
          const targetId = link.getAttribute("href").substring(1);
          const targetSection = document.getElementById(targetId);

          if (targetSection?.closest(".benchmark-set")) {
            const category = targetSection.closest(".benchmark-set").getAttribute("data-category");
            const tags = CATEGORY_TAGS[category] || [];
            navItem.style.display = tag === "all" || tags.includes(tag) ? "block" : "none";
          }
        }
      });

      const clearBtn = domElements.clearFilter;
      if (clearBtn) {
        clearBtn.style.display = tag === "all" ? "none" : "block";
        clearBtn.textContent = tag === "all" ? "" : `Clear Filter: ${tag}`;
      }
    },

    filterBySearch(term) {
      state.searchTerm = term.toLowerCase();

      document.querySelectorAll(".chart-container").forEach((chart) => {
        const benchmarkName = chart.getAttribute("data-benchmark").toLowerCase();
        const chartName = chart.getAttribute("data-chart").toLowerCase();
        const matches = benchmarkName.includes(state.searchTerm) || chartName.includes(state.searchTerm);
        chart.style.display = matches ? "block" : "none";
      });

      document.querySelectorAll(".benchmark-set").forEach((section) => {
        const visibleCharts = section.querySelectorAll(
          ".chart-container[style='display: block;'], .chart-container:not([style]), .chart-container[style='']"
        );
        section.style.display = visibleCharts.length > 0 ? "block" : "none";
      });
    },
  };

  // Navigation management module
  const navigationManager = {
    async expandAll() {
      const sections = document.querySelectorAll(".benchmark-set");
      const updates = [];

      for (const section of sections) {
        if (section.classList.contains("no-data")) continue;

        const category = section.getAttribute("data-category");
        state.expandedSections.add(category);
        if (section.classList.contains("collapsed")) {
          updates.push(() => section.classList.remove("collapsed"));
        }
      }

      utils.batchDOMUpdates(updates);
      urlManager.updateParams({ expanded: "true" });

      // Load charts for all expanded sections
      for (const section of sections) {
        if (!section.classList.contains("no-data")) {
          const category = section.getAttribute("data-category");
          await chartLoader.loadChartsForSection(category);
        }
      }
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

    async focusOnGroup(groupName) {
      const targetSection = document.querySelector(`[data-category="${groupName}"]`);
      if (targetSection) {
        if (utils.isMobile()) {
          domElements.sidebar.classList.remove("active");
        }

        const targetId = targetSection.querySelector(".benchmark-title").id;
        const targetElement = document.getElementById(targetId);
        const headerHeight = document.querySelector(".sticky-header").offsetHeight;
        const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition = elementPosition - headerHeight - CONFIG.SCROLL_OFFSET_PADDING;

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
      domElements.backToTop.classList.toggle("visible", scrollY > CONFIG.BACK_TO_TOP_THRESHOLD);

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
    initializeControls() {
      const elementIds = [
        "menu-toggle", "sidebar", "sidebar-close", "sidebar-overlay",
        "expand-all", "collapse-all", "grid-view", "list-view",
        "category-filter", "clear-filter", "search-filter",
        "back-to-top", "modal-close", "chart-modal", "main", "toc",
      ];

      elementIds.forEach((id) => {
        const camelCaseId = id.replace(/-(.)/g, (_match, char) => char.toUpperCase());
        domElements[camelCaseId] = document.getElementById(id);
      });

      if (window.innerWidth >= 1200) {
        const sidebarPref = localStorage.getItem("sidebarCollapsed");
        if (domElements.sidebar) {
          if (sidebarPref === null) {
            localStorage.setItem("sidebarCollapsed", "true");
          }

          if (sidebarPref === "false") {
            domElements.sidebar.classList.remove("collapsed");
            domElements.sidebar.classList.add("open");
          } else {
            domElements.sidebar.classList.add("collapsed");
            domElements.sidebar.classList.remove("open");
          }
        }
      }

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
          { rootMargin: CONFIG.CHART_OBSERVER_MARGIN }
        );

        const stickyObserver = new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              const stickyContainer = entry.target.querySelector(".sticky-header-container");
              if (stickyContainer) {
                if (entry.intersectionRatio < 1) {
                  stickyContainer.classList.add("is-stuck");
                } else {
                  stickyContainer.classList.remove("is-stuck");
                }
              }
            });
          },
          { threshold: [1], rootMargin: "-72px 0px 0px 0px" }
        );

        setTimeout(() => {
          document.querySelectorAll(".sticky-header-wrapper").forEach((wrapper) => {
            stickyObserver.observe(wrapper);
          });
        }, 100);
      }

      window.chartObserver = chartObserver;
      zoomSync.init(state, utils);
      this.setupEventListeners();
    },

    setupEventListeners() {
      domElements.menuToggle.addEventListener("click", () => {
        const isDesktop = window.innerWidth >= 1200;
        if (isDesktop) {
          domElements.sidebar.classList.toggle("collapsed");
          domElements.sidebar.classList.toggle("open");
          const isCollapsed = domElements.sidebar.classList.contains("collapsed");
          localStorage.setItem("sidebarCollapsed", isCollapsed.toString());
        } else {
          domElements.sidebar.classList.toggle("active");
        }
      });

      domElements.sidebarClose.addEventListener("click", () => {
        const isDesktop = window.innerWidth >= 1200;
        if (isDesktop) {
          domElements.sidebar.classList.add("collapsed");
          domElements.sidebar.classList.remove("open");
          localStorage.setItem("sidebarCollapsed", "true");
        } else {
          domElements.sidebar.classList.remove("active");
        }
      });

      domElements.sidebarOverlay.addEventListener("click", () => {
        const isDesktop = window.innerWidth >= 1200;
        if (isDesktop) {
          domElements.sidebar.classList.add("collapsed");
          domElements.sidebar.classList.remove("open");
          localStorage.setItem("sidebarCollapsed", "true");
        } else {
          domElements.sidebar.classList.remove("active");
        }
      });

      domElements.expandAll.addEventListener("click", () => navigationManager.expandAll());
      domElements.collapseAll.addEventListener("click", () => navigationManager.collapseAll());

      domElements.gridView.addEventListener("click", () => ui.setView("grid"));
      domElements.listView.addEventListener("click", () => ui.setView("list"));

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

      const throttledScroll = utils.throttle(
        () => navigationManager.handleScroll(),
        CONFIG.THROTTLE_SCROLL
      );
      window.addEventListener("scroll", throttledScroll);

      domElements.backToTop.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });

      domElements.modalClose.addEventListener("click", () => chartManager.closeModal());
      domElements.chartModal.addEventListener("click", (e) => {
        if (e.target.id === "chart-modal") {
          chartManager.closeModal();
        }
      });

      document.addEventListener("click", (e) => {
        if (
          utils.isMobile() &&
          !domElements.sidebar.contains(e.target) &&
          !domElements.menuToggle.contains(e.target) &&
          domElements.sidebar.classList.contains("active")
        ) {
          domElements.sidebar.classList.remove("active");
        }
      });

      const debouncedResize = utils.debounce(() => {
        chartManager.updateChartsForResize();

        const isDesktop = window.innerWidth >= 1200;
        const wasDesktop = state.lastWindowWidth >= 1200;

        if (wasDesktop && !isDesktop) {
          domElements.sidebar.classList.remove("collapsed");
          domElements.sidebar.classList.remove("active");
        } else if (!wasDesktop && isDesktop) {
          domElements.sidebar.classList.remove("active");
          const sidebarPref = localStorage.getItem("sidebarCollapsed");
          if (sidebarPref === null) {
            localStorage.setItem("sidebarCollapsed", "true");
          }

          if (sidebarPref === "false") {
            domElements.sidebar.classList.remove("collapsed");
            domElements.sidebar.classList.add("open");
          } else {
            domElements.sidebar.classList.add("collapsed");
            domElements.sidebar.classList.remove("open");
          }
        }

        state.lastWindowWidth = window.innerWidth;
      }, CONFIG.RESIZE_DEBOUNCE);

      window.addEventListener("resize", debouncedResize);
    },
  };

  // Main initialization
  return async function initAndRender(keptGroups) {
    try {
      window.state = state;
      window.domElements = domElements;
      window.zoomSync = zoomSync;
      window.utils = utils;
      window.chartLoader = chartLoader;

      const main = document.getElementById("main");
      const toc = document.getElementById("toc");

      // Show loading
      main.innerHTML = `
        <div class="loading-indicator">
          <div class="spinner"></div>
          <p>Loading benchmark metadata...</p>
        </div>
      `;

      // Fetch metadata
      const metadata = await api.fetchMetadata();
      state.metadata = metadata;

      // Calculate default start commit index (last DEFAULT_COMMIT_RANGE commits)
      const totalCommits = metadata.commits?.length || 0;
      state.defaultStartCommitIndex = Math.max(0, totalCommits - DEFAULT_COMMIT_RANGE);

      // Store filter settings for each group
      if (keptGroups) {
        keptGroups.forEach(([name, settings]) => {
          state.groupFilterSettings.set(name, settings);
        });
      }

      // Clear loading indicator
      main.innerHTML = '';

      // Render sections from metadata
      const groupsToRender = keptGroups
        ? keptGroups.map(([name, settings]) => ({ name, settings }))
        : Object.keys(metadata.groups).map(name => ({ name, settings: {} }));

      for (const { name, settings } of groupsToRender) {
        const groupMetadata = metadata.groups[name];
        if (!groupMetadata) continue;

        const { section } = ui.createBenchmarkSectionFromMetadata(name, groupMetadata, settings);

        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';

        main.appendChild(section);

        requestAnimationFrame(() => {
          section.style.opacity = '1';
          section.style.transform = 'translateY(0)';
        });

        // Create TOC entry
        const tocLi = document.createElement("li");
        const tocLink = document.createElement("a");
        const h1id = name.replace(/\s+/g, "_");
        tocLink.href = "#" + h1id;
        tocLink.innerHTML = name;
        tocLink.onclick = async (e) => {
          e.preventDefault();

          const targetSection = document.querySelector(`[data-category="${name}"]`);
          if (targetSection && targetSection.classList.contains("collapsed") && !targetSection.classList.contains("no-data")) {
            state.expandedSections.add(name);
            targetSection.classList.remove("collapsed");
            await chartLoader.loadChartsForSection(name);
          }

          if (utils.isMobile()) {
            domElements.sidebar.classList.remove("active");
          }

          const targetElement = document.getElementById(h1id);
          const headerHeight = document.querySelector(".sticky-header").offsetHeight;
          const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
          const offsetPosition = elementPosition - headerHeight - CONFIG.SCROLL_OFFSET_PADDING;

          window.scrollTo({
            top: offsetPosition,
            behavior: "smooth",
          });

          navigationManager.updateActiveNavItem(h1id);
        };
        tocLi.appendChild(tocLink);
        toc.appendChild(tocLi);

        // Update zoom sync cache for this category
        window.zoomSync.updateCacheForCategory(name);
      }

      initializer.initializeControls();
      urlManager.initializeFromParams();

      window.addEventListener('beforeunload', () => {
        window.zoomSync.cleanup();
      });

    } catch (error) {
      console.error("Failed to load benchmark data:", error);
      const main = document.getElementById("main");
      main.innerHTML = `
        <div class="loading-indicator">
          <p style="color: red;">Failed to load benchmark data. Please try refreshing the page.</p>
          <p>${error.message}</p>
        </div>
      `;
    }
  };
})();
