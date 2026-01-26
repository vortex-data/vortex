"use strict";

// Import dependencies from other modules
import { CONFIG } from "./config.js";
import { utils } from "./utils.js";

// Chart management module
// This module requires the following global dependencies to be available:
// - window.state: Application state object
// - window.domElements: DOM elements cache
// - window.chartObserver: IntersectionObserver for lazy loading
// - window.zoomSync: Zoom synchronization module
// - window.debouncedSyncZoom: Debounced zoom synchronization function
export const chartManager = {
  remapNames(benchName) {
    const remappings = {
      "COMPRESS TIME": "VORTEX WRITE TIME (COMPRESSION)",
      "DECOMPRESS TIME": "VORTEX SCAN TIME (DECOMPRESSION)",
      "PARQUET RS-ZSTD COMPRESS TIME": "PARQUET WRITE TIME (COMPRESSION)",
      "PARQUET RS-ZSTD DECOMPRESS TIME": "PARQUET SCAN TIME (DECOMPRESSION)",
      "LANCE COMPRESS TIME": "LANCE WRITE TIME (COMPRESSION)",
      "LANCE DECOMPRESS TIME": "LANCE SCAN TIME (DECOMPRESSION)",
      "VORTEX SIZE": "VORTEX SIZE",
      "PARQUET-ZSTD SIZE": "PARQUET SIZE",
      "LANCE SIZE": "LANCE SIZE",
      "VORTEX:RAW SIZE": "VORTEX vs RAW SIZE RATIO",
      "VORTEX:PARQUET-ZSTD SIZE": "VORTEX vs PARQUET SIZE RATIO",
      "VORTEX:LANCE SIZE": "VORTEX vs LANCE SIZE RATIO",
      "VORTEX:PARQUET-ZSTD RATIO COMPRESS TIME":
        "VORTEX vs PARQUET WRITE TIME RATIO",
      "VORTEX:PARQUET-ZSTD RATIO DECOMPRESS TIME":
        "VORTEX vs PARQUET SCAN TIME RATIO",
      "VORTEX:LANCE RATIO COMPRESS TIME": "VORTEX vs LANCE WRITE TIME RATIO",
      "VORTEX:LANCE RATIO DECOMPRESS TIME": "VORTEX vs LANCE SCAN TIME RATIO",
    };
    return remappings[benchName] || benchName;
  },

  createChartContainer(name, benchName, index) {
    const container = document.createElement("div");
    container.className = "chart-container fade-in";
    container.setAttribute("data-benchmark", name);
    container.setAttribute("data-chart", benchName);

    const header = document.createElement("div");
    header.className = "chart-header";

    const title = document.createElement("h3");
    title.className = "chart-title";
    title.textContent = this.remapNames(benchName);

    const actions = document.createElement("div");
    actions.className = "chart-actions";

    // Create zoom/pan controls
    const zoomControls = document.createElement("div");
    zoomControls.className = "chart-zoom-controls";

    const chartKey = `${name}-${index}`;
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
      this.goToStart(name, index), "go-start");
    const panLeftBtn = createControlBtn("«", "Pan left", () =>
      this.panChart(name, index, -0.5), "pan-left");
    const panRightBtn = createControlBtn("»", "Pan right", () =>
      this.panChart(name, index, 0.5), "pan-right");
    const goToEndBtn = createControlBtn("»|", "Go to latest", () =>
      this.goToEnd(name, index), "go-end");
    const zoomInBtn = createControlBtn("+", "Zoom in", () =>
      this.zoomChart(name, index, 0.5), "zoom-in");
    const zoomOutBtn = createControlBtn("−", "Zoom out", () =>
      this.zoomChart(name, index, 2), "zoom-out");

    zoomControls.appendChild(goToStartBtn);
    zoomControls.appendChild(panLeftBtn);
    zoomControls.appendChild(zoomInBtn);
    zoomControls.appendChild(zoomOutBtn);
    zoomControls.appendChild(panRightBtn);
    zoomControls.appendChild(goToEndBtn);

    const fullscreenBtn = document.createElement("button");
    fullscreenBtn.className = "chart-action-btn";
    fullscreenBtn.textContent = "Fullscreen";
    fullscreenBtn.onclick = () =>
      chartManager.openModal(name, benchName, index);

    actions.appendChild(zoomControls);
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
    if (isMobile && window.chartObserver) {
      container.chartData = chartConfig;
      window.chartObserver.observe(container);
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
          // Convert object with numeric keys back to array
          let benchesArray;
          if (Array.isArray(benches)) {
            benchesArray = benches;
          } else if (benches && typeof benches === "object") {
            // Convert object with numeric keys to array
            const maxIndex = Math.max(
              ...Object.keys(benches)
                .map((k) => parseInt(k, 10))
                .filter((n) => !isNaN(n))
            );
            benchesArray = new Array(maxIndex + 1);
            for (let i = 0; i <= maxIndex; i++) {
              benchesArray[i] = benches[i] || null;
            }
          } else {
            benchesArray = [];
          }
          const limitedData = benchesArray.slice(startIndex);
          return {
            label: renamedName,
            data: limitedData.map((b) => (b ? b.value : null)),
            borderColor: color,
            backgroundColor: color + "60", // Add alpha for #rrggbbaa
            hidden:
              (hiddenDatasets !== undefined && hiddenDatasets.has(name)) ||
              name.toLowerCase().startsWith("wide table cols"),
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
    window.state.chartInstances.set(chartKey, { chart, data, options });

    // Update navigation button states for initial load
    this.updateNavigationButtons(chartKey);

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
      animation: false,
      scales: {
        x: {
          title: {
            display: false,
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
        x2: this.createDateAxis(limitedCommits, isMobile, {
          min: isMobile
            ? 0
            : Math.max(0, dataset.commits.length - CONFIG.DEFAULT_VISIBLE_COMMITS),
          max: isMobile
            ? limitedCommits.length - 1
            : undefined,
        }),
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

  // Create a secondary x-axis that shows human-readable dates in a faded style
  createDateAxis(limitedCommits, isMobile, initialRange = {}) {
    // Format date in a human-readable way
    const formatDate = (timestamp) => {
      if (!timestamp) return '';
      const date = new Date(timestamp);
      const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      return `${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
    };

    return {
      type: 'category',
      position: 'bottom',
      offset: true,
      labels: limitedCommits.map(c => c?.timestamp ? formatDate(c.timestamp) : ''),
      min: initialRange.min,
      max: initialRange.max,
      grid: {
        display: false,
        drawOnChartArea: false,
      },
      border: {
        display: false,
      },
      ticks: {
        display: !isMobile,
        color: 'rgba(128, 128, 128, 0.5)', // Faded gray color
        font: {
          size: 10,
          style: 'italic',
        },
        padding: 2,
        maxRotation: 0,
        autoSkip: false,
        // Show only 5 labels: first, last, and 3 evenly spaced in between
        callback: function(value, index, ticks) {
          const totalTicks = ticks.length;
          if (totalTicks <= 5) {
            return this.getLabelForValue(index);
          }

          // Always show first and last
          if (index === 0 || index === totalTicks - 1) {
            return this.getLabelForValue(index);
          }

          // Show 3 evenly spaced labels in between (at 25%, 50%, 75%)
          const step = (totalTicks - 1) / 4;
          for (let i = 1; i <= 3; i++) {
            const targetIndex = Math.round(step * i);
            if (index === targetIndex) {
              return this.getLabelForValue(index);
            }
          }

          return null; // Hide this tick
        },
      },
    };
  },

  // Sync the x2 (date) axis with the x (commit) axis
  syncDateAxis(chart) {
    if (!chart?.scales?.x || !chart?.scales?.x2) return;

    const xScale = chart.scales.x;
    chart.options.scales.x2.min = xScale.min;
    chart.options.scales.x2.max = xScale.max;
    chart.update('none');
  },

  createPlugins(categoryName, isMobile, limitedCommits, index) {
    const chartKey = `${categoryName}-${index}`;
    const self = this;

    // Debounced zoom/pan handler to fetch new data
    let zoomPanTimeout = null;
    const handleZoomPanComplete = (context) => {
      const chart = context.chart;

      // Immediately sync the date axis
      self.syncDateAxis(chart);

      // Clear previous timeout
      if (zoomPanTimeout) {
        clearTimeout(zoomPanTimeout);
      }

      // Debounce the data fetch
      zoomPanTimeout = setTimeout(() => {
        const xScale = chart.scales.x;

        // Get the visible range indices
        const visibleMin = Math.floor(xScale.min);
        const visibleMax = Math.ceil(xScale.max);

        // Get chart instance info
        const chartInstance = window.state?.chartInstances?.get(chartKey);
        if (!chartInstance) return;

        // Map local indices to global commit indices
        const originalData = chartInstance.originalData;
        if (!originalData?.requestedRange) return;

        // Calculate global indices based on the original request range
        const globalStartIndex = originalData.requestedRange.startIndex + visibleMin;
        const globalEndIndex = originalData.requestedRange.startIndex + visibleMax;

        // Check if we need to fetch more data (zooming out or panning beyond current range)
        const currentRangeStart = originalData.requestedRange.startIndex;
        const currentRangeEnd = originalData.requestedRange.endIndex;

        // Only fetch if zooming to a different downsample level would help
        // or if we're showing data at boundaries that might benefit from more context
        const currentLength = chart.data.labels.length;
        const visibleLength = visibleMax - visibleMin + 1;

        // If showing most of the data and it's downsampled, might want to fetch more detail
        if (originalData.downsampleLevel !== '1x' && visibleLength < currentLength * 0.5) {
          // Zooming in - might get better resolution
          if (window.chartLoader?.refreshChartData) {
            window.chartLoader.refreshChartData(chartKey, globalStartIndex, globalEndIndex);
          }
        }
      }, 500); // 500ms debounce
    };

    return {
      zoom: {
        zoom: {
          wheel: {
            enabled: false, // Disable wheel zoom - use button controls instead
            speed: CONFIG.ZOOM_SPEED,
            modifierKey: null,
          },
          mode: "x",
          drag: {
            enabled: !isMobile,
            backgroundColor: "rgba(89, 113, 253, 0.1)",
          },
          onZoomComplete: handleZoomPanComplete,
        },
        pan: {
          enabled: false, // Disable drag panning - use button controls instead
          mode: "x",
          modifierKey: null,
          onPanComplete: handleZoomPanComplete,
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
      const datasetLabel = dataset.label;

      // Toggle the clicked dataset
      dataset.hidden = !dataset.hidden;

      // Find the benchmark group name from the chart canvas
      const canvas = chart.canvas;
      const container = canvas.closest(".chart-container");
      const benchmarkGroup = container?.getAttribute("data-benchmark");

      if (benchmarkGroup) {
        // Synchronize across all charts in the same benchmark group
        window.state.chartInstances.forEach((chartData, key) => {
          if (key.startsWith(benchmarkGroup + "-")) {
            const otherChart = chartData.chart;
            if (otherChart !== chart) {
              // Find dataset with matching label
              otherChart.data.datasets.forEach((ds) => {
                if (ds.label === datasetLabel) {
                  ds.hidden = dataset.hidden;
                }
              });
              otherChart.update("none"); // Update without animation
            }
          }
        });
      }

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
    const modal = window.domElements.chartModal;
    const modalCanvas = document.getElementById("modal-chart");

    const chartKey = `${benchmarkName}-${index}`;
    const originalChart = window.state.chartInstances.get(chartKey);
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
    const modal = window.domElements.chartModal;
    if (modal.modalChart) {
      modal.modalChart.destroy();
      modal.modalChart = null;
    }
    modal.classList.remove("active");
  },

  // Pan chart by a percentage of the visible range
  panChart(categoryName, index, direction) {
    const chartKey = `${categoryName}-${index}`;
    const chartData = window.state.chartInstances.get(chartKey);
    if (!chartData?.chart) return;

    const chart = chartData.chart;
    const originalData = chartData.originalData;

    // Get total commits from metadata (the full global range)
    const totalGlobalCommits = window.state?.metadata?.commits?.length ||
                               originalData?.originalLength ||
                               chart.data.labels.length;

    const xScale = chart.scales.x;
    const localMin = xScale.min;
    const localMax = xScale.max;
    const localRange = localMax - localMin;
    const localDataLength = chart.data.labels.length;

    // Get the actual global range from requestedRange
    const requestedRange = originalData?.requestedRange || { startIndex: 0, endIndex: localDataLength - 1, length: localDataLength };
    const globalRangeStart = requestedRange.startIndex;
    const globalRangeLength = requestedRange.length || (requestedRange.endIndex - globalRangeStart + 1);

    // Calculate the proportion of the loaded data we're viewing
    const viewProportion = localRange / Math.max(1, localDataLength - 1);
    const viewStartProportion = localMin / Math.max(1, localDataLength - 1);

    // Map to actual global indices
    const currentGlobalViewLength = globalRangeLength * viewProportion;
    const currentGlobalMin = globalRangeStart + (globalRangeLength * viewStartProportion);
    const currentGlobalMax = currentGlobalMin + currentGlobalViewLength;

    // Calculate pan amount in global terms
    const panAmount = currentGlobalViewLength * direction;

    // Calculate new global position
    let newGlobalMin = currentGlobalMin + panAmount;
    let newGlobalMax = currentGlobalMax + panAmount;

    // Clamp to global bounds
    if (newGlobalMin < 0) {
      newGlobalMax -= newGlobalMin;
      newGlobalMin = 0;
    }
    if (newGlobalMax > totalGlobalCommits - 1) {
      newGlobalMin -= (newGlobalMax - (totalGlobalCommits - 1));
      newGlobalMax = totalGlobalCommits - 1;
    }
    newGlobalMin = Math.max(0, Math.floor(newGlobalMin));
    newGlobalMax = Math.min(totalGlobalCommits - 1, Math.ceil(newGlobalMax));

    // Check if we need to load more data
    const currentGlobalStart = requestedRange.startIndex;
    const currentGlobalEnd = requestedRange.startIndex + chart.data.labels.length - 1;

    if (newGlobalMin < currentGlobalStart || newGlobalMax > currentGlobalEnd) {
      // Need to load more data - trigger refresh with new range
      this.triggerDataRefreshWithRange(chartKey, newGlobalMin, newGlobalMax);
    } else {
      // Data is already loaded, just adjust the view
      const newLocalMin = newGlobalMin - requestedRange.startIndex;
      const newLocalMax = newGlobalMax - requestedRange.startIndex;

      chart.options.scales.x.min = newLocalMin;
      chart.options.scales.x.max = newLocalMax;
      this.syncDateAxis(chart);

      // Trigger data refresh for potential resolution change
      this.triggerDataRefresh(chartKey, chart);
    }

    // Update navigation button states
    this.updateNavigationButtons(chartKey);
  },

  // Zoom chart by a factor (< 1 = zoom in, > 1 = zoom out)
  zoomChart(categoryName, index, factor) {
    const chartKey = `${categoryName}-${index}`;
    const chartData = window.state.chartInstances.get(chartKey);
    if (!chartData?.chart) return;

    const chart = chartData.chart;
    const originalData = chartData.originalData;

    // Get total commits from metadata (the full global range)
    const totalGlobalCommits = window.state?.metadata?.commits?.length ||
                               originalData?.originalLength ||
                               chart.data.labels.length;

    // Current local range (in downsampled indices)
    const xScale = chart.scales.x;
    const localMin = xScale.min;
    const localMax = xScale.max;
    const localRange = localMax - localMin;
    const localDataLength = chart.data.labels.length;

    // Get the actual global range from requestedRange
    const requestedRange = originalData?.requestedRange || { startIndex: 0, endIndex: localDataLength - 1, length: localDataLength };
    const globalRangeStart = requestedRange.startIndex;
    const globalRangeEnd = requestedRange.endIndex;
    const globalRangeLength = requestedRange.length || (globalRangeEnd - globalRangeStart + 1);

    // Calculate the proportion of the loaded data we're viewing
    const viewProportion = localRange / Math.max(1, localDataLength - 1);
    const viewStartProportion = localMin / Math.max(1, localDataLength - 1);

    // Map to actual global indices
    const currentGlobalViewLength = globalRangeLength * viewProportion;
    const globalMin = globalRangeStart + (globalRangeLength * viewStartProportion);
    const globalMax = globalMin + currentGlobalViewLength;
    const globalCenter = (globalMin + globalMax) / 2;

    // Calculate new global range by scaling the ACTUAL global range we're viewing
    const newGlobalRange = currentGlobalViewLength * factor;

    // Minimum range check
    const minRange = Math.min(CONFIG.MIN_VISIBLE_COMMITS, totalGlobalCommits);
    if (newGlobalRange < minRange && factor < 1) return;

    let newGlobalMin = globalCenter - newGlobalRange / 2;
    let newGlobalMax = globalCenter + newGlobalRange / 2;

    // Clamp to global bounds
    if (newGlobalMin < 0) {
      newGlobalMax -= newGlobalMin;
      newGlobalMin = 0;
    }
    if (newGlobalMax > totalGlobalCommits - 1) {
      newGlobalMin -= (newGlobalMax - (totalGlobalCommits - 1));
      newGlobalMax = totalGlobalCommits - 1;
    }
    newGlobalMin = Math.max(0, Math.floor(newGlobalMin));
    newGlobalMax = Math.min(totalGlobalCommits - 1, Math.ceil(newGlobalMax));

    // Check if we need to load more data (zooming out beyond loaded range)
    const currentGlobalStart = requestedRange.startIndex;
    const currentGlobalEnd = requestedRange.startIndex + chart.data.labels.length - 1;

    if (newGlobalMin < currentGlobalStart || newGlobalMax > currentGlobalEnd) {
      // Need to load more data - trigger refresh with expanded range
      this.triggerDataRefreshWithRange(chartKey, newGlobalMin, newGlobalMax);
    } else {
      // Data is already loaded, just adjust the view
      const newLocalMin = newGlobalMin - requestedRange.startIndex;
      const newLocalMax = newGlobalMax - requestedRange.startIndex;

      chart.options.scales.x.min = newLocalMin;
      chart.options.scales.x.max = newLocalMax;
      this.syncDateAxis(chart);

      // Still trigger refresh in case we want higher resolution data
      this.triggerDataRefresh(chartKey, chart);
    }

    // Update navigation button states
    this.updateNavigationButtons(chartKey);
  },

  // Go to the end of the chart (latest commits) - uses global range
  goToEnd(categoryName, index) {
    const chartKey = `${categoryName}-${index}`;
    const chartData = window.state.chartInstances.get(chartKey);
    if (!chartData?.chart) return;

    const chart = chartData.chart;
    const originalData = chartData.originalData;

    // Get total commits from metadata (the full global range)
    const totalGlobalCommits = window.state?.metadata?.commits?.length ||
                               originalData?.originalLength ||
                               chart.data.labels.length;

    const xScale = chart.scales.x;
    const visibleRange = xScale.max - xScale.min;

    // Calculate new global range at the end
    const newGlobalMax = totalGlobalCommits - 1;
    const newGlobalMin = Math.max(0, newGlobalMax - visibleRange);

    // Check if we need to load more data
    const requestedRange = originalData?.requestedRange || { startIndex: 0 };
    const currentGlobalEnd = requestedRange.startIndex + chart.data.labels.length - 1;

    if (newGlobalMax > currentGlobalEnd) {
      // Need to load more data
      this.triggerDataRefreshWithRange(chartKey, newGlobalMin, newGlobalMax);
    } else {
      // Data is already loaded, just adjust the view
      const newLocalMin = newGlobalMin - requestedRange.startIndex;
      const newLocalMax = newGlobalMax - requestedRange.startIndex;

      chart.options.scales.x.min = Math.max(0, newLocalMin);
      chart.options.scales.x.max = Math.min(chart.data.labels.length - 1, newLocalMax);
      this.syncDateAxis(chart);

      this.triggerDataRefresh(chartKey, chart);
    }

    // Update navigation button states
    this.updateNavigationButtons(chartKey);
  },

  // Go to the start of the chart (oldest commits) - uses global range
  goToStart(categoryName, index) {
    const chartKey = `${categoryName}-${index}`;
    const chartData = window.state.chartInstances.get(chartKey);
    if (!chartData?.chart) return;

    const chart = chartData.chart;
    const originalData = chartData.originalData;

    // Get total commits from metadata (the full global range)
    const totalGlobalCommits = window.state?.metadata?.commits?.length ||
                               originalData?.originalLength ||
                               chart.data.labels.length;

    const xScale = chart.scales.x;
    const visibleRange = xScale.max - xScale.min;

    // Calculate new global range at the start
    const newGlobalMin = 0;
    const newGlobalMax = Math.min(totalGlobalCommits - 1, visibleRange);

    // Check if we need to load more data
    const requestedRange = originalData?.requestedRange || { startIndex: 0 };

    if (newGlobalMin < requestedRange.startIndex) {
      // Need to load more data
      this.triggerDataRefreshWithRange(chartKey, newGlobalMin, newGlobalMax);
    } else {
      // Data is already loaded, just adjust the view
      const newLocalMin = newGlobalMin - requestedRange.startIndex;
      const newLocalMax = newGlobalMax - requestedRange.startIndex;

      chart.options.scales.x.min = Math.max(0, newLocalMin);
      chart.options.scales.x.max = Math.min(chart.data.labels.length - 1, newLocalMax);
      this.syncDateAxis(chart);

      this.triggerDataRefresh(chartKey, chart);
    }

    // Update navigation button states
    this.updateNavigationButtons(chartKey);
  },

  // Trigger data refresh after zoom/pan
  triggerDataRefresh(chartKey, chart) {
    const chartInstance = window.state?.chartInstances?.get(chartKey);
    if (!chartInstance?.originalData?.requestedRange) return;

    const xScale = chart.scales.x;
    const visibleMin = Math.floor(xScale.min);
    const visibleMax = Math.ceil(xScale.max);

    const originalData = chartInstance.originalData;
    const globalStartIndex = originalData.requestedRange.startIndex + visibleMin;
    const globalEndIndex = originalData.requestedRange.startIndex + visibleMax;

    this.triggerDataRefreshWithRange(chartKey, globalStartIndex, globalEndIndex);
  },

  // Trigger data refresh with a specific global range
  triggerDataRefreshWithRange(chartKey, globalStartIndex, globalEndIndex) {
    // Debounce the refresh
    if (this._refreshTimeout) {
      clearTimeout(this._refreshTimeout);
    }
    this._refreshTimeout = setTimeout(() => {
      if (window.chartLoader?.refreshChartData) {
        window.chartLoader.refreshChartData(chartKey, globalStartIndex, globalEndIndex);
      }
    }, 300);
  },

  // Update navigation button states (disable at edges)
  updateNavigationButtons(chartKey) {
    const chartData = window.state?.chartInstances?.get(chartKey);
    if (!chartData?.chart) return;

    const chart = chartData.chart;
    const originalData = chartData.originalData;

    // Get global commits from metadata
    const globalCommits = window.state?.metadata?.commits;
    if (!globalCommits || globalCommits.length === 0) return;

    const firstGlobalCommitId = globalCommits[0]?.id?.slice(0, 7);
    const lastGlobalCommitId = globalCommits[globalCommits.length - 1]?.id?.slice(0, 7);

    // Get current view's visible commit labels
    const xScale = chart.scales.x;
    const localMin = Math.floor(xScale.min);
    const localMax = Math.ceil(xScale.max);

    const labels = chart.data.labels;
    const visibleFirstCommit = labels[localMin];
    const visibleLastCommit = labels[localMax];

    // Check if the visible edges match the global first/last commits
    const atStart = visibleFirstCommit === firstGlobalCommitId;
    const atEnd = visibleLastCommit === lastGlobalCommitId;

    // Find and update buttons
    const buttons = document.querySelectorAll(`button[data-chart-key="${chartKey}"]`);
    buttons.forEach(btn => {
      const action = btn.getAttribute("data-action");
      if (action === "go-start" || action === "pan-left") {
        btn.disabled = atStart;
      } else if (action === "go-end" || action === "pan-right") {
        btn.disabled = atEnd;
      }
    });
  },

  cleanupCharts() {
    window.state.chartInstances.forEach((chartData) => {
      if (chartData?.chart) {
        chartData.chart.destroy();
      }
    });
    window.state.chartInstances.clear();
    window.state.charts = [];
  },

  updateChartsForResize() {
    // Prevent multiple simultaneous resize operations
    if (window.state.isResizing) return;
    window.state.isResizing = true;

    const currentIsMobile = utils.isMobile();
    const wasDesktop = window.state.lastWindowWidth > CONFIG.MOBILE_BREAKPOINT;
    const isDesktop = window.innerWidth > CONFIG.MOBILE_BREAKPOINT;
    const crossedThreshold =
      (wasDesktop && !isDesktop) || (!wasDesktop && isDesktop);

    // Update window width immediately
    window.state.lastWindowWidth = window.innerWidth;

    if (!crossedThreshold) {
      // Simple resize - just update all charts
      requestAnimationFrame(() => {
        window.state.chartInstances.forEach((chartData) => {
          if (chartData?.chart) {
            chartData.chart.resize();
            chartData.chart.update("none");
          }
        });
        window.state.isResizing = false;
      });
      return;
    }

    // For threshold crossing, update chart options
    requestAnimationFrame(() => {
      // Update all charts
      window.state.chartInstances.forEach((chartData, key) => {
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
          chart.options.aspectRatio = currentIsMobile ? 1.5 : 2;
          chart.options.pointStyle = currentIsMobile ? false : "crossRot";
          chart.options.elements.point.radius = currentIsMobile ? 0 : 3;

          // Update zoom settings
          if (chart.options.plugins.zoom) {
            const zoomEnabled = !currentIsMobile;
            chart.options.plugins.zoom.zoom.wheel.enabled = zoomEnabled;
            chart.options.plugins.zoom.zoom.pinch.enabled = zoomEnabled;
            chart.options.plugins.zoom.zoom.drag.enabled = zoomEnabled;
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
        window.state.chartInstances.forEach((chartData) => {
          if (chartData?.chart) {
            chartData.chart.options.animation.duration = currentIsMobile
              ? 0
              : CONFIG.ANIMATION_DURATION;
          }
        });
      }, 100);

      // Recreate debounced sync zoom with new delay
      if (crossedThreshold && window.zoomSync) {
        window.zoomSync.recreateDebouncedSync(window.state, utils);
      }

      window.state.isResizing = false;
    });
  },
};
