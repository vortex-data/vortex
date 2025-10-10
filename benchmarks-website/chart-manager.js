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
                window.zoomSync.synchronizeZoomForCategory(
                  categoryName,
                  chart,
                  index,
                  true,
                  window.state,
                  window.utils
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
                window.zoomSync.synchronizeZoomForCategory(
                  categoryName,
                  chart,
                  index,
                  false,
                  window.state,
                  window.utils
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
