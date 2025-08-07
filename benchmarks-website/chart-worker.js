"use strict";

// Chart preparation web worker
// Handles chart data preparation and processing for Chart.js

// Configuration constants needed in worker context
const CONFIG = {
  MOBILE_MAX_DATA_POINTS: 50,
  DEFAULT_VISIBLE_COMMITS: 20,
  MIN_VISIBLE_COMMITS: 5,
  COMPRESS_THROUGHPUT_MAX: 4000,
  DECOMPRESS_THROUGHPUT_MAX: 8000,
  ANIMATION_DURATION: 300,
  ZOOM_SPEED: 0.1,
  MOBILE_BREAKPOINT: 768,
};

// Color mapping for consistent colors across charts
const SERIES_COLOR_MAP = {
  "vortex-nvme": "#4f46e5",
  "parquet-nvme": "#dc2626", 
  "vortex": "#4f46e5",
  "parquet": "#dc2626",
  "datafusion:vortex": "#4f46e5",
  "datafusion:parquet": "#dc2626",
  "datafusion:in-memory-arrow": "#059669",
  "duckdb:vortex": "#7c3aed",
  "duckdb:parquet": "#ea580c",
  "duckdb:duckdb": "#0891b2",
};

const FALLBACK_PALETTE = [
  "#4f46e5", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2",
  "#be185d", "#047857", "#7c2d12", "#1e40af", "#991b1b", "#166534"
];

// Utility functions needed in worker context
function stringToColor(str) {
  if (SERIES_COLOR_MAP[str]) {
    return SERIES_COLOR_MAP[str];
  }

  // Simple hash function for consistent colors
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  
  const index = Math.abs(hash) % FALLBACK_PALETTE.length;
  return FALLBACK_PALETTE[index];
}

// Deserialize Map from serialized array format
function deserializeMap(serializedMap) {
  const result = new Map();
  for (const [key, value] of serializedMap) {
    if (value.series && Array.isArray(value.series)) {
      value.series = deserializeMap(value.series);
    }
    result.set(key, value);
  }
  return result;
}

// Chart data preparation functions
const chartPreparation = {
  prepareChartData(config) {
    const {
      dataset,
      hiddenDatasets,
      removedDatasets,
      renamedDatasets,
      isMobile,
      maxDataPoints
    } = config;

    // Limit data points for performance
    const actualMaxPoints = isMobile ? 
      Math.min(maxDataPoints || CONFIG.MOBILE_MAX_DATA_POINTS, dataset.commits.length) :
      dataset.commits.length;
    
    const startIndex = Math.max(0, dataset.commits.length - actualMaxPoints);
    const limitedCommits = dataset.commits.slice(startIndex);

    const chartData = {
      labels: limitedCommits.map((commit) => commit.id.slice(0, 7)),
      datasets: Array.from(dataset.series)
        .filter(([name, benches]) => {
          return removedDatasets === undefined || !removedDatasets.has(name);
        })
        .map(([name, benches]) => {
          const renamedName = renamedDatasets === undefined ? 
            name : (renamedDatasets[name] || name);
          const color = stringToColor(renamedName);
          // Convert object with numeric keys back to array
          let benchesArray;
          if (Array.isArray(benches)) {
            benchesArray = benches;
          } else if (benches && typeof benches === 'object') {
            // Convert object with numeric keys to array
            const maxIndex = Math.max(...Object.keys(benches).map(k => parseInt(k, 10)).filter(n => !isNaN(n)));
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
            hidden: (hiddenDatasets !== undefined && hiddenDatasets.has(name)) ||
                   name.toLowerCase().startsWith("wide table cols"),
          };
        }),
    };

    return {
      chartData,
      limitedCommits,
      startIndex
    };
  },

  prepareChartOptions(config) {
    const {
      categoryName,
      benchName,
      dataset,
      limitedCommits,
      isMobile,
      index
    } = config;

    const yAxisScale = this.createYAxisScale(benchName, dataset);

    return {
      responsive: true,
      maintainAspectRatio: false,
      aspectRatio: isMobile ? 1.5 : 2,
      spanGaps: true,
      pointStyle: isMobile ? false : "crossRot",
      resizeDelay: 0,
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
          min: isMobile ? 0 : Math.max(0, dataset.commits.length - CONFIG.DEFAULT_VISIBLE_COMMITS),
          max: isMobile ? limitedCommits.length - 1 : undefined,
        },
        y: yAxisScale,
      },
      plugins: this.createPlugins(categoryName, isMobile, limitedCommits, index),
    };
  },

  createYAxisScale(benchName, dataset) {
    const scale = {
      title: {
        display: true,
        text: dataset.commits.length > 0 ? dataset.unit : "",
      },
      suggestedMin: 0,
      beginAtZero: true,
    };

    if (benchName.includes("COMPRESS") && benchName.includes("THROUGHPUT") && dataset.unit === "MiB/s") {
      scale.suggestedMax = CONFIG.COMPRESS_THROUGHPUT_MAX;
      scale.max = CONFIG.COMPRESS_THROUGHPUT_MAX;
    }

    if (benchName.includes("DECOMPRESS") && benchName.includes("THROUGHPUT") && dataset.unit === "MiB/s") {
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
        },
        pan: {
          enabled: !isMobile,
          mode: "x",
          modifierKey: null,
        },
        limits: {
          x: {
            min: 0,
            max: limitedCommits.length - 1,
            minRange: Math.min(CONFIG.MIN_VISIBLE_COMMITS, limitedCommits.length),
          },
        },
      },
      legend: {
        display: true,
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context) {
            const dataIndex = context.dataIndex;
            const commit = limitedCommits[dataIndex];
            if (!commit) return [];

            return [
              "",
              commit.message.split("\n")[0],
              `${commit.author.name} - ${new Date(commit.timestamp).toLocaleDateString()}`,
            ];
          }
        },
      },
    };
  },

  prepareBatchCharts(charts, isMobile) {
    const results = [];
    let processed = 0;
    
    for (const chartConfig of charts) {
      const result = this.prepareChartConfiguration({
        ...chartConfig,
        isMobile
      });
      
      results.push(result);
      processed++;
      
      // Send progress updates every 10 charts
      if (processed % 10 === 0) {
        self.postMessage({
          type: 'progress',
          progress: (processed / charts.length) * 100,
          message: `Preparing charts: ${processed}/${charts.length}`
        });
      }
    }
    
    return results;
  },

  prepareChartConfiguration(config) {
    const {
      name,
      benchName,
      dataset,
      hiddenDatasets,
      removedDatasets,
      renamedDatasets,
      index,
      isMobile
    } = config;

    // Deserialize the dataset if it's in serialized format
    const deserializedDataset = {
      ...dataset,
      series: dataset.series instanceof Map ? dataset.series : deserializeMap(dataset.series)
    };

    const dataPrep = this.prepareChartData({
      dataset: deserializedDataset,
      hiddenDatasets,
      removedDatasets,
      renamedDatasets,
      isMobile
    });

    const options = this.prepareChartOptions({
      categoryName: name,
      benchName,
      dataset: deserializedDataset,
      limitedCommits: dataPrep.limitedCommits,
      isMobile,
      index
    });

    return {
      name,
      benchName,
      index,
      chartData: dataPrep.chartData,
      options,
      limitedCommits: dataPrep.limitedCommits
    };
  }
};

// Worker message handler
self.addEventListener('message', async function(e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'prepareChart':
        const result = chartPreparation.prepareChartConfiguration(data);
        self.postMessage({
          type: 'chartPrepared',
          result
        });
        break;

      case 'prepareBatchCharts':
        self.postMessage({
          type: 'progress',
          progress: 0,
          message: 'Starting chart preparation...'
        });

        const results = chartPreparation.prepareBatchCharts(data.charts, data.isMobile);
        
        self.postMessage({
          type: 'chartsBatchPrepared',
          results,
          progress: 100,
          message: 'Chart preparation complete!'
        });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error.message,
      stack: error.stack
    });
  }
});