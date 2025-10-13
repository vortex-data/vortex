"use strict";

import { scoring } from './scoring.js';

/**
 * Base class for all benchmark types.
 * Provides common functionality and template methods for subclasses.
 */
export class BaseBenchmark {
  constructor(config) {
    this.name = config.name;
    this.displayName = config.displayName || config.name;
    this.datasets = config.datasets || new Map();
    this.hiddenDatasets = config.hiddenDatasets || new Set();
    this.removedDatasets = config.removedDatasets || new Set();
    this.renamedDatasets = config.renamedDatasets || {};
    this.keptCharts = config.keptCharts || undefined;
    this.description = config.description || "";
    this.tags = config.tags || [];
    this.unit = config.unit || "ms/iter";
  }

  /**
   * Calculate benchmark scores.
   * Override in subclasses that support scoring.
   */
  calculateScore(benchSet) {
    return null;
  }

  /**
   * Format the calculated score for display.
   * Override in subclasses that support scoring.
   */
  formatScore(score) {
    return "";
  }

  /**
   * Get Chart.js configuration options.
   * Override in subclasses to customize chart appearance.
   */
  getChartOptions() {
    return {
      scales: {
        y: {
          type: 'logarithmic',
          title: {
            display: true,
            text: this.unit
          }
        }
      }
    };
  }

  /**
   * Determine if a specific chart should be shown.
   * Can be overridden to implement custom filtering logic.
   */
  shouldShowChart(chartName) {
    if (!this.keptCharts) return true;
    return this.keptCharts.includes(chartName);
  }

  /**
   * Transform raw benchmark data.
   * Override to implement custom data transformations.
   */
  transformData(data) {
    return data;
  }

  /**
   * Validate benchmark data.
   * Override to implement custom validation logic.
   */
  validateData(data) {
    return true;
  }

  /**
   * Check if this benchmark has any data.
   */
  hasData(benchSet) {
    if (!benchSet || benchSet.size === 0) return false;

    for (const [queryName, queryData] of benchSet.entries()) {
      if (!queryData.series || queryData.series.size === 0) continue;

      for (const [seriesName, seriesData] of queryData.series.entries()) {
        for (let i = 0; i < seriesData.length; i++) {
          if (seriesData[i] && seriesData[i].value !== null && seriesData[i].value !== undefined) {
            return true;
          }
        }
      }
    }

    return false;
  }
}

/**
 * Query benchmark class for Clickbench, TPC-H, TPC-DS, and StatPopGen benchmarks.
 * These benchmarks show query performance and calculate geometric mean scores.
 */
export class QueryBenchmark extends BaseBenchmark {
  constructor(config) {
    super(config);
    this.queryType = config.queryType; // 'clickbench', 'tpch', 'tpcds', 'statpopgen'
    this.scaleFactor = config.scaleFactor;
    this.storage = config.storage; // 'nvme' or 's3'
  }

  /**
   * Calculate geometric mean scores for query benchmarks.
   * Uses the existing scoring logic from scoring.js.
   */
  calculateScore(benchSet) {
    // Use the existing scoring logic
    return scoring.calculateClickBenchScore(benchSet);
  }

  /**
   * Format scores for display.
   */
  formatScore(scores) {
    return scoring.formatScoresSummary(scores);
  }

  /**
   * Get chart options optimized for query benchmarks.
   */
  getChartOptions() {
    return {
      scales: {
        y: {
          type: 'logarithmic',
          title: {
            display: true,
            text: this.unit
          },
          ticks: {
            callback: function(value) {
              // Format logarithmic ticks nicely
              if (value === 1 || value === 10 || value === 100 ||
                  value === 1000 || value === 10000 || value === 100000) {
                return value.toLocaleString();
              }
              return null;
            }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 10
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                label += context.parsed.y.toFixed(2) + ' ms';
              }
              return label;
            }
          }
        }
      }
    };
  }

  /**
   * Check if this is a query benchmark type.
   */
  static isQueryBenchmark(benchmarkName) {
    return scoring.isQueryBenchmark(benchmarkName);
  }
}

/**
 * Compression benchmark class for compression time and size benchmarks.
 * These benchmarks show throughput or size metrics with optional ratio charts.
 */
export class CompressionBenchmark extends BaseBenchmark {
  constructor(config) {
    super(config);
    this.compressionType = config.compressionType; // 'time' or 'size'
    this.showRatios = config.showRatios !== false;
  }

  /**
   * Filter charts based on keptCharts configuration.
   */
  shouldShowChart(chartName) {
    if (!this.keptCharts) return true;
    return this.keptCharts.includes(chartName);
  }

  /**
   * Get chart options optimized for compression benchmarks.
   */
  getChartOptions() {
    const isRatio = this.name.includes("Ratio");
    const yAxisType = isRatio ? 'linear' : 'logarithmic';

    let yAxisTitle;
    if (isRatio) {
      yAxisTitle = 'Ratio';
    } else if (this.compressionType === 'size') {
      yAxisTitle = 'MiB';
    } else {
      yAxisTitle = 'MiB/s';
    }

    return {
      scales: {
        y: {
          type: yAxisType,
          title: {
            display: true,
            text: yAxisTitle
          },
          ticks: isRatio ? {
            callback: function(value) {
              return value.toFixed(2) + 'x';
            }
          } : {
            callback: function(value) {
              // Format logarithmic ticks nicely for non-ratio charts
              if (value === 1 || value === 10 || value === 100 ||
                  value === 1000 || value === 10000) {
                return value.toLocaleString();
              }
              return null;
            }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 10
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                if (isRatio) {
                  label += context.parsed.y.toFixed(3) + 'x';
                } else if (this.compressionType === 'size') {
                  label += context.parsed.y.toFixed(2) + ' MiB';
                } else {
                  label += context.parsed.y.toFixed(2) + ' MiB/s';
                }
              }
              return label;
            }.bind(this)
          }
        }
      }
    };
  }
}

/**
 * Random access benchmark class.
 * Shows performance of random row access operations.
 */
export class RandomAccessBenchmark extends BaseBenchmark {
  constructor(config) {
    super(config);
  }

  /**
   * Get chart options optimized for random access benchmarks.
   */
  getChartOptions() {
    return {
      scales: {
        y: {
          type: 'logarithmic',
          title: {
            display: true,
            text: 'ms/iter'
          },
          ticks: {
            callback: function(value) {
              // Format logarithmic ticks nicely
              if (value === 0.1 || value === 1 || value === 10 ||
                  value === 100 || value === 1000) {
                return value.toLocaleString();
              }
              return null;
            }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 10
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                label += context.parsed.y.toFixed(3) + ' ms/iter';
              }
              return label;
            }
          }
        }
      }
    };
  }
}