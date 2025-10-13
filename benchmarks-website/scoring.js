"use strict";

// Scoring module for benchmarks
const scoring = {
  isQueryBenchmark(categoryName) {
    return (
      categoryName === "Clickbench" ||
      categoryName.startsWith("TPC-H") ||
      categoryName.startsWith("TPC-DS") ||
      categoryName === "Statistical and Population Genetics"
    );
  },

  isRandomAccessBenchmark(categoryName) {
    return categoryName === "Random Access";
  },

  isCompressionBenchmark(categoryName) {
    return categoryName === "Compression";
  },

  isCompressionSizeBenchmark(categoryName) {
    return categoryName === "Compression Size";
  },

  // Helper: Get latest data point for each series across all queries
  getLatestDataPerSeries(benchSet) {
    const seriesLatestData = new Map();

    // First pass: collect all unique series names across all queries
    const allSeriesNames = new Set();
    for (const [queryName, queryData] of benchSet.entries()) {
      if (!queryData.series || queryData.series.size === 0) continue;
      for (const seriesName of queryData.series.keys()) {
        allSeriesNames.add(seriesName);
      }
    }

    // Initialize the map with all series names (empty Maps for each)
    for (const seriesName of allSeriesNames) {
      seriesLatestData.set(seriesName, new Map());
    }

    // Second pass: populate with latest non-null values where available
    for (const [queryName, queryData] of benchSet.entries()) {
      if (!queryData.series || queryData.series.size === 0) continue;

      for (const [seriesName, seriesData] of queryData.series.entries()) {
        // Find the most recent non-null value for this series
        for (let i = seriesData.length - 1; i >= 0; i--) {
          const result = seriesData[i];
          if (result && result.value !== null && result.value !== undefined) {
            seriesLatestData.get(seriesName).set(queryName, result.value);
            break;
          }
        }
      }
    }

    return seriesLatestData;
  },

  // Helper: Calculate geometric mean score for query benchmarks
  calculateGeometricMeanScores(seriesLatestData, benchSet) {
    const seriesScores = new Map();

    for (const [seriesName, queryResults] of seriesLatestData.entries()) {
      const ratios = [];
      let totalRuntime = 0;
      let maxRuntime = 0;

      // Calculate max runtime for penalty
      for (const runtime of queryResults.values()) {
        maxRuntime = Math.max(maxRuntime, runtime);
        totalRuntime += runtime;
      }

      // Apply penalty rules: if max runtime < 300s, use 300s, then multiply by 2
      const penalty = Math.max(300000, maxRuntime) * 2;

      // For each query, calculate ratio against baseline
      for (const [queryName, queryData] of benchSet.entries()) {
        // Find baseline (best result) across all series for this query
        let baseline = Infinity;

        for (const [sName, latestData] of seriesLatestData.entries()) {
          if (latestData.has(queryName)) {
            baseline = Math.min(baseline, latestData.get(queryName));
          }
        }

        if (baseline === Infinity) continue;

        // Get this series' result or use penalty
        const seriesRuntime = queryResults.has(queryName)
          ? queryResults.get(queryName)
          : penalty;

        // Calculate ratio with 10ms constant shift
        const ratio = (10 + seriesRuntime) / (10 + baseline);
        ratios.push(ratio);
      }

      if (ratios.length > 0) {
        const product = ratios.reduce((acc, ratio) => acc * ratio, 1);
        const geometricMean = Math.pow(product, 1 / ratios.length);
        seriesScores.set(seriesName, {
          score: geometricMean,
          queryCount: ratios.length,
          totalRuntime: totalRuntime,
          actualQueryCount: queryResults.size,
        });
      }
    }

    return seriesScores;
  },

  calculateClickBenchScore(benchSet) {
    if (!benchSet || benchSet.size === 0) return null;

    // Step 1: Get latest data for each series
    const seriesLatestData = this.getLatestDataPerSeries(benchSet);

    if (seriesLatestData.size === 0) return null;

    // Step 2: Calculate scores using the modular scoring function
    return this.calculateGeometricMeanScores(seriesLatestData, benchSet);
  },

  formatScoresSummary(scores) {
    if (!scores || scores.size === 0) return null;

    // Sort by score (lower is better)
    const sortedScores = Array.from(scores.entries()).sort(
      (a, b) => a[1].score - b[1].score,
    );

    const summaryDiv = document.createElement("div");
    summaryDiv.className = "benchmark-scores-summary";

    const title = document.createElement("h3");
    title.className = "scores-title";
    title.textContent = "Performance Summary";
    summaryDiv.appendChild(title);

    const scoresList = document.createElement("div");
    scoresList.className = "scores-list";

    sortedScores.forEach(([seriesName, data], index) => {
      const scoreItem = document.createElement("div");
      scoreItem.className = "score-item";

      const rank = index + 1;
      const scoreText = data.score.toFixed(2);

      // Format runtime - assuming it's in milliseconds
      const formatRuntime = (ms) => {
        if (ms < 1000) return `${ms.toFixed(0)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
      };

      const totalRuntimeText = formatRuntime(data.totalRuntime);

      scoreItem.innerHTML = `
        <span class="score-rank">#${rank}</span>
        <span class="score-series">${seriesName}</span>
        <span class="score-metrics">
          <span class="score-value">${scoreText}x</span>
          <span class="score-runtime">${totalRuntimeText}</span>
        </span>
      `;

      scoresList.appendChild(scoreItem);
    });

    summaryDiv.appendChild(scoresList);

    const explanation = document.createElement("div");
    explanation.className = "scores-explanation";
    explanation.textContent =
      "Score: geometric mean of query time ratio to fastest with 10ms constant shift | Total: sum of all query times (lower is better)";
    summaryDiv.appendChild(explanation);

    return summaryDiv;
  },

  calculateRandomAccessMetrics(benchSet) {
    if (!benchSet || benchSet.size === 0) return null;

    // For Random Access, we want the latest data point for each series
    const latestResults = new Map();

    // Get the first (and likely only) query in the benchmark set
    for (const [queryName, queryData] of benchSet.entries()) {
      if (!queryData.series || queryData.series.size === 0) continue;

      // Find the most recent commit with data
      let latestCommitWithData = -1;
      for (let i = queryData.commits.length - 1; i >= 0; i--) {
        let hasData = false;
        for (const [seriesName, seriesData] of queryData.series.entries()) {
          const result = seriesData[i];
          if (result && result.value !== null && result.value !== undefined) {
            hasData = true;
            break;
          }
        }
        if (hasData) {
          latestCommitWithData = i;
          break;
        }
      }

      if (latestCommitWithData === -1) continue;

      // Get results for all series at the latest commit with data
      for (const [seriesName, seriesData] of queryData.series.entries()) {
        if (latestCommitWithData < seriesData.length) {
          const result = seriesData[latestCommitWithData];
          if (result && result.value !== null && result.value !== undefined) {
            latestResults.set(seriesName, result.value);
          }
        }
      }

      break; // Only process the first query for Random Access
    }

    if (latestResults.size === 0) return null;

    // Find the fastest time
    let fastestTime = Infinity;
    for (const time of latestResults.values()) {
      fastestTime = Math.min(fastestTime, time);
    }

    // Calculate metrics for each series
    const seriesMetrics = new Map();
    for (const [seriesName, time] of latestResults.entries()) {
      seriesMetrics.set(seriesName, {
        time: time,
        ratio: time / fastestTime,
      });
    }

    return seriesMetrics;
  },

  formatRandomAccessSummary(metrics) {
    if (!metrics || metrics.size === 0) return null;

    // Sort by time (lower is better)
    const sortedMetrics = Array.from(metrics.entries()).sort(
      (a, b) => a[1].time - b[1].time,
    );

    const summaryDiv = document.createElement("div");
    summaryDiv.className = "benchmark-scores-summary";

    const title = document.createElement("h3");
    title.className = "scores-title";
    title.textContent = "Random Access Performance";
    summaryDiv.appendChild(title);

    const metricsList = document.createElement("div");
    metricsList.className = "scores-list";

    // Format time helper
    const formatTime = (ms) => {
      if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
      if (ms < 1000) return `${ms.toFixed(1)}ms`;
      return `${(ms / 1000).toFixed(2)}s`;
    };

    sortedMetrics.forEach(([seriesName, data], index) => {
      const metricItem = document.createElement("div");
      metricItem.className = "score-item";

      const rank = index + 1;
      const timeText = formatTime(data.time);
      const ratioText = data.ratio.toFixed(2);

      metricItem.innerHTML = `
        <span class="score-rank">#${rank}</span>
        <span class="score-series">${seriesName}</span>
        <span class="score-metrics">
          <span class="score-runtime">${timeText}</span>
          <span class="score-value">${ratioText}x</span>
        </span>
      `;

      metricsList.appendChild(metricItem);
    });

    summaryDiv.appendChild(metricsList);

    const explanation = document.createElement("div");
    explanation.className = "scores-explanation";
    explanation.textContent =
      "Random access time | Ratio to fastest (lower is better)";
    summaryDiv.appendChild(explanation);

    return summaryDiv;
  },

  calculateCompressionMetrics(benchSet) {
    if (!benchSet || benchSet.size === 0) return null;

    // For Compression, we want the geometric mean of the ratio charts
    const compressRatios = [];
    const decompressRatios = [];

    // Find the specific ratio charts
    const compressRatioChart = benchSet.get(
      "VORTEX:PARQUET-ZSTD RATIO COMPRESS TIME",
    );
    const decompressRatioChart = benchSet.get(
      "VORTEX:PARQUET-ZSTD RATIO DECOMPRESS TIME",
    );

    if (!compressRatioChart && !decompressRatioChart) return null;

    // Find the most recent commit with data
    let latestCommitWithData = -1;

    // Check compress ratio chart
    if (compressRatioChart && compressRatioChart.series) {
      for (let i = compressRatioChart.commits.length - 1; i >= 0; i--) {
        let hasData = false;
        for (const [
          seriesName,
          seriesData,
        ] of compressRatioChart.series.entries()) {
          const result = seriesData[i];
          if (result && result.value !== null && result.value !== undefined) {
            hasData = true;
            break;
          }
        }
        if (hasData) {
          latestCommitWithData = i;
          break;
        }
      }
    }

    // Check decompress ratio chart if we haven't found data yet
    if (
      latestCommitWithData === -1 &&
      decompressRatioChart &&
      decompressRatioChart.series
    ) {
      for (let i = decompressRatioChart.commits.length - 1; i >= 0; i--) {
        let hasData = false;
        for (const [
          seriesName,
          seriesData,
        ] of decompressRatioChart.series.entries()) {
          const result = seriesData[i];
          if (result && result.value !== null && result.value !== undefined) {
            hasData = true;
            break;
          }
        }
        if (hasData) {
          latestCommitWithData = i;
          break;
        }
      }
    }

    if (latestCommitWithData === -1) return null;

    // Collect compress ratios (excluding wide table cols)
    if (compressRatioChart && compressRatioChart.series) {
      for (const [
        seriesName,
        seriesData,
      ] of compressRatioChart.series.entries()) {
        // Skip wide table cols datasets
        if (seriesName.toLowerCase().startsWith("wide table cols")) continue;

        if (latestCommitWithData < seriesData.length) {
          const result = seriesData[latestCommitWithData];
          if (
            result &&
            result.value !== null &&
            result.value !== undefined &&
            result.value > 0
          ) {
            // Invert the ratio (1/value) so higher is better
            compressRatios.push(1 / result.value);
          }
        }
      }
    }

    // Collect decompress ratios (excluding wide table cols)
    if (decompressRatioChart && decompressRatioChart.series) {
      for (const [
        seriesName,
        seriesData,
      ] of decompressRatioChart.series.entries()) {
        // Skip wide table cols datasets
        if (seriesName.toLowerCase().startsWith("wide table cols")) continue;

        if (latestCommitWithData < seriesData.length) {
          const result = seriesData[latestCommitWithData];
          if (
            result &&
            result.value !== null &&
            result.value !== undefined &&
            result.value > 0
          ) {
            // Invert the ratio (1/value) so higher is better
            decompressRatios.push(1 / result.value);
          }
        }
      }
    }

    // Calculate geometric means
    const calculateGeometricMean = (values) => {
      if (values.length === 0) return null;
      const product = values.reduce((acc, val) => acc * val, 1);
      return Math.pow(product, 1 / values.length);
    };

    const metrics = {
      compressRatio: calculateGeometricMean(compressRatios),
      decompressRatio: calculateGeometricMean(decompressRatios),
      compressCount: compressRatios.length,
      decompressCount: decompressRatios.length,
    };

    return metrics;
  },

  formatCompressionSummary(metrics) {
    if (!metrics) return null;

    const summaryDiv = document.createElement("div");
    summaryDiv.className = "benchmark-scores-summary";

    const title = document.createElement("h3");
    title.className = "scores-title";
    title.textContent = "Compression Throughput vs Parquet";
    summaryDiv.appendChild(title);

    const metricsList = document.createElement("div");
    metricsList.className = "scores-list";

    // Compress ratio
    if (metrics.compressRatio !== null) {
      const compressItem = document.createElement("div");
      compressItem.className = "score-item";
      compressItem.innerHTML = `
        <span class="score-rank">⚡</span>
        <span class="score-series">Write Speed (Compression)</span>
        <span class="score-metrics">
          <span class="score-value">${metrics.compressRatio.toFixed(2)}x</span>
        </span>
      `;
      metricsList.appendChild(compressItem);
    }

    // Decompress ratio
    if (metrics.decompressRatio !== null) {
      const decompressItem = document.createElement("div");
      decompressItem.className = "score-item";
      decompressItem.innerHTML = `
        <span class="score-rank">📤</span>
        <span class="score-series">Scan Speed (Decompression)</span>
        <span class="score-metrics">
          <span class="score-value">${metrics.decompressRatio.toFixed(
            2,
          )}x</span>
        </span>
      `;
      metricsList.appendChild(decompressItem);
    }

    summaryDiv.appendChild(metricsList);

    const explanation = document.createElement("div");
    explanation.className = "scores-explanation";
    explanation.textContent =
      "Inverse geometric mean of Vortex/Parquet ratios across 9 datasets (higher is better)";
    summaryDiv.appendChild(explanation);

    return summaryDiv;
  },

  calculateCompressionSizeMetrics(benchSet) {
    if (!benchSet || benchSet.size === 0) return null;

    // For Compression Size, we want the geometric mean of the size ratio chart
    const sizeRatios = [];

    // Find the size ratio chart
    const sizeRatioChart = benchSet.get("VORTEX:PARQUET-ZSTD SIZE");

    if (!sizeRatioChart) return null;

    // Find the most recent commit with data
    let latestCommitWithData = -1;

    if (sizeRatioChart.series) {
      for (let i = sizeRatioChart.commits.length - 1; i >= 0; i--) {
        let hasData = false;
        for (const [
          seriesName,
          seriesData,
        ] of sizeRatioChart.series.entries()) {
          const result = seriesData[i];
          if (result && result.value !== null && result.value !== undefined) {
            hasData = true;
            break;
          }
        }
        if (hasData) {
          latestCommitWithData = i;
          break;
        }
      }
    }

    if (latestCommitWithData === -1) return null;

    // Collect size ratios (excluding wide table cols)
    for (const [seriesName, seriesData] of sizeRatioChart.series.entries()) {
      // Skip wide table cols datasets
      if (seriesName.toLowerCase().startsWith("wide table cols")) continue;

      if (latestCommitWithData < seriesData.length) {
        const result = seriesData[latestCommitWithData];
        if (
          result &&
          result.value !== null &&
          result.value !== undefined &&
          result.value > 0
        ) {
          // Keep the ratio as-is (lower is better for size)
          sizeRatios.push(result.value);
        }
      }
    }

    // Calculate geometric mean of size ratios
    const calculateGeometricMean = (values) => {
      if (values.length === 0) return null;
      const product = values.reduce((acc, val) => acc * val, 1);
      return Math.pow(product, 1 / values.length);
    };

    // Calculate min and max
    const minRatio = sizeRatios.length > 0 ? Math.min(...sizeRatios) : null;
    const maxRatio = sizeRatios.length > 0 ? Math.max(...sizeRatios) : null;

    const metrics = {
      sizeRatio: calculateGeometricMean(sizeRatios),
      minRatio: minRatio,
      maxRatio: maxRatio,
      sizeRatioCount: sizeRatios.length,
    };

    return metrics;
  },

  formatCompressionSizeSummary(metrics) {
    if (!metrics || metrics.sizeRatio === null) return null;

    const summaryDiv = document.createElement("div");
    summaryDiv.className = "benchmark-scores-summary";

    const title = document.createElement("h3");
    title.className = "scores-title";
    title.textContent = "Compression Size Summary";
    summaryDiv.appendChild(title);

    const metricsList = document.createElement("div");
    metricsList.className = "scores-list";

    // Min ratio
    const minItem = document.createElement("div");
    minItem.className = "score-item";
    minItem.innerHTML = `
      <span class="score-rank">⬇️</span>
      <span class="score-series">Min Size Ratio</span>
      <span class="score-metrics">
        <span class="score-value">${metrics.minRatio.toFixed(2)}x</span>
      </span>
    `;
    metricsList.appendChild(minItem);

    // Mean ratio
    const meanItem = document.createElement("div");
    meanItem.className = "score-item";
    meanItem.innerHTML = `
      <span class="score-rank">📊</span>
      <span class="score-series">Mean Size Ratio</span>
      <span class="score-metrics">
        <span class="score-value">${metrics.sizeRatio.toFixed(2)}x</span>
      </span>
    `;
    metricsList.appendChild(meanItem);

    // Max ratio
    const maxItem = document.createElement("div");
    maxItem.className = "score-item";
    maxItem.innerHTML = `
      <span class="score-rank">⬆️</span>
      <span class="score-series">Max Size Ratio</span>
      <span class="score-metrics">
        <span class="score-value">${metrics.maxRatio.toFixed(2)}x</span>
      </span>
    `;
    metricsList.appendChild(maxItem);

    summaryDiv.appendChild(metricsList);

    const explanation = document.createElement("div");
    explanation.className = "scores-explanation";
    explanation.textContent = `Geometric mean of Vortex/Parquet size ratios across ${metrics.sizeRatioCount} datasets (lower is better)`;
    summaryDiv.appendChild(explanation);

    return summaryDiv;
  },
};

// Export the scoring module
export { scoring };
