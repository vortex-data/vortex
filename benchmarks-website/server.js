import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import zlib from 'zlib';
import readline from 'readline';
import { LTTB } from 'downsample';

// Import shared data processing functions
import { BENCHMARK_GROUPS, QUERY_NAME_MAP } from './config.js';
import { shared } from './data-shared.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const PORT = process.env.PORT || 3000;
const DATA_URL = process.env.DATA_URL || 'https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz';
const COMMITS_URL = process.env.COMMITS_URL || 'https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json';
const REFRESH_INTERVAL = process.env.REFRESH_INTERVAL || 5 * 60 * 1000; // 5 minutes
const MAX_POINTS_PER_CHART = 200; // Target max points per chart
const USE_LOCAL_DATA = process.env.USE_LOCAL_DATA === 'true'; // Use local sample data for development

// Common renamed datasets for query benchmarks (TPC-H, TPC-DS, Clickbench)
const QUERY_RENAME_MAP = {
  'DataFusion:vortex-file-compressed': 'datafusion:vortex',
  'datafusion:vortex-file-compressed': 'datafusion:vortex',
  'DataFusion:parquet': 'datafusion:parquet',
  'datafusion:parquet': 'datafusion:parquet',
  'DataFusion:arrow': 'datafusion:in-memory-arrow',
  'datafusion:arrow': 'datafusion:in-memory-arrow',
  'DataFusion:lance': 'datafusion:lance',
  'datafusion:lance': 'datafusion:lance',
  'DataFusion:vortex-compact': 'datafusion:vortex-compact',
  'datafusion:vortex-compact': 'datafusion:vortex-compact',
  'DuckDB:vortex-file-compressed': 'duckdb:vortex',
  'duckdb:vortex-file-compressed': 'duckdb:vortex',
  'DuckDB:parquet': 'duckdb:parquet',
  'duckdb:parquet': 'duckdb:parquet',
  'DuckDB:duckdb': 'duckdb:duckdb',
  'duckdb:duckdb': 'duckdb:duckdb',
  'DuckDB:vortex-compact': 'duckdb:vortex-compact',
  'duckdb:vortex-compact': 'duckdb:vortex-compact',
};

// Series renaming configuration (matching index.html settings)
const SERIES_RENAME_MAP = {
  'Random Access': {
    'vortex-tokio-local-disk': 'vortex-nvme',
    'lance-tokio-local-disk': 'lance-nvme',
    'parquet-tokio-local-disk': 'parquet-nvme',
  },
  'Compression': {
    'lance': 'lance',
    'Lance': 'lance',
    'LANCE': 'lance',
  },
  'Compression Size': {
    'lance': 'lance',
    'Lance': 'lance',
    'LANCE': 'lance',
  },
  'Clickbench': QUERY_RENAME_MAP,
  'TPC-H (NVMe) (SF=1)': QUERY_RENAME_MAP,
  'TPC-H (S3) (SF=1)': QUERY_RENAME_MAP,
  'TPC-H (NVMe) (SF=10)': QUERY_RENAME_MAP,
  'TPC-H (S3) (SF=10)': QUERY_RENAME_MAP,
  'TPC-H (NVMe) (SF=100)': QUERY_RENAME_MAP,
  'TPC-H (S3) (SF=100)': QUERY_RENAME_MAP,
  'TPC-H (NVMe) (SF=1000)': QUERY_RENAME_MAP,
  'TPC-H (S3) (SF=1000)': QUERY_RENAME_MAP,
  'TPC-DS (NVMe) (SF=1)': QUERY_RENAME_MAP,
  'TPC-DS (NVMe) (SF=10)': QUERY_RENAME_MAP,
  'Statistical and Population Genetics': {
    'DuckDB:vortex-file-compressed': 'duckdb:vortex',
    'duckdb:vortex-file-compressed': 'duckdb:vortex',
    'DuckDB:parquet': 'duckdb:parquet',
    'duckdb:parquet': 'duckdb:parquet',
    'DuckDB:duckdb': 'duckdb:duckdb',
    'duckdb:duckdb': 'duckdb:duckdb',
    'DuckDB:vortex-compact': 'duckdb:vortex-compact',
    'duckdb:vortex-compact': 'duckdb:vortex-compact',
  },
};

// Apply series renaming based on group
function applySeriesRename(seriesName, groupId) {
  const renameMap = SERIES_RENAME_MAP[groupId];
  if (!renameMap) return seriesName;

  // Try exact match first
  if (renameMap[seriesName]) return renameMap[seriesName];

  // Try case-insensitive match
  const lowerName = seriesName.toLowerCase();
  for (const [key, value] of Object.entries(renameMap)) {
    if (key.toLowerCase() === lowerName) return value;
  }

  return seriesName;
}

// In-memory data store
let dataStore = {
  commits: [],          // Sorted array of commits
  commitsMap: {},       // commit_id -> commit object
  groups: {},           // groupId -> Map<chartName, chartData>
  metadata: null,       // Precomputed metadata
  downsampledData: {},  // groupId -> chartName -> { '1x': data, '2x': data, '4x': data, '8x': data }
  lastUpdated: null
};

// MIME types for static files
const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.webmanifest': 'application/manifest+json'
};

// === LTTB Downsampling (using downsample library) ===

// Multi-series LTTB: pick indices based on averaged series, then apply to all
function multiSeriesLttbIndices(seriesMap, targetPoints) {
  const seriesNames = Array.from(seriesMap.keys());
  if (seriesNames.length === 0) return [];

  const firstSeries = seriesMap.get(seriesNames[0]);
  if (!firstSeries) return [];

  const dataLength = firstSeries.length;
  if (dataLength <= targetPoints) {
    return Array.from({ length: dataLength }, (_, i) => i);
  }

  // Calculate average values across all series at each index
  // Format data as [x, y] pairs for LTTB library
  const avgData = [];
  for (let i = 0; i < dataLength; i++) {
    let sum = 0, count = 0;
    for (const series of seriesMap.values()) {
      const point = series[i];
      // Handle both { value: x } objects and raw values
      if (point !== null && point !== undefined) {
        const val = typeof point === 'object' && point.value !== undefined ? point.value : point;
        if (val !== null && val !== undefined && !isNaN(val)) {
          sum += val;
          count++;
        }
      }
    }
    // LTTB library expects [x, y] format
    avgData.push([i, count > 0 ? sum / count : 0]);
  }

  // Run LTTB using the library
  const sampled = LTTB(avgData, targetPoints);

  // Extract indices from sampled points (x values are original indices)
  const indices = sampled.map(point => Math.round(point[0]));

  // Ensure first and last indices are always included
  const firstIndex = 0;
  const lastIndex = dataLength - 1;

  if (!indices.includes(firstIndex)) {
    indices.unshift(firstIndex);
  }
  if (!indices.includes(lastIndex)) {
    indices.push(lastIndex);
  }

  // Sort to maintain order (in case we added first/last out of order)
  indices.sort((a, b) => a - b);

  return indices;
}

function downsampleChartData(chartData, factor) {
  const targetPoints = Math.ceil(chartData.commits.length / factor);
  if (targetPoints >= chartData.commits.length) {
    return chartData; // No downsampling needed
  }

  const indices = multiSeriesLttbIndices(chartData.series, targetPoints);

  const downsampledCommits = indices.map(i => chartData.commits[i]);
  const downsampledSeries = new Map();

  for (const [seriesName, seriesData] of chartData.series.entries()) {
    downsampledSeries.set(seriesName, indices.map(i => seriesData[i]));
  }

  return {
    ...chartData,
    commits: downsampledCommits,
    series: downsampledSeries,
    originalLength: chartData.commits.length
  };
}

// === Data Fetching and Processing ===

async function fetchData(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return response;
}

async function fetchAndParseGzippedJsonl(url) {
  const response = await fetchData(url);
  const buffer = await response.arrayBuffer();
  const decompressed = zlib.gunzipSync(Buffer.from(buffer));
  const text = decompressed.toString('utf-8');

  return text.split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => JSON.parse(line));
}

async function fetchCommits(url) {
  const response = await fetchData(url);
  const text = await response.text();
  return text.split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => JSON.parse(line));
}

async function loadLocalData() {
  const dataPath = path.join(__dirname, 'sample', 'data.json');
  const commitsPath = path.join(__dirname, 'sample', 'commits.json');

  console.log('Loading local data from sample directory...');

  // Use streaming to read large JSONL files
  const readJsonlStream = (filePath) => {
    return new Promise((resolve, reject) => {
      const results = [];
      const rl = readline.createInterface({
        input: fs.createReadStream(filePath, { encoding: 'utf-8' }),
        crlfDelay: Infinity
      });

      rl.on('line', (line) => {
        if (line.trim().length > 0) {
          try {
            results.push(JSON.parse(line));
          } catch (e) {
            // Skip invalid JSON lines
          }
        }
      });

      rl.on('close', () => resolve(results));
      rl.on('error', reject);
    });
  };

  const [benchmarkData, commitsArray] = await Promise.all([
    readJsonlStream(dataPath),
    readJsonlStream(commitsPath)
  ]);

  return [benchmarkData, commitsArray];
}

async function refreshData() {
  console.log('Refreshing data...');
  const startTime = Date.now();

  try {
    let benchmarkData, commitsArray;

    if (USE_LOCAL_DATA) {
      [benchmarkData, commitsArray] = await loadLocalData();
    } else {
      [benchmarkData, commitsArray] = await Promise.all([
        fetchAndParseGzippedJsonl(DATA_URL),
        fetchCommits(COMMITS_URL)
      ]);
    }

    console.log(`Fetched ${benchmarkData.length} benchmark entries and ${commitsArray.length} commits`);

    // Build commits map
    const commitsMap = {};
    commitsArray.forEach(commit => {
      commitsMap[commit.id] = commit;
    });

    // Parse and sort commits using shared function
    const commits = shared.parseCommits(commitsMap);

    // Initialize groups using shared function
    const groups = shared.initializeGroups();
    const missingCommits = new Set();
    const uncategorizableNames = new Set();

    // Process benchmarks - filter out those with missing commits instead of creating placeholders
    for (const benchmark of benchmarkData) {
      // Check if commit exists, skip if missing
      if (!benchmark.commit) {
        benchmark.commit = commitsMap[benchmark.commit_id];
        if (!benchmark.commit) {
          missingCommits.add(benchmark.commit_id);
          continue; // Skip benchmarks with missing commits
        }
      }

      // Use shared determineGroupId function
      const groupId = shared.determineGroupId(benchmark);
      if (!groupId) {
        uncategorizableNames.add(benchmark.name);
        continue;
      }

      const group = groups[groupId];
      if (!group) continue;

      // Process benchmark data using shared functions
      let [query, seriesName] = benchmark.name.split("/");
      const normalized = shared.normalizeSeriesName(query, seriesName, groupId);
      query = normalized.name;
      seriesName = normalized.seriesName;

      // Apply series renaming
      seriesName = applySeriesRename(seriesName, groupId);

      const prettyQ = shared.formatQueryName(query);
      if (prettyQ.includes("PARQUET-UNC")) continue;

      // Set units
      let unit = benchmark.unit;
      if (!unit && benchmark.name.startsWith("vortex size/")) {
        unit = "bytes";
      } else if (!unit && (benchmark.name.startsWith("vortex:raw size/") ||
                          benchmark.name.startsWith("vortex:parquet-zstd size/") ||
                          benchmark.name.startsWith("vortex:lance size/"))) {
        unit = "ratio";
      }

      const sortPosition = (query.slice(0, 4) === "tpch" || query.slice(0, 5) === "tpcds")
        ? parseInt(prettyQ.split(" ")[1]?.substring(1) || "0", 10)
        : 0;

      // Add to group
      let arr = group.get(prettyQ);
      if (!arr) {
        group.set(prettyQ, {
          sort_position: sortPosition,
          commits: commits,
          unit: shared.getUnit(unit),
          series: new Map()
        });
        arr = group.get(prettyQ);
      }

      let series = arr.series.get(seriesName);
      if (!series) {
        arr.series.set(seriesName, new Array(commits.length).fill(null));
        series = arr.series.get(seriesName);
      }

      series[benchmark.commit.sortedIndex] = {
        value: shared.convertValue(benchmark.value, unit)
      };
    }

    // Find the first commit index that has any benchmark data
    let firstDataIndex = commits.length;
    for (const groupCharts of Object.values(groups)) {
      for (const chartData of groupCharts.values()) {
        for (const seriesData of chartData.series.values()) {
          for (let i = 0; i < seriesData.length; i++) {
            if (seriesData[i] !== null) {
              firstDataIndex = Math.min(firstDataIndex, i);
              break;
            }
          }
        }
      }
    }

    // Filter out commits with no data
    if (firstDataIndex > 0 && firstDataIndex < commits.length) {
      console.log(`Filtering out ${firstDataIndex} commits with no benchmark data`);

      // Create filtered commits array
      const filteredCommits = commits.slice(firstDataIndex);

      // Update sortedIndex for remaining commits
      filteredCommits.forEach((commit, newIndex) => {
        commit.sortedIndex = newIndex;
      });

      // Update all chart data to use filtered commits and trimmed series
      for (const groupCharts of Object.values(groups)) {
        for (const chartData of groupCharts.values()) {
          chartData.commits = filteredCommits;

          // Trim series data
          for (const [seriesName, seriesData] of chartData.series.entries()) {
            chartData.series.set(seriesName, seriesData.slice(firstDataIndex));
          }
        }
      }

      // Replace commits reference for metadata
      commits.splice(0, commits.length, ...filteredCommits);
    }

    // Sort groups using shared function
    shared.sortGroups(groups);

    // Precompute downsampled versions
    const downsampledData = {};
    for (const [groupName, groupCharts] of Object.entries(groups)) {
      downsampledData[groupName] = {};
      for (const [chartName, chartData] of groupCharts.entries()) {
        downsampledData[groupName][chartName] = {
          '1x': chartData,
          '2x': downsampleChartData(chartData, 2),
          '4x': downsampleChartData(chartData, 4),
          '8x': downsampleChartData(chartData, 8)
        };
      }
    }

    // Build metadata
    const metadata = buildMetadata(groups, commits);

    // Update data store
    dataStore = {
      commits,
      commitsMap,
      groups,
      metadata,
      downsampledData,
      lastUpdated: new Date().toISOString()
    };

    console.log(`Data refresh complete in ${Date.now() - startTime}ms`);
    console.log(`Groups: ${Object.keys(groups).length}, Missing commits: ${missingCommits.size}`);

    if (uncategorizableNames.size > 0) {
      console.log(`Uncategorizable names: ${uncategorizableNames.size}`);
    }

  } catch (error) {
    console.error('Error refreshing data:', error);
  }
}

// === Summary Calculation (mirrors scoring.js logic) ===

function calculateGroupSummary(groupName, groupCharts) {
  if (groupName === "Random Access") {
    return calculateRandomAccessSummary(groupCharts);
  } else if (groupName === "Compression") {
    return calculateCompressionSummary(groupCharts);
  } else if (groupName === "Compression Size") {
    return calculateCompressionSizeSummary(groupCharts);
  } else if (groupName === "Clickbench" || groupName.startsWith("TPC-H") ||
             groupName.startsWith("TPC-DS") || groupName === "Statistical and Population Genetics") {
    return calculateQueryBenchmarkSummary(groupCharts);
  }
  return null;
}

function calculateRandomAccessSummary(groupCharts) {
  const latestResults = new Map();

  for (const [queryName, queryData] of groupCharts.entries()) {
    if (!queryData.series || queryData.series.size === 0) continue;

    // Find the most recent commit with data
    let latestCommitWithData = -1;
    for (let i = queryData.commits.length - 1; i >= 0; i--) {
      let hasData = false;
      for (const seriesData of queryData.series.values()) {
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

    for (const [seriesName, seriesData] of queryData.series.entries()) {
      if (latestCommitWithData < seriesData.length) {
        const result = seriesData[latestCommitWithData];
        if (result && result.value !== null && result.value !== undefined) {
          latestResults.set(seriesName, result.value);
        }
      }
    }
    break;
  }

  if (latestResults.size === 0) return null;

  let fastestTime = Infinity;
  for (const time of latestResults.values()) {
    fastestTime = Math.min(fastestTime, time);
  }

  const rankings = [];
  for (const [seriesName, time] of latestResults.entries()) {
    rankings.push({
      name: seriesName,
      time: time,
      ratio: time / fastestTime
    });
  }

  rankings.sort((a, b) => a.time - b.time);

  return {
    type: 'randomAccess',
    title: 'Random Access Performance',
    rankings: rankings,
    explanation: 'Random access time | Ratio to fastest (lower is better)'
  };
}

function calculateCompressionSummary(groupCharts) {
  const compressRatioChart = groupCharts.get("VORTEX:PARQUET-ZSTD RATIO COMPRESS TIME");
  const decompressRatioChart = groupCharts.get("VORTEX:PARQUET-ZSTD RATIO DECOMPRESS TIME");

  if (!compressRatioChart && !decompressRatioChart) return null;

  const compressRatios = [];
  const decompressRatios = [];

  // Find latest commit with data
  let latestCommit = -1;
  const chartToCheck = compressRatioChart || decompressRatioChart;
  if (chartToCheck?.series) {
    for (let i = chartToCheck.commits.length - 1; i >= 0; i--) {
      for (const seriesData of chartToCheck.series.values()) {
        if (seriesData[i]?.value !== null && seriesData[i]?.value !== undefined) {
          latestCommit = i;
          break;
        }
      }
      if (latestCommit !== -1) break;
    }
  }

  if (latestCommit === -1) return null;

  // Collect ratios (excluding wide table cols)
  if (compressRatioChart?.series) {
    for (const [seriesName, seriesData] of compressRatioChart.series.entries()) {
      if (seriesName.toLowerCase().startsWith("wide table cols")) continue;
      const result = seriesData[latestCommit];
      if (result?.value > 0) {
        compressRatios.push(1 / result.value);
      }
    }
  }

  if (decompressRatioChart?.series) {
    for (const [seriesName, seriesData] of decompressRatioChart.series.entries()) {
      if (seriesName.toLowerCase().startsWith("wide table cols")) continue;
      const result = seriesData[latestCommit];
      if (result?.value > 0) {
        decompressRatios.push(1 / result.value);
      }
    }
  }

  const geometricMean = (values) => {
    if (values.length === 0) return null;
    const product = values.reduce((acc, val) => acc * val, 1);
    return Math.pow(product, 1 / values.length);
  };

  return {
    type: 'compression',
    title: 'Compression Throughput vs Parquet',
    compressRatio: geometricMean(compressRatios),
    decompressRatio: geometricMean(decompressRatios),
    datasetCount: compressRatios.length,
    explanation: `Inverse geometric mean of Vortex/Parquet ratios across ${compressRatios.length} datasets (higher is better)`
  };
}

function calculateCompressionSizeSummary(groupCharts) {
  const sizeRatioChart = groupCharts.get("VORTEX:PARQUET-ZSTD SIZE");
  if (!sizeRatioChart?.series) return null;

  let latestCommit = -1;
  for (let i = sizeRatioChart.commits.length - 1; i >= 0; i--) {
    for (const seriesData of sizeRatioChart.series.values()) {
      if (seriesData[i]?.value !== null && seriesData[i]?.value !== undefined) {
        latestCommit = i;
        break;
      }
    }
    if (latestCommit !== -1) break;
  }

  if (latestCommit === -1) return null;

  const sizeRatios = [];
  for (const [seriesName, seriesData] of sizeRatioChart.series.entries()) {
    if (seriesName.toLowerCase().startsWith("wide table cols")) continue;
    const result = seriesData[latestCommit];
    if (result?.value > 0) {
      sizeRatios.push(result.value);
    }
  }

  if (sizeRatios.length === 0) return null;

  const geometricMean = (values) => {
    const product = values.reduce((acc, val) => acc * val, 1);
    return Math.pow(product, 1 / values.length);
  };

  return {
    type: 'compressionSize',
    title: 'Compression Size Summary',
    minRatio: Math.min(...sizeRatios),
    meanRatio: geometricMean(sizeRatios),
    maxRatio: Math.max(...sizeRatios),
    datasetCount: sizeRatios.length,
    explanation: `Geometric mean of Vortex/Parquet size ratios across ${sizeRatios.length} datasets (lower is better)`
  };
}

function calculateQueryBenchmarkSummary(groupCharts) {
  // Get latest data per series across all queries
  const seriesLatestData = new Map();
  const allSeriesNames = new Set();

  for (const queryData of groupCharts.values()) {
    if (!queryData.series) continue;
    for (const seriesName of queryData.series.keys()) {
      allSeriesNames.add(seriesName);
    }
  }

  for (const seriesName of allSeriesNames) {
    seriesLatestData.set(seriesName, new Map());
  }

  for (const [queryName, queryData] of groupCharts.entries()) {
    if (!queryData.series) continue;

    for (const [seriesName, seriesData] of queryData.series.entries()) {
      for (let i = seriesData.length - 1; i >= 0; i--) {
        const result = seriesData[i];
        if (result?.value !== null && result?.value !== undefined) {
          seriesLatestData.get(seriesName).set(queryName, result.value);
          break;
        }
      }
    }
  }

  if (seriesLatestData.size === 0) return null;

  // Calculate geometric mean scores
  const seriesScores = new Map();

  for (const [seriesName, queryResults] of seriesLatestData.entries()) {
    const ratios = [];
    let totalRuntime = 0;
    let maxRuntime = 0;

    for (const runtime of queryResults.values()) {
      maxRuntime = Math.max(maxRuntime, runtime);
      totalRuntime += runtime;
    }

    const penalty = Math.max(300000, maxRuntime) * 2;

    for (const [queryName] of groupCharts.entries()) {
      let baseline = Infinity;
      for (const latestData of seriesLatestData.values()) {
        if (latestData.has(queryName)) {
          baseline = Math.min(baseline, latestData.get(queryName));
        }
      }

      if (baseline === Infinity) continue;

      const seriesRuntime = queryResults.has(queryName)
        ? queryResults.get(queryName)
        : penalty;

      const ratio = (10 + seriesRuntime) / (10 + baseline);
      ratios.push(ratio);
    }

    if (ratios.length > 0) {
      const product = ratios.reduce((acc, ratio) => acc * ratio, 1);
      const geometricMean = Math.pow(product, 1 / ratios.length);
      seriesScores.set(seriesName, {
        score: geometricMean,
        totalRuntime: totalRuntime,
        queryCount: queryResults.size
      });
    }
  }

  if (seriesScores.size === 0) return null;

  const rankings = Array.from(seriesScores.entries())
    .map(([name, data]) => ({ name, score: data.score, totalRuntime: data.totalRuntime }))
    .sort((a, b) => a.score - b.score);

  return {
    type: 'queryBenchmark',
    title: 'Performance Summary',
    rankings: rankings,
    explanation: 'Score: geometric mean of query time ratio to fastest with 10ms constant shift | Total: sum of all query times (lower is better)'
  };
}

function buildMetadata(groups, commits) {
  const groupMetadata = {};

  for (const [groupName, groupCharts] of Object.entries(groups)) {
    const charts = [];
    for (const [chartName, chartData] of groupCharts.entries()) {
      const seriesNames = Array.from(chartData.series.keys());

      // Calculate latest values for summary
      const latestValues = {};
      for (const [seriesName, seriesData] of chartData.series.entries()) {
        for (let i = seriesData.length - 1; i >= 0; i--) {
          if (seriesData[i] !== null && seriesData[i].value !== null) {
            latestValues[seriesName] = seriesData[i].value;
            break;
          }
        }
      }

      charts.push({
        name: chartName,
        unit: chartData.unit,
        seriesNames,
        sortPosition: chartData.sort_position,
        totalPoints: chartData.commits.length,
        latestValues
      });
    }

    // Calculate summary for this group
    const summary = calculateGroupSummary(groupName, groupCharts);

    groupMetadata[groupName] = {
      charts,
      totalCharts: charts.length,
      hasData: charts.length > 0,
      summary: summary
    };
  }

  return {
    groups: groupMetadata,
    totalCommits: commits.length,
    commits: commits.map(c => ({
      id: c.id,
      message: c.message.split('\n')[0],
      timestamp: c.timestamp,
      author: c.author.name
    })),
    lastUpdated: new Date().toISOString()
  };
}

// === API Handlers ===

function handleMetadataRequest(res) {
  if (!dataStore.metadata) {
    sendJson(res, 503, { error: 'Data not yet loaded' });
    return;
  }
  sendJson(res, 200, dataStore.metadata);
}

function handleDataRequest(res, groupName, chartName, startCommit, endCommit) {
  if (!dataStore.downsampledData) {
    sendJson(res, 503, { error: 'Data not yet loaded' });
    return;
  }

  const groupData = dataStore.downsampledData[groupName];
  if (!groupData) {
    sendJson(res, 404, { error: `Group '${groupName}' not found` });
    return;
  }

  const chartVersions = groupData[chartName];
  if (!chartVersions) {
    sendJson(res, 404, { error: `Chart '${chartName}' not found in group '${groupName}'` });
    return;
  }

  // Get full data
  const fullData = chartVersions['1x'];

  // Determine range
  let startIndex = 0;
  let endIndex = fullData.commits.length - 1;

  if (startCommit) {
    const idx = fullData.commits.findIndex(c => c.id === startCommit || c.id.startsWith(startCommit));
    if (idx !== -1) startIndex = idx;
  }

  if (endCommit) {
    const idx = fullData.commits.findIndex(c => c.id === endCommit || c.id.startsWith(endCommit));
    if (idx !== -1) endIndex = idx;
  }

  const rangeLength = endIndex - startIndex + 1;

  // Select appropriate downsampling level based on range
  let selectedVersion;
  if (rangeLength <= MAX_POINTS_PER_CHART) {
    selectedVersion = '1x';
  } else if (rangeLength <= MAX_POINTS_PER_CHART * 2) {
    selectedVersion = '2x';
  } else if (rangeLength <= MAX_POINTS_PER_CHART * 4) {
    selectedVersion = '4x';
  } else {
    selectedVersion = '8x';
  }

  const chartData = chartVersions[selectedVersion];

  // For downsampled data, we need to find the appropriate range in the downsampled arrays
  // Since indices are preserved relatively, we can map the range
  let resultCommits, resultSeries;

  // Helper to extract value from data point
  const extractValue = (d) => {
    if (d === null || d === undefined) return null;
    if (typeof d === 'object' && d.value !== undefined) return d.value;
    if (typeof d === 'number') return d;
    return null;
  };

  if (selectedVersion === '1x') {
    resultCommits = fullData.commits.slice(startIndex, endIndex + 1);
    resultSeries = {};
    for (const [seriesName, seriesData] of fullData.series.entries()) {
      // Extract just the value from each data point
      resultSeries[seriesName] = seriesData.slice(startIndex, endIndex + 1)
        .map(extractValue);
    }
  } else {
    // For downsampled data, filter to range based on originalLength mapping
    const scale = parseInt(selectedVersion);
    const dsStartIndex = Math.floor(startIndex / scale);
    const dsEndIndex = Math.min(Math.ceil(endIndex / scale), chartData.commits.length - 1);

    resultCommits = chartData.commits.slice(dsStartIndex, dsEndIndex + 1);
    resultSeries = {};
    for (const [seriesName, seriesData] of chartData.series.entries()) {
      // Extract just the value from each data point
      resultSeries[seriesName] = seriesData.slice(dsStartIndex, dsEndIndex + 1)
        .map(extractValue);
    }
  }

  sendJson(res, 200, {
    group: groupName,
    chart: chartName,
    unit: chartData.unit,
    downsampleLevel: selectedVersion,
    originalLength: fullData.commits.length,
    requestedRange: { startIndex, endIndex, length: rangeLength },
    commits: resultCommits.map(c => ({
      id: c.id,
      message: c.message?.split('\n')[0] || '',
      timestamp: c.timestamp,
      author: c.author?.name || 'Unknown',
      url: c.url
    })),
    series: resultSeries
  });
}

// === HTTP Server ===

function sendJson(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*'
  });
  res.end(JSON.stringify(data));
}

function sendFile(res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404);
        res.end('Not Found');
      } else {
        res.writeHead(500);
        res.end('Internal Server Error');
      }
      return;
    }

    // Disable caching for JS files during development
    const headers = { 'Content-Type': mimeType };
    if (ext === '.js') {
      headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0';
      headers['Pragma'] = 'no-cache';
      headers['Expires'] = '0';
    }
    res.writeHead(200, headers);
    res.end(data);
  });
}

function parseUrl(url) {
  const [pathname, queryString] = url.split('?');
  const params = new URLSearchParams(queryString || '');
  return { pathname, params };
}

const server = http.createServer((req, res) => {
  const { pathname, params } = parseUrl(req.url);

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    });
    res.end();
    return;
  }

  // API routes
  if (pathname === '/api/metadata') {
    handleMetadataRequest(res);
    return;
  }

  if (pathname.startsWith('/api/data/')) {
    const parts = pathname.replace('/api/data/', '').split('/');
    const groupName = decodeURIComponent(parts[0] || '');
    const chartName = decodeURIComponent(parts.slice(1).join('/') || '');
    const startCommit = params.get('start_commit');
    const endCommit = params.get('end_commit');

    handleDataRequest(res, groupName, chartName, startCommit, endCommit);
    return;
  }

  // Static files
  let filePath = pathname === '/' ? '/index.html' : pathname;
  filePath = path.join(__dirname, filePath);

  // Security: prevent directory traversal
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  // Don't serve sample directory
  if (filePath.includes('/sample/')) {
    res.writeHead(404);
    res.end('Not Found');
    return;
  }

  sendFile(res, filePath);
});

// === Startup ===

async function start() {
  console.log('Starting Vortex Benchmarks Server...');

  // Initial data load
  await refreshData();

  // Schedule periodic refresh
  setInterval(refreshData, REFRESH_INTERVAL);

  // Start HTTP server
  server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

start().catch(console.error);
