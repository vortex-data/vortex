"use strict";

import { QueryBenchmark, CompressionBenchmark, RandomAccessBenchmark } from './benchmark-types.js';
import { BENCHMARK_DESCRIPTIONS, SCALE_FACTOR_DESCRIPTIONS } from './config.js';

// Standard dataset renames for query benchmarks
const STANDARD_QUERY_RENAMES = {
  "DataFusion:vortex-file-compressed": "datafusion:vortex",
  "DataFusion:parquet": "datafusion:parquet",
  "DataFusion:arrow": "datafusion:in-memory-arrow",
  "DataFusion:lance": "datafusion:lance",
  "DuckDB:vortex-file-compressed": "duckdb:vortex",
  "DuckDB:parquet": "duckdb:parquet",
  "DuckDB:duckdb": "duckdb:duckdb",
};

/**
 * Helper to generate TPC-H description based on scale factor and storage type.
 */
function getTpcHDescription(scaleFactor, storage) {
  const scaleFactorInfo = SCALE_FACTOR_DESCRIPTIONS[scaleFactor] || "various scale factors";

  if (storage === "nvme") {
    return `TPC-H benchmark queries executed on local NVMe storage, testing analytical query performance at ${scaleFactorInfo}`;
  } else if (storage === "s3") {
    return `TPC-H benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance and the impact of network latency on analytical workloads at ${scaleFactorInfo}`;
  }
  return "";
}

/**
 * Helper to generate TPC-DS description based on scale factor and storage type.
 */
function getTpcDsDescription(scaleFactor, storage) {
  const scaleFactorInfo = SCALE_FACTOR_DESCRIPTIONS[scaleFactor] || "various scale factors";

  if (storage === "nvme") {
    return `TPC-DS benchmark queries executed on local NVMe storage, testing complex analytical query performance with a retail sales dataset at ${scaleFactorInfo}`;
  } else if (storage === "s3") {
    return `TPC-DS benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance for complex retail analytics workloads at ${scaleFactorInfo}`;
  }
  return "";
}

/**
 * Helper to generate TPC-H configs for all scale factors and storage types.
 */
function generateTPCHConfigs() {
  const configs = {};
  const scaleFactors = [1, 10, 100, 1000];
  const storageTypes = ['NVMe', 'S3'];

  for (const sf of scaleFactors) {
    for (const storage of storageTypes) {
      const key = `TPC-H (${storage}) (SF=${sf})`;
      configs[key] = {
        type: QueryBenchmark,
        config: {
          queryType: "tpch",
          scaleFactor: sf,
          storage: storage.toLowerCase(),
          description: getTpcHDescription(sf, storage.toLowerCase()),
          tags: [`Queries (${storage})`, `TPC-H (SF=${sf})`],
          renamedDatasets: STANDARD_QUERY_RENAMES,
          hiddenDatasets: new Set(["datafusion:lance"])
        }
      };
    }
  }

  return configs;
}

/**
 * Helper to generate TPC-DS configs for available scale factors.
 */
function generateTPCDSConfigs() {
  const configs = {};
  const scaleFactors = [1, 10]; // Only SF=1 and SF=10 are currently available
  const storage = 'NVMe'; // Only NVMe storage for TPC-DS currently

  for (const sf of scaleFactors) {
    const key = `TPC-DS (${storage}) (SF=${sf})`;
    configs[key] = {
      type: QueryBenchmark,
      config: {
        queryType: "tpcds",
        scaleFactor: sf,
        storage: storage.toLowerCase(),
        description: getTpcDsDescription(sf, storage.toLowerCase()),
        tags: [`Queries (${storage})`, `TPC-DS (SF=${sf})`],
        renamedDatasets: {
          "DataFusion:vortex-file-compressed": "datafusion:vortex",
          "DataFusion:parquet": "datafusion:parquet",
          "DataFusion:arrow": "datafusion:in-memory-arrow",
          "DuckDB:vortex-file-compressed": "duckdb:vortex",
          "DuckDB:parquet": "duckdb:parquet",
          "DuckDB:duckdb": "duckdb:duckdb",
        }
      }
    };
  }

  return configs;
}

/**
 * Master configuration for all benchmark types.
 * This replaces the inline configuration previously in index.html.
 */
export const BENCHMARK_CONFIGS = {
  "Random Access": {
    type: RandomAccessBenchmark,
    config: {
      description: BENCHMARK_DESCRIPTIONS["Random Access"],
      tags: ["Read/Write"],
      renamedDatasets: {
        "vortex-tokio-local-disk": "vortex-nvme",
        "lance-tokio-local-disk": "lance-nvme",
        "parquet-tokio-local-disk": "parquet-nvme",
      }
    }
  },

  "Compression": {
    type: CompressionBenchmark,
    config: {
      compressionType: "time",
      description: BENCHMARK_DESCRIPTIONS["Compression"],
      tags: ["Read/Write"],
      keptCharts: [
        "COMPRESS TIME",
        "DECOMPRESS TIME",
        "PARQUET RS-ZSTD COMPRESS TIME",
        "PARQUET RS-ZSTD DECOMPRESS TIME",
        "LANCE COMPRESS TIME",
        "LANCE DECOMPRESS TIME",
        "VORTEX:PARQUET-ZSTD RATIO COMPRESS TIME",
        "VORTEX:PARQUET-ZSTD RATIO DECOMPRESS TIME",
        "VORTEX:LANCE RATIO COMPRESS TIME",
        "VORTEX:LANCE RATIO DECOMPRESS TIME",
      ],
      hiddenDatasets: new Set([
        "wide table cols=1000 chunks=1 rows=1000",
        "wide table cols=1000 chunks=50 rows=1000",
      ]),
      removedDatasets: new Set([
        "TPC-H l_comment canonical",
        "TPC-H l_comment chunked without fsst",
        "wide table cols=10 chunks=1 rows=1000",
        "wide table cols=100 chunks=1 rows=1000",
        "wide table cols=10 chunks=50 rows=1000",
        "wide table cols=100 chunks=50 rows=1000",
      ]),
      renamedDatasets: undefined
    }
  },

  "Compression Size": {
    type: CompressionBenchmark,
    config: {
      compressionType: "size",
      description: BENCHMARK_DESCRIPTIONS["Compression Size"],
      tags: ["Read/Write"],
      keptCharts: [
        "VORTEX SIZE",
        "PARQUET-ZSTD SIZE",
        "LANCE SIZE",
        "VORTEX:PARQUET-ZSTD SIZE",
        "VORTEX:LANCE SIZE"
      ],
      hiddenDatasets: new Set(["wide table cols=1000"]),
      removedDatasets: new Set([
        "wide table cols=10 chunks=1 rows=1000",
        "wide table cols=100 chunks=1 rows=1000",
        "wide table cols=10 chunks=50 rows=1000",
        "wide table cols=100 chunks=50 rows=1000",
      ])
    }
  },

  "Clickbench": {
    type: QueryBenchmark,
    config: {
      queryType: "clickbench",
      description: BENCHMARK_DESCRIPTIONS["Clickbench"],
      tags: ["Queries (NVMe)"],
      renamedDatasets: STANDARD_QUERY_RENAMES,
      hiddenDatasets: new Set(["datafusion:lance"])
    }
  },

  // Generate all TPC-H configurations
  ...generateTPCHConfigs(),

  // Generate all TPC-DS configurations
  ...generateTPCDSConfigs(),

  "Statistical and Population Genetics": {
    type: QueryBenchmark,
    config: {
      queryType: "statpopgen",
      description: BENCHMARK_DESCRIPTIONS["Statistical and Population Genetics"],
      tags: ["Queries (NVMe)", "StatPopGen"],
      renamedDatasets: {
        "DuckDB:vortex-file-compressed": "duckdb:vortex",
        "DuckDB:parquet": "duckdb:parquet",
        "DuckDB:duckdb": "duckdb:duckdb",
      }
    }
  }
};

/**
 * Get configuration for a specific benchmark.
 */
export function getBenchmarkConfig(benchmarkName) {
  return BENCHMARK_CONFIGS[benchmarkName];
}

/**
 * Get all benchmark names.
 */
export function getAllBenchmarkNames() {
  return Object.keys(BENCHMARK_CONFIGS);
}

/**
 * Get benchmarks by tag.
 */
export function getBenchmarksByTag(tag) {
  return Object.entries(BENCHMARK_CONFIGS)
    .filter(([name, entry]) => entry.config.tags && entry.config.tags.includes(tag))
    .map(([name]) => name);
}

/**
 * Check if a benchmark exists.
 */
export function benchmarkExists(benchmarkName) {
  return benchmarkName in BENCHMARK_CONFIGS;
}