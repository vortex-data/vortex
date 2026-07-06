// =============================================================================
// SQL query benchmark suites — single source of truth.
// To add a new SQL query benchmark, add one entry to QUERY_SUITES.
// The server (routing/formatting) and frontend (UI config) both derive from this.
// =============================================================================

export const QUERY_SUITES = [
  {
    prefix: "clickbench",
    displayName: "Clickbench",
    queryPrefix: "CLICKBENCH",
    description:
      "ClickHouse's analytical benchmark suite testing real-world query patterns on web analytics data",
    tags: ["Queries (NVMe)"],
    hiddenDatasets: ["datafusion:lance"],
  },
  {
    prefix: "clickbench-sorted",
    displayName: "Clickbench Sorted",
    queryPrefix: "CLICKBENCH SORTED",
    description:
      "ClickBench queries over data globally sorted by event date and event time",
    tags: ["Queries (NVMe)"],
    hiddenDatasets: ["datafusion:lance"],
  },
  {
    prefix: "statpopgen",
    displayName: "Statistical and Population Genetics",
    queryPrefix: "STATPOPGEN",
    description:
      "A suite of Statistical and Population genetics queries using the gnomAD dataset",
    tags: ["Queries (NVMe)", "StatPopGen"],
  },
  {
    prefix: "polarsignals",
    displayName: "PolarSignals Profiling",
    queryPrefix: "POLARSIGNALS",
    description:
      "Profiling data benchmark modeled on PolarSignals/Parca, exercising scan-layer performance with projection and filter pushdown on deeply nested schemas",
    tags: ["Queries (NVMe)", "PolarSignals"],
  },
  {
    prefix: "tpch",
    displayName: "TPC-H",
    queryPrefix: "TPC-H",
    datasetKey: "tpch",
    fanOut: true,
    hiddenDatasets: ["datafusion:lance"],
  },
  {
    prefix: "tpcds",
    displayName: "TPC-DS",
    queryPrefix: "TPC-DS",
    datasetKey: "tpcds",
    fanOut: true,
  },
  { prefix: "fineweb", skip: true },
];

// Pre-registered fan-out groups (storage x scale factor).
export const FAN_OUT_GROUPS = [
  "TPC-H (NVMe) (SF=1)",
  "TPC-H (S3) (SF=1)",
  "TPC-H (NVMe) (SF=10)",
  "TPC-H (S3) (SF=10)",
  "TPC-H (NVMe) (SF=100)",
  "TPC-H (S3) (SF=100)",
  "TPC-H (NVMe) (SF=1000)",
  "TPC-H (S3) (SF=1000)",
  "TPC-DS (NVMe) (SF=1)",
  "TPC-DS (NVMe) (SF=10)",
];

// Canonical engine:format renaming used by all query suites.
export const ENGINE_RENAMES = {
  "datafusion:vortex-file-compressed": "datafusion:vortex",
  "datafusion:parquet": "datafusion:parquet",
  "datafusion:arrow": "datafusion:in-memory-arrow",
  "datafusion:lance": "datafusion:lance",
  "datafusion:vortex-compact": "datafusion:vortex-compact",
  "duckdb:vortex-file-compressed": "duckdb:vortex",
  "duckdb:parquet": "duckdb:parquet",
  "duckdb:duckdb": "duckdb:duckdb",
  "duckdb:vortex-compact": "duckdb:vortex-compact",
  "vortex-tokio-local-disk": "vortex-nvme",
  "vortex-compact-tokio-local-disk": "vortex-compact-nvme",
  "lance-tokio-local-disk": "lance-nvme",
  "parquet-tokio-local-disk": "parquet-nvme",
  lance: "lance",
};

// =============================================================================
// Below: frontend UI config, derived from QUERY_SUITES where possible.
// =============================================================================

// Build BENCHMARK_CONFIGS: bespoke non-query groups + generated query group entries.
const BESPOKE_CONFIGS = [
  {
    name: "Random Access",
    renamedDatasets: {
      "vortex-tokio-local-disk": "vortex-nvme",
      "vortex-compact-tokio-local-disk": "vortex-compact-nvme",
      "lance-tokio-local-disk": "lance-nvme",
      "parquet-tokio-local-disk": "parquet-nvme",
    },
  },
  {
    name: "Compression",
    keptCharts: [
      "COMPRESS TIME",
      "DECOMPRESS TIME",
      "PARQUET RS ZSTD COMPRESS TIME",
      "PARQUET RS ZSTD DECOMPRESS TIME",
      "LANCE COMPRESS TIME",
      "LANCE DECOMPRESS TIME",
      "VORTEX:PARQUET ZSTD RATIO COMPRESS TIME",
      "VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME",
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
    renamedDatasets: { lance: "lance", Lance: "lance", LANCE: "lance" },
  },
  {
    name: "Compression Size",
    keptCharts: [
      "VORTEX SIZE",
      "PARQUET SIZE",
      "LANCE SIZE",
      "VORTEX:PARQUET ZSTD SIZE",
      "VORTEX:LANCE SIZE",
    ],
    hiddenDatasets: new Set(["wide table cols=1000"]),
    removedDatasets: new Set([
      "wide table cols=10 chunks=1 rows=1000",
      "wide table cols=100 chunks=1 rows=1000",
      "wide table cols=10 chunks=50 rows=1000",
      "wide table cols=100 chunks=50 rows=1000",
    ]),
    renamedDatasets: { lance: "lance", Lance: "lance", LANCE: "lance" },
  },
];

function querySuiteConfig(name, suite) {
  const cfg = { name, renamedDatasets: { ...ENGINE_RENAMES } };
  if (suite?.hiddenDatasets?.length)
    cfg.hiddenDatasets = new Set(suite.hiddenDatasets);
  return cfg;
}

function buildQueryConfigs() {
  const configs = [];
  for (const s of QUERY_SUITES) {
    if (s.skip) continue;
    if (!s.fanOut) {
      configs.push(querySuiteConfig(s.displayName, s));
    }
  }
  for (const g of FAN_OUT_GROUPS) {
    const suite = QUERY_SUITES.find(
      (s) => s.fanOut && g.startsWith(s.displayName),
    );
    const cfg = querySuiteConfig(g, suite);
    if (g.includes("SF=1000") || (g.includes("TPC-DS") && g.includes("SF=10)")))
      cfg.hidden = true;
    configs.push(cfg);
  }
  return configs;
}

export const BENCHMARK_CONFIGS = [...BESPOKE_CONFIGS, ...buildQueryConfigs()];

// Chart name remapping (compression benchmarks only)
export const CHART_NAME_MAP = {
  "COMPRESS TIME": "VORTEX WRITE TIME (COMPRESSION)",
  "DECOMPRESS TIME": "VORTEX SCAN TIME (DECOMPRESSION)",
  "PARQUET RS ZSTD COMPRESS TIME": "PARQUET WRITE TIME (COMPRESSION)",
  "PARQUET RS ZSTD DECOMPRESS TIME": "PARQUET SCAN TIME (DECOMPRESSION)",
  "LANCE COMPRESS TIME": "LANCE WRITE TIME (COMPRESSION)",
  "LANCE DECOMPRESS TIME": "LANCE SCAN TIME (DECOMPRESSION)",
  "VORTEX SIZE": "VORTEX SIZE",
  "PARQUET ZSTD SIZE": "PARQUET SIZE",
  "LANCE SIZE": "LANCE SIZE",
  "VORTEX:RAW SIZE": "VORTEX vs RAW SIZE RATIO",
  "VORTEX:PARQUET ZSTD SIZE": "VORTEX vs PARQUET SIZE RATIO",
  "VORTEX:LANCE SIZE": "VORTEX vs LANCE SIZE RATIO",
  "VORTEX:PARQUET ZSTD RATIO COMPRESS TIME":
    "VORTEX vs PARQUET WRITE TIME RATIO",
  "VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME":
    "VORTEX vs PARQUET SCAN TIME RATIO",
  "VORTEX:LANCE RATIO COMPRESS TIME": "VORTEX vs LANCE WRITE TIME RATIO",
  "VORTEX:LANCE RATIO DECOMPRESS TIME": "VORTEX vs LANCE SCAN TIME RATIO",
};

// Category tags for sidebar filtering
export const CATEGORY_TAGS = {
  "Random Access": ["Read/Write"],
  Compression: ["Read/Write"],
  "Compression Size": ["Read/Write"],
};
for (const s of QUERY_SUITES) {
  if (!s.skip && !s.fanOut && s.tags) CATEGORY_TAGS[s.displayName] = s.tags;
}
for (const g of FAN_OUT_GROUPS) {
  const m = g.match(/^(.+?) \((NVMe|S3)\) \((SF=\d+)\)$/);
  CATEGORY_TAGS[g] = [
    m[2] === "S3" ? "Queries (S3)" : "Queries (NVMe)",
    `${m[1]} (${m[3]})`,
  ];
}

// Benchmark descriptions
export const BENCHMARK_DESCRIPTIONS = {
  "Random Access":
    "Tests performance of selecting arbitrary row indices from a file on NVMe storage",
  Compression:
    "Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet files (with zstd page compression)",
  "Compression Size":
    "Compares compressed file sizes and compression ratios across different encoding strategies",
};
for (const s of QUERY_SUITES) {
  if (s.description) BENCHMARK_DESCRIPTIONS[s.displayName] = s.description;
}

// Scale factor descriptions
export const SCALE_FACTOR_DESCRIPTIONS = {
  1: "SF=1 (~1GB of data)",
  10: "SF=10 (~10GB of data)",
  100: "SF=100 (~100GB of data)",
  1000: "SF=1000 (~1TB of data)",
};

// Engine filter labels
export const ENGINE_LABELS = {
  all: "All",
  duckdb: "DuckDB",
  datafusion: "DataFusion",
  vortex: "Vortex",
  parquet: "Parquet",
};

// Series color map
export const SERIES_COLOR_MAP = {
  "vortex-nvme": "#19a508",
  "vortex-compact-nvme": "#15850a",
  "parquet-nvme": "#ef7f1d",
  "lance-nvme": "#3B82F6",
  "datafusion:arrow": "#7a27b1",
  "datafusion:in-memory-arrow": "#7a27b1",
  "datafusion:parquet": "#ef7f1d",
  "datafusion:vortex": "#19a508",
  "datafusion:vortex-compact": "#15850a",
  "datafusion:lance": "#2D936C",
  "duckdb:parquet": "#985113",
  "duckdb:vortex": "#0e5e04",
  "duckdb:vortex-compact": "#0b4a03",
  "duckdb:duckdb": "#87752e",
  "vortex:lance": "#FF8787",
};

// Fallback color palette
export const FALLBACK_PALETTE = [
  "#5971FD",
  "#CEE562",
  "#EEB3E1",
  "#FF8C42",
  "#B8336A",
  "#726DA8",
  "#2D936C",
  "#E9B44C",
];

// Default visible commits
export const DEFAULT_COMMIT_RANGE = 100;
