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
