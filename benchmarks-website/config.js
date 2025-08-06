"use strict";

// Configuration constants
export const CONFIG = {
  MOBILE_BREAKPOINT: 1199,
  MOBILE_MAX_DATA_POINTS: 100,
  DEFAULT_VISIBLE_COMMITS: 50,
  DEBOUNCE_DELAY: 150, // Increased from 50ms to reduce update frequency
  MOBILE_DEBOUNCE_DELAY: 300, // Increased from 200ms for better mobile performance
  ZOOM_THROTTLE_DELAY: 16, // ~60fps throttling for zoom operations
  THROTTLE_SCROLL: 100,
  SEARCH_DEBOUNCE: 300,
  CHART_OBSERVER_MARGIN: "50px",
  SCROLL_OFFSET_PADDING: 20,
  ZOOM_SPEED: 0.1,
  MIN_VISIBLE_COMMITS: 10,
  COMPRESS_THROUGHPUT_MAX: 1024,
  DECOMPRESS_THROUGHPUT_MAX: 8192,
  ANIMATION_DURATION: 1000,
  LINK_FEEDBACK_DURATION: 2000,
  BACK_TO_TOP_THRESHOLD: 200,
  SCROLL_ACTIVE_THRESHOLD: 100,
  URL_INIT_DELAY: 100,
  RESIZE_DEBOUNCE: 250,
  // Performance monitoring (set to true to enable console timing)
  ENABLE_ZOOM_PERFORMANCE_TIMING: false,
};

// Color mappings for series
export const SERIES_COLOR_MAP = {
  "datafusion:arrow": "#7a27b1",
  "datafusion:in-memory-arrow": "#7a27b1",
  "datafusion:parquet": "#ef7f1d",
  "datafusion:vortex": "#19a508",
  "duckdb:parquet": "#985113",
  "duckdb:vortex": "#0e5e04",
  "duckdb:duckdb": "#87752e",
};

// Brand colors
export const VORTEX_COLORS = {
  primary: "#5971FD", // Vortex Blue
  accent: "#CEE562", // Vortex Green
  pink: "#EEB3E1", // Vortex Pink
  black: "#101010", // Vortex Black
  gray: "#666666", // Secondary gray
};

// Fallback color palette
export const FALLBACK_PALETTE = [
  VORTEX_COLORS.primary,
  VORTEX_COLORS.accent,
  VORTEX_COLORS.pink,
  "#FF8C42", // Orange
  "#B8336A", // Deep pink
  "#726DA8", // Purple
  "#2D936C", // Teal
  "#E9B44C", // Gold
];

// Benchmark descriptions
export const BENCHMARK_DESCRIPTIONS = {
  "Random Access":
    "Tests performance of selecting arbitrary row indices from a file on NVMe storage",
  Compression:
    "Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet files (with zstd page compression)",
  "Compression Size":
    "Compares compressed file sizes and compression ratios across different encoding strategies, helping evaluate the space efficiency trade-offs between Vortex and Parquet formats",
  "TPC-H (NVMe)":
    "TPC-H benchmark queries executed on local NVMe storage, testing analytical query performance",
  "TPC-H (S3)":
    "TPC-H benchmark queries executed against data stored in Amazon S3, measuring cloud storage query performance and the impact of network latency on analytical workloads",
  Clickbench:
    "ClickHouse's analytical benchmark suite testing real-world query patterns on web analytics data, run against NVMe storage",
};

// Category tags mapping
export const CATEGORY_TAGS = {
  "Random Access": ["Read/Write"],
  Compression: ["Read/Write"],
  "Compression Size": ["Read/Write"],
  Clickbench: ["Queries (NVMe)"],
  "TPC-H (NVMe) (SF=1)": ["Queries (NVMe)", "TPC-H (SF=1)"],
  "TPC-H (S3) (SF=1)": ["Queries (S3)", "TPC-H (SF=1)"],
  "TPC-H (NVMe) (SF=10)": ["Queries (NVMe)", "TPC-H (SF=10)"],
  "TPC-H (S3) (SF=10)": ["Queries (S3)", "TPC-H (SF=10)"],
  "TPC-H (NVMe) (SF=100)": ["Queries (NVMe)", "TPC-H (SF=100)"],
  "TPC-H (S3) (SF=100)": ["Queries (S3)", "TPC-H (SF=100)"],
  "TPC-H (NVMe) (SF=1000)": ["Queries (NVMe)", "TPC-H (SF=1000)"],
  "TPC-H (S3) (SF=1000)": ["Queries (S3)", "TPC-H (SF=1000)"],
};

// Scale factor descriptions
export const SCALE_FACTOR_DESCRIPTIONS = {
  1: "SF=1 (~1GB of data)",
  10: "SF=10 (~10GB of data)",
  100: "SF=100 (~100GB of data)",
  1000: "SF=1000 (~1TB of data)",
};

// Query name transformations
export const QUERY_NAME_MAP = {
  "VORTEX:RAW SIZE": "VORTEX COMPRESSION RATIO",
  "VORTEX:PARQUET-ZSTD SIZE": "VORTEX:PARQUET-ZSTD SIZE RATIO",
};

// Engine labels
export const ENGINE_LABELS = {
  all: "All",
  duckdb: "DuckDB",
  datafusion: "DataFusion",
  vortex: "Vortex",
  parquet: "Parquet",
};

// Group definitions
export const BENCHMARK_GROUPS = [
  "Random Access",
  "Compression",
  "Compression Size",
  "Clickbench",
  "TPC-H (NVMe) (SF=1)",
  "TPC-H (S3) (SF=1)",
  "TPC-H (NVMe) (SF=10)",
  "TPC-H (S3) (SF=10)",
  "TPC-H (NVMe) (SF=100)",
  "TPC-H (S3) (SF=100)",
  "TPC-H (NVMe) (SF=1000)",
  "TPC-H (S3) (SF=1000)",
];