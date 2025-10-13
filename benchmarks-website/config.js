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
  "vortex-nvme": "#19a508",
  "parquet-nvme": "#ef7f1d",
  "datafusion:arrow": "#7a27b1",
  "datafusion:in-memory-arrow": "#7a27b1",
  "datafusion:parquet": "#ef7f1d",
  "datafusion:vortex": "#19a508",
  "datafusion:lance": "#2D936C",
  "duckdb:parquet": "#985113",
  "duckdb:vortex": "#0e5e04",
  "duckdb:duckdb": "#87752e",
  "vortex:lance": "#FF8787",
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
  "TPC-DS (NVMe)":
    "TPC-DS benchmark queries executed on local NVMe storage, testing complex analytical query performance with a retail sales dataset",
  "Statistical and Population Genetics":`A suite of Statistical and Population genetics queries executed on local NVMe storage.

A custom benchmark for statistical and population genetics workloads using the gnomAD v3.1.2 release of the jointly called One Thousand Genomes (1kG) and Human Genome Diversity Project (HGDP) dataset (1kG+HGDP). Only a prefix of Chromosome 21 is used for benchmarking.

Data source: <https://gnomad.broadinstitute.org/>.`,
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
  "TPC-DS (NVMe) (SF=1)": ["Queries (NVMe)", "TPC-DS (SF=1)"],
  "TPC-DS (NVMe) (SF=10)": ["Queries (NVMe)", "TPC-DS (SF=10)"],
  "Statistical and Population Genetics": ["Queries (NVMe)", "StatPopGen"],
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
  "TPCH Q1": "TPC-H Q1",
  "TPCH Q2": "TPC-H Q2",
  "TPCH Q3": "TPC-H Q3",
  "TPCH Q4": "TPC-H Q4",
  "TPCH Q5": "TPC-H Q5",
  "TPCH Q6": "TPC-H Q6",
  "TPCH Q7": "TPC-H Q7",
  "TPCH Q8": "TPC-H Q8",
  "TPCH Q9": "TPC-H Q9",
  "TPCH Q10": "TPC-H Q10",
  "TPCH Q11": "TPC-H Q11",
  "TPCH Q12": "TPC-H Q12",
  "TPCH Q13": "TPC-H Q13",
  "TPCH Q14": "TPC-H Q14",
  "TPCH Q15": "TPC-H Q15",
  "TPCH Q16": "TPC-H Q16",
  "TPCH Q17": "TPC-H Q17",
  "TPCH Q18": "TPC-H Q18",
  "TPCH Q19": "TPC-H Q19",
  "TPCH Q20": "TPC-H Q20",
  "TPCH Q21": "TPC-H Q21",
  "TPCH Q22": "TPC-H Q22"
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
  "TPC-DS (NVMe) (SF=1)",
  "TPC-DS (NVMe) (SF=10)",
  "Statistical and Population Genetics"
];
