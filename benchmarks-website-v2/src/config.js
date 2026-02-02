// Benchmark group configurations
export const BENCHMARK_CONFIGS = [
  {
    name: 'Random Access',
    renamedDatasets: {
      'vortex-tokio-local-disk': 'vortex-nvme',
      'lance-tokio-local-disk': 'lance-nvme',
      'parquet-tokio-local-disk': 'parquet-nvme',
    },
  },
  {
    name: 'Compression',
    keptCharts: [
      'COMPRESS TIME',
      'DECOMPRESS TIME',
      'PARQUET RS ZSTD COMPRESS TIME',
      'PARQUET RS ZSTD DECOMPRESS TIME',
      'LANCE COMPRESS TIME',
      'LANCE DECOMPRESS TIME',
      'VORTEX:PARQUET ZSTD RATIO COMPRESS TIME',
      'VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME',
      'VORTEX:LANCE RATIO COMPRESS TIME',
      'VORTEX:LANCE RATIO DECOMPRESS TIME',
    ],
    hiddenDatasets: new Set([
      'wide table cols=1000 chunks=1 rows=1000',
      'wide table cols=1000 chunks=50 rows=1000',
    ]),
    removedDatasets: new Set([
      'TPC-H l_comment canonical',
      'TPC-H l_comment chunked without fsst',
      'wide table cols=10 chunks=1 rows=1000',
      'wide table cols=100 chunks=1 rows=1000',
      'wide table cols=10 chunks=50 rows=1000',
      'wide table cols=100 chunks=50 rows=1000',
    ]),
    renamedDatasets: {
      lance: 'lance',
      Lance: 'lance',
      LANCE: 'lance',
    },
  },
  {
    name: 'Compression Size',
    keptCharts: [
      'VORTEX SIZE',
      'PARQUET SIZE',
      'LANCE SIZE',
      'VORTEX:PARQUET ZSTD SIZE',
      'VORTEX:LANCE SIZE',
    ],
    hiddenDatasets: new Set(['wide table cols=1000']),
    removedDatasets: new Set([
      'wide table cols=10 chunks=1 rows=1000',
      'wide table cols=100 chunks=1 rows=1000',
      'wide table cols=10 chunks=50 rows=1000',
      'wide table cols=100 chunks=50 rows=1000',
    ]),
    renamedDatasets: {
      lance: 'lance',
      Lance: 'lance',
      LANCE: 'lance',
    },
  },
  {
    name: 'Clickbench',
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
    hiddenDatasets: new Set(['datafusion:lance']),
  },
  {
    name: 'TPC-H (NVMe) (SF=1)',
    hiddenDatasets: new Set(['datafusion:lance']),
    keptCharts: [
      'TPC-H Q1', 'TPC-H Q2', 'TPC-H Q3', 'TPC-H Q4', 'TPC-H Q5', 'TPC-H Q6',
      'TPC-H Q7', 'TPC-H Q8', 'TPC-H Q9', 'TPC-H Q10', 'TPC-H Q11', 'TPC-H Q12',
      'TPC-H Q13', 'TPC-H Q14', 'TPC-H Q15', 'TPC-H Q16', 'TPC-H Q17', 'TPC-H Q18',
      'TPC-H Q19', 'TPC-H Q20', 'TPC-H Q21', 'TPC-H Q22',
    ],
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (S3) (SF=1)',
    hiddenDatasets: new Set(['datafusion:lance']),
    keptCharts: [
      'TPC-H Q1', 'TPC-H Q2', 'TPC-H Q3', 'TPC-H Q4', 'TPC-H Q5', 'TPC-H Q6',
      'TPC-H Q7', 'TPC-H Q8', 'TPC-H Q9', 'TPC-H Q10', 'TPC-H Q11', 'TPC-H Q12',
      'TPC-H Q13', 'TPC-H Q14', 'TPC-H Q15', 'TPC-H Q16', 'TPC-H Q17', 'TPC-H Q18',
      'TPC-H Q19', 'TPC-H Q20', 'TPC-H Q21', 'TPC-H Q22',
    ],
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (NVMe) (SF=10)',
    hiddenDatasets: new Set(['datafusion:lance']),
    keptCharts: [
      'TPC-H Q1', 'TPC-H Q2', 'TPC-H Q3', 'TPC-H Q4', 'TPC-H Q5', 'TPC-H Q6',
      'TPC-H Q7', 'TPC-H Q8', 'TPC-H Q9', 'TPC-H Q10', 'TPC-H Q11', 'TPC-H Q12',
      'TPC-H Q13', 'TPC-H Q14', 'TPC-H Q15', 'TPC-H Q16', 'TPC-H Q17', 'TPC-H Q18',
      'TPC-H Q19', 'TPC-H Q20', 'TPC-H Q21', 'TPC-H Q22',
    ],
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (S3) (SF=10)',
    hiddenDatasets: new Set(['datafusion:lance']),
    keptCharts: [
      'TPC-H Q1', 'TPC-H Q2', 'TPC-H Q3', 'TPC-H Q4', 'TPC-H Q5', 'TPC-H Q6',
      'TPC-H Q7', 'TPC-H Q8', 'TPC-H Q9', 'TPC-H Q10', 'TPC-H Q11', 'TPC-H Q12',
      'TPC-H Q13', 'TPC-H Q14', 'TPC-H Q15', 'TPC-H Q16', 'TPC-H Q17', 'TPC-H Q18',
      'TPC-H Q19', 'TPC-H Q20', 'TPC-H Q21', 'TPC-H Q22',
    ],
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (NVMe) (SF=100)',
    hiddenDatasets: new Set(['datafusion:lance']),
    keptCharts: [
      'TPC-H Q1', 'TPC-H Q2', 'TPC-H Q3', 'TPC-H Q4', 'TPC-H Q5', 'TPC-H Q6',
      'TPC-H Q7', 'TPC-H Q8', 'TPC-H Q9', 'TPC-H Q10', 'TPC-H Q11', 'TPC-H Q12',
      'TPC-H Q13', 'TPC-H Q14', 'TPC-H Q15', 'TPC-H Q16', 'TPC-H Q17', 'TPC-H Q18',
      'TPC-H Q19', 'TPC-H Q20', 'TPC-H Q21', 'TPC-H Q22',
    ],
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (S3) (SF=100)',
    hiddenDatasets: new Set(['datafusion:lance']),
    keptCharts: [
      'TPC-H Q1', 'TPC-H Q2', 'TPC-H Q3', 'TPC-H Q4', 'TPC-H Q5', 'TPC-H Q6',
      'TPC-H Q7', 'TPC-H Q8', 'TPC-H Q9', 'TPC-H Q10', 'TPC-H Q11', 'TPC-H Q12',
      'TPC-H Q13', 'TPC-H Q14', 'TPC-H Q15', 'TPC-H Q16', 'TPC-H Q17', 'TPC-H Q18',
      'TPC-H Q19', 'TPC-H Q20', 'TPC-H Q21', 'TPC-H Q22',
    ],
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (NVMe) (SF=1000)',
    hidden: true,
    hiddenDatasets: new Set(['datafusion:lance']),
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-H (S3) (SF=1000)',
    hidden: true,
    hiddenDatasets: new Set(['datafusion:lance']),
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DataFusion:lance': 'datafusion:lance',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-DS (NVMe) (SF=1)',
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'TPC-DS (NVMe) (SF=10)',
    hidden: true,
    renamedDatasets: {
      'DataFusion:vortex-file-compressed': 'datafusion:vortex',
      'DataFusion:parquet': 'datafusion:parquet',
      'DataFusion:arrow': 'datafusion:in-memory-arrow',
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
  {
    name: 'Statistical and Population Genetics',
    renamedDatasets: {
      'DuckDB:vortex-file-compressed': 'duckdb:vortex',
      'DuckDB:parquet': 'duckdb:parquet',
      'DuckDB:duckdb': 'duckdb:duckdb',
    },
  },
];

// Chart name remapping
export const CHART_NAME_MAP = {
  'COMPRESS TIME': 'VORTEX WRITE TIME (COMPRESSION)',
  'DECOMPRESS TIME': 'VORTEX SCAN TIME (DECOMPRESSION)',
  'PARQUET RS ZSTD COMPRESS TIME': 'PARQUET WRITE TIME (COMPRESSION)',
  'PARQUET RS ZSTD DECOMPRESS TIME': 'PARQUET SCAN TIME (DECOMPRESSION)',
  'LANCE COMPRESS TIME': 'LANCE WRITE TIME (COMPRESSION)',
  'LANCE DECOMPRESS TIME': 'LANCE SCAN TIME (DECOMPRESSION)',
  'VORTEX SIZE': 'VORTEX SIZE',
  'PARQUET ZSTD SIZE': 'PARQUET SIZE',
  'LANCE SIZE': 'LANCE SIZE',
  'VORTEX:RAW SIZE': 'VORTEX vs RAW SIZE RATIO',
  'VORTEX:PARQUET ZSTD SIZE': 'VORTEX vs PARQUET SIZE RATIO',
  'VORTEX:LANCE SIZE': 'VORTEX vs LANCE SIZE RATIO',
  'VORTEX:PARQUET ZSTD RATIO COMPRESS TIME': 'VORTEX vs PARQUET WRITE TIME RATIO',
  'VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME': 'VORTEX vs PARQUET SCAN TIME RATIO',
  'VORTEX:LANCE RATIO COMPRESS TIME': 'VORTEX vs LANCE WRITE TIME RATIO',
  'VORTEX:LANCE RATIO DECOMPRESS TIME': 'VORTEX vs LANCE SCAN TIME RATIO',
};

// Category tags for filtering
export const CATEGORY_TAGS = {
  'Random Access': ['Read/Write'],
  Compression: ['Read/Write'],
  'Compression Size': ['Read/Write'],
  Clickbench: ['Queries (NVMe)'],
  'TPC-H (NVMe) (SF=1)': ['Queries (NVMe)', 'TPC-H (SF=1)'],
  'TPC-H (S3) (SF=1)': ['Queries (S3)', 'TPC-H (SF=1)'],
  'TPC-H (NVMe) (SF=10)': ['Queries (NVMe)', 'TPC-H (SF=10)'],
  'TPC-H (S3) (SF=10)': ['Queries (S3)', 'TPC-H (SF=10)'],
  'TPC-H (NVMe) (SF=100)': ['Queries (NVMe)', 'TPC-H (SF=100)'],
  'TPC-H (S3) (SF=100)': ['Queries (S3)', 'TPC-H (SF=100)'],
  'TPC-H (NVMe) (SF=1000)': ['Queries (NVMe)', 'TPC-H (SF=1000)'],
  'TPC-H (S3) (SF=1000)': ['Queries (S3)', 'TPC-H (SF=1000)'],
  'TPC-DS (NVMe) (SF=1)': ['Queries (NVMe)', 'TPC-DS (SF=1)'],
  'TPC-DS (NVMe) (SF=10)': ['Queries (NVMe)', 'TPC-DS (SF=10)'],
  'Statistical and Population Genetics': ['Queries (NVMe)', 'StatPopGen'],
};

// Benchmark descriptions
export const BENCHMARK_DESCRIPTIONS = {
  'Random Access':
    'Tests performance of selecting arbitrary row indices from a file on NVMe storage',
  Compression:
    'Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet files (with zstd page compression)',
  'Compression Size':
    'Compares compressed file sizes and compression ratios across different encoding strategies',
  Clickbench:
    "ClickHouse's analytical benchmark suite testing real-world query patterns on web analytics data",
  'Statistical and Population Genetics':
    'A suite of Statistical and Population genetics queries using the gnomAD dataset',
};

// Scale factor descriptions
export const SCALE_FACTOR_DESCRIPTIONS = {
  1: 'SF=1 (~1GB of data)',
  10: 'SF=10 (~10GB of data)',
  100: 'SF=100 (~100GB of data)',
  1000: 'SF=1000 (~1TB of data)',
};

// Engine filter labels
export const ENGINE_LABELS = {
  all: 'All',
  duckdb: 'DuckDB',
  datafusion: 'DataFusion',
  vortex: 'Vortex',
  parquet: 'Parquet',
};

// Series color map
export const SERIES_COLOR_MAP = {
  'vortex-nvme': '#19a508',
  'parquet-nvme': '#ef7f1d',
  'lance-nvme': '#3B82F6',
  'datafusion:arrow': '#7a27b1',
  'datafusion:in-memory-arrow': '#7a27b1',
  'datafusion:parquet': '#ef7f1d',
  'datafusion:vortex': '#19a508',
  'datafusion:vortex-compact': '#15850a',
  'datafusion:lance': '#2D936C',
  'duckdb:parquet': '#985113',
  'duckdb:vortex': '#0e5e04',
  'duckdb:vortex-compact': '#0b4a03',
  'duckdb:duckdb': '#87752e',
  'vortex:lance': '#FF8787',
};

// Fallback color palette
export const FALLBACK_PALETTE = [
  '#5971FD', // Vortex Blue
  '#CEE562', // Vortex Green
  '#EEB3E1', // Vortex Pink
  '#FF8C42', // Orange
  '#B8336A', // Deep pink
  '#726DA8', // Purple
  '#2D936C', // Teal
  '#E9B44C', // Gold
];

// Default visible commits
export const DEFAULT_COMMIT_RANGE = 100;
