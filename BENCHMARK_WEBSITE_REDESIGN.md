# Vortex Benchmarks Website: Architecture & Implementation Plan

This document describes the architecture for a high-performance benchmarks visualization website for Vortex. It serves as both a design specification and implementation guide for humans and AI agents working on this project.

_This is mostly Claude-written, but it is also the result of quite a lot of prototyping and research on how to make our benchmarks website not absolutely terrible with respect to load times._

---

## Table of Contents

1. [Overview & Goals](#overview--goals)
2. [Current State & Problems](#current-state--problems)
3. [Architecture Overview](#architecture-overview)
4. [Data Model & Schema](#data-model--schema)
5. [Server Architecture](#server-architecture)
6. [Client Architecture](#client-architecture)
7. [API Design](#api-design)
8. [Deployment & Infrastructure](#deployment--infrastructure)
9. [Implementation Plan](#implementation-plan)
10. [Future Considerations](#future-considerations)

---

## Overview & Goals

### What We're Building

A benchmarks visualization website (https://bench.vortex.dev/) that displays performance data for the Vortex columnar file format across multiple benchmark suites. The site shows time-series-like charts of benchmark results over git commits, allowing users to track performance regressions and improvements.

### Goals

1. **Fast initial load**: Sub-second time to first contentful paint
2. **Interactive charts**: Instant zoom/pan/scroll across entire commit history (up to 5000+ commits)
3. **Real-time updates**: New benchmark results from CI appear within minutes
4. **Maintainability**: Rust-native stack (Leptos) for team familiarity

### Secondary Goals

- Dogfooding: Use Vortex file format for storing benchmark data via DuckDB extension
- Reusability: Architected as a library that others can adapt for their benchmarking needs
- SEO optimization: This would be a nice to have, but speed is paramount
- Mobile-first design: Desktop is primary use case, but this should work on mobile too
- Public API for benchmark data (internal tooling only)

---

## Current State & Problems

### Current Architecture

```
GitHub Actions (CI) -> JSON files -> S3 bucket -> Client downloads entire dataset
```

- Single monolithic JSON file containing all benchmark results (~80MB uncompressed, ~8MB gzipped)
- Separate commits.json for ordering commits by timestamp
- Client downloads everything, then processes in JavaScript
- Data has been manually truncated to ~2000 commits to keep things more manageable (would be ~4500 at this point)

### Pain Points

| Problem                                   | Impact                                             |
| ----------------------------------------- | -------------------------------------------------- |
| 8MB gzipped download                      | 10-15 seconds on fast internet                     |
| Client-side JSON parsing + join           | Additional 10 seconds processing                   |
| No incremental loading                    | Must wait for everything before seeing anything    |
| Schema changes require JSON restructuring | Difficult to add new targets/benchmarks            |
| No server-side computation                | Summary statistics computed client-side repeatedly |
| Manual data truncation                    | Losing historical data to keep site usable         |

### Data Volume

- ~2000 commits currently (truncated), would be ~4500+ without truncation
- 10+ benchmark groups (TPC-H at multiple scale factors, ClickBench, compression, random access, various micro-benchmarks)
  - TPC-H groups: 19 queries each (× multiple scale factors)
  - ClickBench: 44 queries
  - etc.
- ~200 individual charts across all groups
- Series names vary by group (e.g., "vortex", "parquet", "duckdb:vortex", "datafusion:parquet")

---

## Architecture Overview

### High-Level Design

```
┌─────────────────┐          ┌─────────────────────────────────────────────┐
│  GitHub Actions │  POST    │              AWS Infrastructure             │
│  (benchmark CI) │ ───────► │                                             │
└─────────────────┘          │  ┌─────────────────────────────────────┐    │
                             │  │           CloudFront (CDN)          │    │
         Users ─────────────►│  │   - Caches HTML/WASM/static assets  │    │
        (global)             │  │   - Caches API responses (60s TTL)  │    │
                             │  └──────────────┬──────────────────────┘    │
                             │                 │                           │
                             │                 ▼                           │
                             │  ┌─────────────────────────────────────┐    │
                             │  │         EC2 / ECS (Leptos)          │    │
                             │  │                                     │    │
                             │  │  ┌───────────────────────────────┐  │    │
                             │  │  │     DuckDB (embedded)         │  │    │
                             │  │  │     + Vortex Extension        │  │    │
                             │  │  │                               │  │    │
                             │  │  │  commits.vortex               │  │    │
                             │  │  │  compression.vortex           │  │    │
                             │  │  │  tpch_sf1.vortex              │  │    │
                             │  │  │  tpch_sf10.vortex             │  │    │
                             │  │  │  clickbench.vortex            │  │    │
                             │  │  │  random_access.vortex         │  │    │
                             │  │  └───────────────────────────────┘  │    │
                             │  └─────────────────────────────────────┘    │
                             │                 │                           │
                             │                 ▼                           │
                             │  ┌─────────────────────────────────────┐    │
                             │  │              S3 Bucket              │    │
                             │  │   - Vortex file backups (hourly)    │    │
                             │  │   - Disaster recovery               │    │
                             │  └─────────────────────────────────────┘    │
                             └─────────────────────────────────────────────┘
```

## Key Architectural Decisions

### Decision 1: Server-Side DuckDB (Optionally with Vortex Extension)

**Choice**: Embedded DuckDB on server, optionally using Vortex extension for storage.

**Alternatives Considered**:

- Client-side DuckDB-WASM: Adds complexity, requires shipping data to client
- Static JSON files on CDN: Current approach, proven to be too slow
- PostgreSQL/MySQL: Overkill for this workload

**Rationale**:

- DuckDB is extremely fast for analytical queries (~5ms for typical chart query)
- Vortex extension lets us dogfood our format, but plain DuckDB works too
- Schema evolution is trivial (`ALTER TABLE ADD COLUMN`)
- Keep in mind future library extraction: storage should be swappable

### Decision 2: Leptos with Islands Architecture

**Choice**: Leptos SSR with `#[island]` components for interactive charts

**Alternatives Considered**:

- Dioxus: Better cross-platform, but weaker SSR/streaming support
- Yew: No streaming SSR, less active development
- Next.js/React: Team prefers Rust, would require JS expertise

**Rationale**:

- SSR means fast initial paint (HTML renders before WASM loads)
- Islands keep WASM bundle small (only interactive parts ship as WASM)
- Fine-grained reactivity makes chart updates efficient
- Team already knows Rust

### Decision 3: Progressive Data Loading

**Choice**: SSR with most recent N commits (configurable, default ~50), lazy-load full history on demand per chart

**Alternatives Considered**:

- Load all data upfront: 20-30MB initial payload for large groups
- Paginated loading: Poor UX for time-series charts
- Virtual scrolling of data: Complex, doesn't match use case

**Rationale**:

- Initial HTML is small (~500KB for 44 charts × 50 commits)
- Users see useful content immediately
- Full history loads only when user explicitly needs it
- Each chart loads independently (don't pay for charts you don't view)
- N is configurable (25-100 range, tune based on testing)

### Decision 4: No Materialized Views

**Choice**: Direct queries against Vortex tables with indexes

**Alternatives Considered**:

- 1000+ materialized views (one per chart): Complexity, refresh overhead
- Pre-computed JSON cache: Loses benefits of SQL, cache invalidation issues

**Rationale**:

- Data volume is small (~100MB total across all groups)
- DuckDB queries complete in <10ms with proper indexes
- Materialized views add complexity without meaningful performance gain
- Easier to add new charts/series without maintaining view definitions

---

## Data Model & Schema

### Overview

Each benchmark group is stored as a separate table (or Vortex file when using the Vortex extension). All tables share a common `commits` table for ordering.

**Important**: All benchmark measurements are stored as **unsigned 64-bit integers** (typically nanoseconds). Conversion to human-readable units (seconds, milliseconds, MB/s) happens at display time. This preserves precision and simplifies storage.

**Sparse Data**: Not all commits have benchmark data. The `commits` table contains all commits, but benchmark tables only have rows for commits where benchmarks actually ran. Queries use LEFT JOIN to include all commits, with NULL values for missing benchmark data.

### Commits Table

```sql
-- commits table
-- Stores git commit metadata for ordering and display
CREATE TABLE commits (
    commit_hash VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    message VARCHAR,
    author VARCHAR
);

-- Index for efficient "most recent N commits" queries
CREATE INDEX idx_commits_timestamp ON commits(timestamp DESC);
```

### Benchmark Group Tables

Each benchmark group follows one of two patterns. Series columns are named dynamically based on the targets being compared (e.g., "vortex", "parquet", "duckdb_vortex", "datafusion_parquet").

**Note on series naming**: Series names may be compound (e.g., "duckdb:vortex" means "DuckDB engine reading Vortex format"). In SQL column names, colons are replaced with underscores.

#### Pattern A: Single Chart (e.g., compression, random_access)

```sql
-- compression table
-- No chart dimension - just one chart per group
-- All values are u64 (nanoseconds, bytes, etc.)
CREATE TABLE compression (
    commit_hash VARCHAR NOT NULL REFERENCES commits(commit_hash),
    -- Series columns - names vary by group
    -- Stored as UBIGINT (u64), converted to display units in UI
    vortex_throughput_ns UBIGINT,      -- nanoseconds per operation
    parquet_throughput_ns UBIGINT,
    lance_throughput_ns UBIGINT,
    vortex_compressed_bytes UBIGINT,   -- bytes
    parquet_compressed_bytes UBIGINT,
    lance_compressed_bytes UBIGINT,
    PRIMARY KEY (commit_hash)
);
```

#### Pattern B: Multiple Charts (e.g., TPC-H, ClickBench)

```sql
-- tpch_sf1 table
-- Multiple charts (queries), each showing multiple series (targets)
-- Series names can be compound: "duckdb_vortex" = DuckDB engine + Vortex format
CREATE TABLE tpch_sf1 (
    commit_hash VARCHAR NOT NULL REFERENCES commits(commit_hash),
    chart VARCHAR NOT NULL,  -- 'q1', 'q2', ..., 'q19'
    -- Series columns (nullable for sparse data)
    -- All times stored as nanoseconds (u64)
    vortex_ns UBIGINT,           -- Vortex native reader
    parquet_ns UBIGINT,          -- Parquet native reader
    duckdb_vortex_ns UBIGINT,    -- DuckDB reading Vortex
    duckdb_parquet_ns UBIGINT,   -- DuckDB reading Parquet
    datafusion_vortex_ns UBIGINT,
    datafusion_parquet_ns UBIGINT,
    lance_ns UBIGINT,
    PRIMARY KEY (commit_hash, chart)
);

-- Index for filtering by chart
CREATE INDEX idx_tpch_sf1_chart ON tpch_sf1(chart);
```

```sql
-- clickbench table
CREATE TABLE clickbench (
    commit_hash VARCHAR NOT NULL REFERENCES commits(commit_hash),
    chart VARCHAR NOT NULL,  -- 'q0', 'q1', ..., 'q43'
    vortex_ns UBIGINT,
    parquet_ns UBIGINT,
    duckdb_vortex_ns UBIGINT,
    duckdb_parquet_ns UBIGINT,
    PRIMARY KEY (commit_hash, chart)
);

CREATE INDEX idx_clickbench_chart ON clickbench(chart);
```

### Schema Evolution

Adding a new target (series) to an existing benchmark group:

```sql
ALTER TABLE tpch_sf1 ADD COLUMN arrow_ns UBIGINT;
```

- New column is nullable by default (existing rows have NULL)
- No data migration required
- UI picks up new series from configuration

Adding a new chart to an existing group:

```sql
INSERT INTO clickbench (commit_hash, chart, vortex_ns, parquet_ns)
VALUES ('abc123', 'q44', 150000000, 200000000);
```

Adding a new benchmark group:

1. Create table with appropriate schema
2. Add group metadata to configuration (see [Group Configuration](#group-configuration))
3. CI starts posting results to new endpoint

### Group Configuration

Store benchmark group metadata in a configuration file or table:

```rust
// benchmark_groups.rs
pub struct BenchmarkGroup {
    pub id: &'static str,           // "tpch_sf1"
    pub display_name: &'static str, // "TPC-H Scale Factor 1"
    pub chart_column: Option<&'static str>, // Some("chart") or None
    pub series: Vec<SeriesConfig>,
    pub summary_type: SummaryType,  // How to compute summary stats
}

pub struct SeriesConfig {
    pub column: &'static str,       // "duckdb_vortex_ns" (SQL column name)
    pub display_name: &'static str, // "DuckDB + Vortex" (human readable)
    pub color: &'static str,        // "#3b82f6"
    pub unit: MeasurementUnit,      // How to convert/display values
}

/// All benchmark data is stored as u64. This enum defines how to display it.
pub enum MeasurementUnit {
    Nanoseconds,        // Display as seconds, ms, or μs depending on magnitude
    Bytes,              // Display as B, KB, MB, GB
    BytesPerSecond,     // Throughput: display as MB/s, GB/s
    Ratio,              // Compression ratio: value / 1_000_000 (stored as ratio * 1e6)
    Count,              // Raw count, no conversion
}

impl MeasurementUnit {
    /// Convert u64 stored value to f64 display value with appropriate unit
    pub fn format(&self, value: u64) -> (f64, &'static str) {
        match self {
            MeasurementUnit::Nanoseconds => {
                let ns = value as f64;
                if ns >= 1e9 { (ns / 1e9, "s") }
                else if ns >= 1e6 { (ns / 1e6, "ms") }
                else if ns >= 1e3 { (ns / 1e3, "μs") }
                else { (ns, "ns") }
            }
            MeasurementUnit::Bytes => {
                let b = value as f64;
                if b >= 1e9 { (b / 1e9, "GB") }
                else if b >= 1e6 { (b / 1e6, "MB") }
                else if b >= 1e3 { (b / 1e3, "KB") }
                else { (b, "B") }
            }
            // ... etc
        }
    }
}

pub enum SummaryType {
    GeometricMean,  // For TPC-H, ClickBench (ratios/times)
    ArithmeticMean, // For throughput benchmarks
    Custom(fn(&[u64]) -> f64),
}
```

---

## Server Architecture

### Technology Stack

- **Framework**: Leptos 0.8+ with Axum backend
- **Database**: DuckDB (embedded), optionally with Vortex extension
- **Runtime**: Tokio async runtime

### Project Structure

```
bench-website/
├── Cargo.toml
├── src/
│   ├── main.rs              # Entry point, Axum router setup
│   ├── lib.rs               # Leptos app root
│   ├── db/
│   │   ├── mod.rs
│   │   ├── connection.rs    # DuckDB connection pool
│   │   ├── queries.rs       # SQL query functions
│   │   └── models.rs        # Data structures
│   ├── api/
│   │   ├── mod.rs
│   │   ├── ingest.rs        # POST endpoint for CI
│   │   └── charts.rs        # Server functions for chart data
│   ├── components/
│   │   ├── mod.rs
│   │   ├── app.rs           # Root app component
│   │   ├── layout.rs        # Navigation, layout
│   │   ├── group_page.rs    # Benchmark group page
│   │   ├── chart.rs         # Interactive chart island
│   │   └── summary.rs       # Summary statistics component
│   └── config/
│       ├── mod.rs
│       └── groups.rs        # Benchmark group definitions
├── style/
│   └── main.css
└── data/                    # Vortex files (gitignored, populated at runtime)
    ├── commits.vortex
    ├── compression.vortex
    ├── tpch_sf1.vortex
    └── ...
```

### DuckDB Connection Management

```rust
// src/db/connection.rs
use duckdb::{Connection, Result};
use std::sync::Arc;

/// DuckDB connection wrapper
///
/// DuckDB supports concurrent reads AND writes from the same process
/// (MVCC handles isolation). We use Arc to share the connection across
/// async tasks. DuckDB's internal locking handles thread safety.
pub struct DbPool {
    conn: Arc<Connection>,
}

impl DbPool {
    pub fn new(config: &StorageConfig) -> Result<Self> {
        let conn = Connection::open(&config.database_path)?;

        // Optionally load Vortex extension if configured
        if config.use_vortex_extension {
            conn.execute("INSTALL vortex FROM 'path/to/extension'; LOAD vortex;", [])?;
        }

        // Create tables if they don't exist
        Self::init_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(conn),
        })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS commits (
                commit_hash VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                message VARCHAR,
                author VARCHAR
            );
            CREATE INDEX IF NOT EXISTS idx_commits_timestamp
                ON commits(timestamp DESC);
        "#)?;
        Ok(())
    }

    /// Execute a query (DuckDB handles concurrent access internally)
    pub fn query<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        f(&self.conn)
    }
}

/// Configuration for storage backend (supports library reuse)
pub struct StorageConfig {
    pub database_path: String,      // ":memory:" or file path
    pub use_vortex_extension: bool, // false for standard DuckDB
    pub vortex_extension_path: Option<String>,
}
```

### Ingest Endpoint

```rust
// src/api/ingest.rs
use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct IngestRequest {
    pub group: String,              // "tpch_sf1", "clickbench", etc.
    pub commit_hash: String,        // git commit SHA
    pub commit_timestamp: i64,      // Unix timestamp (seconds)
    pub commit_message: Option<String>,
    pub commit_author: Option<String>,
    pub chart: Option<String>,      // None for single-chart groups, Some("q1") for multi-chart
    pub results: HashMap<String, u64>,  // series_name -> value (always u64)
}

/// POST /api/ingest
///
/// Called by GitHub Actions after each benchmark run.
/// DuckDB handles concurrent writes via MVCC.
pub async fn ingest_benchmark(
    State(db): State<DbPool>,
    Json(req): Json<IngestRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Verify CI token (from header)
    // ... token verification ...

    // Upsert commit metadata
    db.query(|conn| {
        conn.execute(
            r#"INSERT INTO commits (commit_hash, timestamp, message, author)
               VALUES (?, to_timestamp(?), ?, ?)
               ON CONFLICT (commit_hash) DO NOTHING"#,
            params![
                &req.commit_hash,
                req.commit_timestamp,
                &req.commit_message,
                &req.commit_author,
            ],
        )
    }).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Build dynamic INSERT for benchmark results
    let columns: Vec<_> = req.results.keys().map(|s| s.as_str()).collect();
    let placeholders: Vec<_> = columns.iter().map(|_| "?").collect();

    let sql = if let Some(chart) = &req.chart {
        format!(
            r#"INSERT INTO {group} (commit_hash, chart, {cols})
               VALUES (?, ?, {placeholders})
               ON CONFLICT (commit_hash, chart) DO UPDATE SET {updates}"#,
            group = req.group,
            cols = columns.join(", "),
            placeholders = placeholders.join(", "),
            updates = columns.iter().map(|c| format!("{c} = EXCLUDED.{c}")).collect::<Vec<_>>().join(", "),
        )
    } else {
        format!(
            r#"INSERT INTO {group} (commit_hash, {cols})
               VALUES (?, {placeholders})
               ON CONFLICT (commit_hash) DO UPDATE SET {updates}"#,
            group = req.group,
            cols = columns.join(", "),
            placeholders = placeholders.join(", "),
            updates = columns.iter().map(|c| format!("{c} = EXCLUDED.{c}")).collect::<Vec<_>>().join(", "),
        )
    };

    db.query(|conn| {
        // Execute with parameters...
        // Note: actual implementation needs to handle dynamic params
    }).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Trigger async backup to S3 (don't block response)
    tokio::spawn(backup_to_s3(req.group.clone()));

    Ok(StatusCode::OK)
}
```

### Example POST Requests from CI

**Single-chart benchmark (compression):**

```bash
curl -X POST https://bench.vortex.dev/api/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${BENCH_CI_TOKEN}" \
  -d '{
    "group": "compression",
    "commit_hash": "abc123def456789",
    "commit_timestamp": 1705968000,
    "commit_message": "feat: improve compression ratio",
    "commit_author": "developer@example.com",
    "results": {
      "vortex_compress_ns": 1500000000,
      "parquet_compress_ns": 2100000000,
      "vortex_decompress_ns": 800000000,
      "parquet_decompress_ns": 950000000,
      "vortex_size_bytes": 52428800,
      "parquet_size_bytes": 58720256
    }
  }'
```

**Multi-chart benchmark group, single query (TPC-H q1):**

```bash
curl -X POST https://bench.vortex.dev/api/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${BENCH_CI_TOKEN}" \
  -d '{
    "group": "tpch_sf1",
    "commit_hash": "abc123def456789",
    "commit_timestamp": 1705968000,
    "commit_message": "feat: improve compression ratio",
    "commit_author": "developer@example.com",
    "chart": "q1",
    "results": {
      "vortex_ns": 150000000,
      "parquet_ns": 200000000,
      "duckdb_vortex_ns": 180000000,
      "duckdb_parquet_ns": 175000000,
      "datafusion_vortex_ns": 220000000,
      "datafusion_parquet_ns": 210000000
    }
  }'
```

**Multi-chart benchmark group, all queries at once (recommended):**

For efficiency, CI can POST multiple charts at once:

```bash
curl -X POST https://bench.vortex.dev/api/ingest/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${BENCH_CI_TOKEN}" \
  -d '{
    "group": "tpch_sf1",
    "commit_hash": "abc123def456789",
    "commit_timestamp": 1705968000,
    "commit_message": "feat: improve compression ratio",
    "commit_author": "developer@example.com",
    "charts": {
      "q1": {
        "vortex_ns": 150000000,
        "parquet_ns": 200000000,
        "duckdb_vortex_ns": 180000000
      },
      "q2": {
        "vortex_ns": 85000000,
        "parquet_ns": 120000000,
        "duckdb_vortex_ns": 95000000
      },
      "q3": {
        "vortex_ns": 220000000,
        "parquet_ns": 310000000,
        "duckdb_vortex_ns": 250000000
      }
    }
  }'
```

**GitHub Actions workflow snippet:**

```yaml
# .github/workflows/benchmarks.yml
- name: Run TPC-H Benchmarks
  run: cargo bench --bench tpch -- --output json > results.json

- name: Upload Results
  env:
    BENCH_CI_TOKEN: ${{ secrets.BENCH_CI_TOKEN }}
  run: |
    # Parse results and POST to benchmark server
    # This script transforms benchmark output to our API format
    python scripts/upload_benchmarks.py \
      --group tpch_sf1 \
      --commit ${{ github.sha }} \
      --timestamp $(git show -s --format=%ct ${{ github.sha }}) \
      --message "$(git show -s --format=%s ${{ github.sha }})" \
      --author "$(git show -s --format=%ae ${{ github.sha }})" \
      --results results.json \
      --endpoint https://bench.vortex.dev/api/ingest
```

---

## Client Architecture

### Rendering Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Initial Request                               │
│                                                                         │
│  1. Server receives request for /benchmarks/clickbench                  │
│                                                                         │
│  2. Leptos SSR queries DuckDB:                                          │
│     WITH recent AS (                                                    │
│       SELECT commit_hash, timestamp FROM commits                        │
│       ORDER BY timestamp DESC LIMIT {N}  -- configurable, e.g. 50       │
│     )                                                                   │
│     SELECT * FROM clickbench JOIN recent USING (commit_hash)            │
│                                                                         │
│  3. Server renders complete HTML with:                                  │
│     - Navigation, layout (static)                                       │
│     - 44 charts with N data points each (SSR'd canvas/svg)              │
│     - Island markers for hydration                                      │
│     - Serialized props for each island                                  │
│                                                                         │
│  4. HTML streams to browser (~500KB for N=50)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Browser Receives                              │
│                                                                         │
│  5. HTML renders immediately - user sees charts with N commits          │
│                                                                         │
│  6. WASM bundle loads (~300KB gzipped)                                  │
│                                                                         │
│  7. Islands hydrate - charts become interactive                         │
│     - Zoom/pan works within N-commit range                              │
│     - "Show full history" button enabled                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (on user action)
┌─────────────────────────────────────────────────────────────────────────┐
│                        Lazy Load Full History                           │
│                                                                         │
│  8. User clicks "Show full history" on Chart 7                          │
│                                                                         │
│  9. Client calls server function get_full_chart_history("q6")           │
│                                                                         │
│  10. Server queries DuckDB for all commits for that chart               │
│                                                                         │
│  11. ~500KB of data returns for that one chart (5000 commits)           │
│                                                                         │
│  12. Chart re-renders with full history                                 │
│      User can now zoom/pan across entire commit range                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
App
├── Layout
│   ├── Navigation (static)
│   │   └── Group links (compression, tpch_sf1, clickbench, ...)
│   └── Main Content
│       └── <Outlet /> (router)
│
└── Routes
    ├── / -> HomePage (static)
    │
    ├── /benchmarks/:group -> GroupPage
    │   ├── GroupHeader (static)
    │   │   ├── Title
    │   │   └── Summary statistics
    │   │
    │   └── ChartList
    │       ├── ChartWithLazyHistory [island] (chart=q1)
    │       ├── ChartWithLazyHistory [island] (chart=q2)
    │       ├── ...
    │       └── ChartWithLazyHistory [island] (chart=qN)
    │
    └── /compare -> ComparePage (future)
```

### Key Components

#### GroupPage (Server Component)

```rust
// src/components/group_page.rs

/// Configurable initial commit count (tune based on testing)
const INITIAL_COMMITS: usize = 50;  // Could be 25-100

#[component]
pub fn GroupPage(group_id: String) -> impl IntoView {
    // This runs on the server during SSR
    let group_config = get_group_config(&group_id);

    // Fetch most recent N commits of data for ALL charts in this group
    let initial_data = create_resource(
        move || group_id.clone(),
        |group| async move {
            get_initial_group_data(group, INITIAL_COMMITS).await
        }
    );

    view! {
        <div class="group-page">
            <GroupHeader config=group_config.clone() />

            <Suspense fallback=|| view! { <GroupSkeleton /> }>
                {move || initial_data.get().map(|data| {
                    let charts = group_by_chart(data);
                    view! {
                        <div class="charts-grid">
                            <For
                                each=move || charts.clone()
                                key=|(chart_name, _)| chart_name.clone()
                                children=move |(chart_name, initial_points)| {
                                    view! {
                                        <ChartWithLazyHistory
                                            group=group_id.clone()
                                            chart=chart_name
                                            initial_data=initial_points
                                            config=group_config.clone()
                                        />
                                    }
                                }
                            />
                        </div>
                    }
                })}
            </Suspense>
        </div>
    }
}
```

#### ChartWithLazyHistory (Island Component)

```rust
// src/components/chart.rs

/// Commit metadata shared across all charts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub timestamp: i64,  // Unix timestamp
    pub message: Option<String>,
}

/// A single measurement point on a chart.
/// Only exists for commits that have actual benchmark data.
/// If a commit has no data for a series, there simply won't be a ChartPoint for it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChartPoint {
    pub commit_idx: usize,      // Index into CommitInfo array
    pub series: String,         // e.g., "vortex_ns", "duckdb_parquet_ns"
    pub value: u64,             // Raw measurement (e.g., nanoseconds)
}

/// All data needed to render a chart
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChartData {
    /// All commits in the time range, ordered by timestamp (oldest first).
    /// This includes commits WITHOUT benchmark data - the x-axis should show
    /// all commits consistently, with gaps where data is missing.
    pub commits: Vec<CommitInfo>,
    
    /// Measurement points - ONLY for commits that have data.
    /// Sparse: if a commit has no data for a series, there's no ChartPoint.
    /// When rendering, check if commit_idx exists for the series.
    pub points: Vec<ChartPoint>,
    
    /// Series names available in this chart (e.g., ["vortex_ns", "parquet_ns"])
    pub series_names: Vec<String>,
}

impl ChartData {
    /// Get all points for a specific series.
    /// Returns (commit_idx, value) pairs - only for commits WITH data.
    /// Gaps in commit indices indicate missing data (should show as line breaks).
    pub fn series_points(&self, series: &str) -> Vec<(usize, u64)> {
        self.points
            .iter()
            .filter(|p| p.series == series)
            .map(|p| (p.commit_idx, p.value))
            .collect()
    }
    
    /// Check if a specific commit has data for a series
    pub fn has_data(&self, commit_idx: usize, series: &str) -> bool {
        self.points.iter().any(|p| p.commit_idx == commit_idx && p.series == series)
    }
}

// ============================================================================
// HANDLING SPARSE DATA IN CHARTS
// ============================================================================
// 
// Not all commits have benchmark data. This can happen because:
// - Benchmarks only run on certain branches or commit types
// - A benchmark was added after many commits already existed  
// - CI failures prevented benchmark runs
// - A new series (target) was added to an existing benchmark
//
// The data model handles this by:
// 1. `commits` array contains ALL commits in the time range (from LEFT JOIN)
// 2. `points` array contains ONLY commits with actual measurements
// 3. When rendering, absence of a ChartPoint for a commit_idx = no data
//
// Chart rendering should:
// - Show all commits on x-axis (consistent spacing)
// - Draw line segments only between consecutive commits WITH data
// - Show gaps (line breaks) where data is missing
// - Tooltips should show "No data" for commits without measurements
// - Do NOT interpolate or connect across gaps
// ============================================================================

/// Interactive chart with lazy-loaded full history
///
/// This is an island - ships as WASM for client-side interactivity.
/// Initial data (N commits) comes from SSR. Full history loads on demand.
#[island]
pub fn ChartWithLazyHistory(
    group: String,
    chart: String,
    initial_data: ChartData,
    config: GroupConfig,
) -> impl IntoView {
    let (data, set_data) = create_signal(initial_data.clone());
    let (full_history_loaded, set_full_history_loaded) = create_signal(false);
    let (loading_history, set_loading_history) = create_signal(false);

    // View range as indices into commits array
    let (view_start, set_view_start) = create_signal(0usize);
    let (view_end, set_view_end) = create_signal(initial_data.commits.len());

    let load_full_history = move |_| {
        if full_history_loaded.get() || loading_history.get() {
            return;
        }
        set_loading_history.set(true);

        let group = group.clone();
        let chart = chart.clone();

        spawn_local(async move {
            match get_full_chart_history(group, Some(chart)).await {
                Ok(full_data) => {
                    let len = full_data.commits.len();
                    set_data.set(full_data);
                    set_view_end.set(len);
                    set_full_history_loaded.set(true);
                }
                Err(e) => {
                    log::error!("Failed to load history: {}", e);
                }
            }
            set_loading_history.set(false);
        });
    };

    // Auto-load full history when user pans past available data
    let on_view_change = move |new_start: usize, new_end: usize| {
        if new_start > 0 && !full_history_loaded.get() {
            load_full_history(());
        }
        set_view_start.set(new_start);
        set_view_end.set(new_end);
    };

    view! {
        <div class="chart-container">
            <div class="chart-header">
                <h3 class="chart-title">{&chart}</h3>

                <Show when=move || !full_history_loaded.get()>
                    <button
                        class="load-history-btn"
                        on:click=load_full_history
                        disabled=loading_history
                    >
                        {move || if loading_history.get() { "Loading..." } else { "Show full history" }}
                    </button>
                </Show>

                <Show when=move || full_history_loaded.get()>
                    <span class="history-badge">
                        {move || format!("{} commits", data.get().commits.len())}
                    </span>
                </Show>
            </div>

            <ChartCanvas
                data=data
                view_start=view_start
                view_end=view_end
                config=config.clone()
                on_view_change=on_view_change
            />

            <Show when=move || !full_history_loaded.get()>
                <div class="history-hint">"← Pan left for full history"</div>
            </Show>
        </div>
    }
}
```

#### ChartCanvas (Chart Rendering)

For charting, use one of these approaches:

**Option A: plotters + Canvas (Recommended)**

```rust
use plotters::prelude::*;
use plotters_canvas::CanvasBackend;

#[component]
fn ChartCanvas(
    data: ReadSignal<ChartData>,
    view_start: ReadSignal<usize>,
    view_end: ReadSignal<usize>,
    config: GroupConfig,
    on_view_change: impl Fn(usize, usize) + 'static,
) -> impl IntoView {
    let canvas_ref = create_node_ref::<Canvas>();

    create_effect(move |_| {
        let Some(canvas) = canvas_ref.get() else { return };
        let chart_data = data.get();
        let start = view_start.get();
        let end = view_end.get().min(chart_data.commits.len());

        let backend = CanvasBackend::with_canvas_object(canvas.clone()).unwrap();
        let root = backend.into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Draw each series - handle sparse data by drawing line segments
        // only between consecutive commits that have data
        for series_name in &chart_data.series_names {
            let points = chart_data.series_points(series_name);
            // Filter to visible range [start, end)
            let visible: Vec<_> = points.iter()
                .filter(|(idx, _)| *idx >= start && *idx < end)
                .collect();
            
            // Draw line segments between consecutive points
            // Gaps in commit indices mean missing data - don't connect across gaps
            for window in visible.windows(2) {
                let (idx1, val1) = window[0];
                let (idx2, val2) = window[1];
                // Only draw line if commits are adjacent (no gap)
                // Or always draw lines and let gaps show as visual discontinuities
                // ... plotters line drawing ...
            }
        }
        }
    });

    let on_wheel = move |e: WheelEvent| {
        e.prevent_default();
        // Zoom logic
    };

    view! {
        <canvas
            node_ref=canvas_ref
            width="800"
            height="400"
            on:wheel=on_wheel
        />
    }
}
```

**Option B: Custom SVG**

Alternative if you need more control over rendering. Build SVG elements directly with Leptos's `view!` macro, using computed scales for x/y positioning. More verbose but avoids external dependencies.

### Data Refresh

Poll for updates periodically (e.g., every 60 seconds) or rely on users refreshing the page. Since benchmarks don't run that frequently, simple polling is sufficient.

---

## API Design

### Server Functions (Leptos)

```rust
// src/api/charts.rs

/// Number of commits to show in initial view (configurable)
const INITIAL_COMMIT_COUNT: usize = 50;

/// Get initial data for a benchmark group (most recent N commits)
/// Called during SSR to populate initial page
#[server(GetInitialGroupData)]
pub async fn get_initial_group_data(
    group: String,
) -> Result<GroupData, ServerFnError> {
    let db = use_context::<DbPool>().unwrap();
    let config = get_group_config(&group)?;

    // Use LEFT JOIN: we want ALL recent commits, even those without benchmark data.
    // This ensures the x-axis is consistent across all charts.
    // Commits without data will have NULL values for series columns.
    let sql = if config.chart_column.is_some() {
        format!(r#"
            WITH recent AS (
                SELECT commit_hash, timestamp, message
                FROM commits
                ORDER BY timestamp DESC
                LIMIT {limit}
            )
            SELECT r.commit_hash, r.timestamp, r.message, b.chart, b.*
            FROM recent r
            LEFT JOIN {group} b ON r.commit_hash = b.commit_hash
            ORDER BY b.chart, r.timestamp DESC
        "#, limit = INITIAL_COMMIT_COUNT, group = group)
    } else {
        format!(r#"
            WITH recent AS (
                SELECT commit_hash, timestamp, message
                FROM commits
                ORDER BY timestamp DESC
                LIMIT {limit}
            )
            SELECT r.commit_hash, r.timestamp, r.message, b.*
            FROM recent r
            LEFT JOIN {group} b ON r.commit_hash = b.commit_hash
            ORDER BY r.timestamp DESC
        "#, limit = INITIAL_COMMIT_COUNT, group = group)
    };

    let rows = db.query(|conn| conn.query(&sql, []))?;
    Ok(GroupData::from_rows(rows, &config))
}

/// Get full history for a single chart
/// Called client-side when user requests full history
#[server(GetFullChartHistory)]
pub async fn get_full_chart_history(
    group: String,
    chart: Option<String>,
) -> Result<ChartData, ServerFnError> {
    let db = use_context::<DbPool>().unwrap();

    // Use LEFT JOIN: include all commits, benchmark data may be NULL.
    // For multi-chart groups, filter by chart in the ON clause to get
    // NULL for commits that don't have data for this specific chart.
    let sql = if let Some(ref chart_name) = chart {
        format!(r#"
            SELECT c.commit_hash, c.timestamp, c.message, b.*
            FROM commits c
            LEFT JOIN {group} b ON c.commit_hash = b.commit_hash 
                AND b.chart = ?
            ORDER BY c.timestamp ASC
        "#, group = group)
    } else {
        format!(r#"
            SELECT c.commit_hash, c.timestamp, c.message, b.*
            FROM commits c
            LEFT JOIN {group} b ON c.commit_hash = b.commit_hash
            ORDER BY c.timestamp ASC
        "#, group = group)
    };

    let rows = db.query(|conn| {
        if let Some(chart_name) = chart {
            conn.query(&sql, [&chart_name])
        } else {
            conn.query(&sql, [])
        }
    })?;

    Ok(ChartData::from_rows(rows))
}

/// Get summary statistics for a benchmark group
#[server(GetGroupSummary)]
pub async fn get_group_summary(
    group: String,
) -> Result<GroupSummary, ServerFnError> {
    let db = use_context::<DbPool>().unwrap();
    let config = get_group_config(&group)?;

    // Compute statistics over recent commits
    // Note: geometric mean requires special handling for u64 values
    let sql = format!(r#"
        WITH recent AS (
            SELECT commit_hash FROM commits
            ORDER BY timestamp DESC
            LIMIT 100  -- Summary over last 100 commits
        )
        SELECT
            AVG(CAST(vortex_ns AS DOUBLE)) as vortex_avg,
            EXP(AVG(LN(CAST(NULLIF(vortex_ns, 0) AS DOUBLE)))) as vortex_geomean,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vortex_ns) as vortex_median,
            -- ... same for other series ...
        FROM {group}
        WHERE commit_hash IN (SELECT commit_hash FROM recent)
    "#, group = group);

    // ...
}
```

### REST Endpoint (CI Ingest)

```
POST /api/ingest
Content-Type: application/json
Authorization: Bearer <CI_TOKEN>

{
    "group": "tpch_sf1",
    "commit_hash": "abc123def456",
    "commit_timestamp": 1705968000,
    "chart": "q1",
    "results": {
        "vortex_ns": 150000000,
        "parquet_ns": 200000000,
        "lance_ns": 180000000
    }
}

Response: 200 OK
```

---

## Deployment & Infrastructure

### AWS Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              AWS Account                               │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        CloudFront                               │   │
│  │  Distribution: bench.vortex.dev                                 │   │
│  │  - Origin: ALB                                                  │   │
│  │  - Cache behaviors:                                             │   │
│  │    - /api/*: TTL 60s, stale-while-revalidate                    │   │
│  │    - /pkg/*: TTL 1 year (WASM, immutable)                       │   │
│  │    - /*: TTL 60s (HTML)                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                       │
│                                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Application Load Balancer                    │   │
│  │  - Health check: /health                                        │   │
│  │  - Target: ECS service                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                       │
│                                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         ECS Fargate                             │   │
│  │  Service: bench-website                                         │   │
│  │  - Task: 1 vCPU, 2GB RAM                                        │   │
│  │  - Container: bench-website:latest                              │   │
│  │  - Volume: EFS mount at /data                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                       │
│                                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         EFS (Elastic File System)               │   │
│  │  - Stores Vortex files                                          │   │
│  │  - Persists across container restarts                           │   │
│  │  - Single-AZ (cost savings, acceptable for this use case)       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                              S3                                 │   │
│  │  Bucket: vortex-benchmarks-backup                               │   │
│  │  - Hourly backups of Vortex files                               │   │
│  │  - Lifecycle: Delete after 30 days                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Alternative: Simpler Single EC2

For lower traffic/budget:

```
CloudFront -> EC2 (t3.medium) -> Local EBS for data
                             -> S3 for backups
```

This is simpler and cheaper (~$30/month) but less resilient. Acceptable for internal tooling.

### Environment Variables

```bash
# Required
BENCH_DATA_DIR=/data              # Path to Vortex files
BENCH_CI_TOKEN=<secret>           # Token for CI authentication
BENCH_S3_BUCKET=vortex-benchmarks-backup

# Optional
BENCH_PORT=3000
BENCH_LOG_LEVEL=info
RUST_LOG=bench_website=debug
```

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy Benchmarks Website

on:
  push:
    branches: [main]
    paths:
      - "bench-website/**"

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t bench-website ./bench-website

      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker tag bench-website:latest $ECR_REGISTRY/bench-website:latest
          docker push $ECR_REGISTRY/bench-website:latest

      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster bench --service bench-website --force-new-deployment
```

### Dockerfile

```dockerfile
# Build stage
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# Install wasm target for Leptos client
RUN rustup target add wasm32-unknown-unknown

# Install cargo-leptos
RUN cargo install cargo-leptos

# Build both server and client
RUN cargo leptos build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy server binary
COPY --from=builder /app/target/release/bench-website .

# Copy client WASM and assets
COPY --from=builder /app/target/site ./site

# Copy Vortex extension (pre-built)
COPY --from=builder /app/vortex_duckdb.so /usr/local/lib/

ENV BENCH_DATA_DIR=/data
ENV BENCH_PORT=3000

EXPOSE 3000

CMD ["./bench-website"]
```

---

## Implementation Plan

The implementation is divided into phases. Each phase produces a working system; later phases add optimizations. **Always get things working before optimizing.**

### Phase 1: Data Layer Foundation

**Goal**: DuckDB + Vortex storing real benchmark data, queryable via SQL

**Steps**:

1.1. **Set up project structure**

- Create `bench-website/` directory in vortex repo
- Initialize Cargo workspace with leptos, axum, duckdb dependencies
- Create basic module structure (db/, api/, components/)

  1.2. **Implement DuckDB connection layer**

- Create `DbPool` wrapper around DuckDB connection
- Optionally load Vortex extension
- Write helper functions for common query patterns
- Add integration tests with in-memory DuckDB

  1.3. **Define schemas for all benchmark groups**

- Create SQL schema files for each group
- Write migration script to create tables
- Document schema in this file (already done above)

  1.4. **Implement ingest endpoint**

- POST /api/ingest handler
- CI token verification
- INSERT OR REPLACE logic for idempotency
- Test with curl / httpie

  1.5. **Migrate existing JSON data**

- Write one-time migration script
- Parse existing JSON files
- Bulk insert into DuckDB/Vortex tables
- Verify data integrity (row counts, spot checks)

  1.6. **Set up CI integration**

- Modify existing benchmark workflows to POST results
- Add BENCH_CI_TOKEN secret to GitHub
- Test end-to-end: commit -> benchmark -> POST -> verify in DB

**Deliverable**: Running server that accepts benchmark results and stores them in Vortex files. Can query data via DuckDB CLI.

### Phase 2: Basic Website (No Lazy Loading)

**Goal**: Functional website showing all charts with all data

**Steps**:

2.1. **Set up Leptos app shell**

- Configure cargo-leptos
- Create App component with router
- Add basic CSS (can use Tailwind or simple custom CSS)
- Verify SSR works (view page source shows content)

  2.2. **Implement navigation and layout**

- Header with site title
- Sidebar/tabs for benchmark groups
- Responsive layout basics

  2.3. **Create GroupPage component**

- Route: /benchmarks/:group
- Fetch ALL data for group (no lazy loading yet)
- Display loading state via Suspense

  2.4. **Implement basic chart rendering**

- Choose charting approach (plotters or SVG)
- Render static charts (no interactivity yet)
- Display all series with different colors
- Add axis labels, legend

  2.5. **Add chart interactivity (zoom/pan)**

- Convert chart to island component
- Implement mouse wheel zoom
- Implement click-drag pan
- Ensure smooth 60fps rendering

  2.6. **Implement summary statistics**

- Server-side computation (DuckDB aggregates)
- Display above charts
- Different summary types per group (geomean for TPC-H, etc.)

  2.7. **Handle sparse data gracefully**

- Skip null values in series
- Don't break line connections (or show gaps appropriately)
- Show "no data" indicator for missing series

**Deliverable**: Fully functional website. May be slow for ClickBench (loads all 44 charts × 5000 commits) but everything works.

### Phase 3: Progressive Loading Optimization

**Goal**: Fast initial load with lazy history loading

**Steps**:

3.1. **Modify data fetching for initial load**

- Change `get_initial_group_data` to fetch only last 50 commits
- Update GroupPage to pass initial_data to charts
- Verify initial HTML size is reasonable (~500KB for ClickBench)

  3.2. **Implement `GetFullChartHistory` server function**

- Fetches all commits for single chart
- Called from client when user requests full history
- Returns data sorted by timestamp ASC

  3.3. **Add lazy loading UI to ChartWithLazyHistory**

- "Show full history" button
- Loading state indicator
- History loaded indicator (commit count badge)

  3.4. **Implement auto-load on pan**

- Detect when user pans left past available data
- Automatically trigger full history load
- Smooth transition as data expands

  3.5. **Add data refresh polling**

- Poll every 60 seconds for updates
- Only refresh if new data available
- Update charts without full page reload

**Deliverable**: Production-ready website with fast initial load and on-demand history.

### Phase 4: Deployment & Production Hardening

**Goal**: Reliable production deployment on AWS

**Steps**:

4.1. **Create Dockerfile**

- Multi-stage build (builder + runtime)
- Include Vortex DuckDB extension
- Minimize image size

  4.2. **Set up AWS infrastructure**

- Create ECS cluster and service (or EC2 instance)
- Configure EFS for data persistence
- Set up S3 bucket for backups
- Configure CloudFront distribution

  4.3. **Implement backup system**

- Hourly backup to S3 after successful ingests
- Restore script for disaster recovery
- Test backup/restore cycle

  4.4. **Add monitoring and alerting**

- Health check endpoint
- CloudWatch metrics (request latency, error rate)
- Alerts for failures

  4.5. **Configure caching**

- CloudFront cache behaviors
- Cache-Control headers on API responses
- Verify cache invalidation on data updates

  4.6. **Security hardening**

- HTTPS only (CloudFront handles TLS)
- CI token rotation procedure
- Rate limiting on ingest endpoint

  4.7. **Documentation**

- Update this design doc with any changes
- Runbook for common operations
- Architecture diagram for team reference

**Deliverable**: Production deployment at bench.vortex.dev with monitoring and backups.

### Phase 5: Polish & Future Features (Optional)

**Goal**: Nice-to-have improvements

**Steps**:

5.1. **Compare view**

- Select two commits to compare
- Show diff in benchmark results
- Highlight regressions/improvements

  5.2. **Permalink to specific chart/commit**

- URL includes chart and view range
- Share links to specific data points

  5.3. **Export functionality**

- Download chart as PNG/SVG
- Export data as CSV

  5.4. **Performance profiling**

- Measure actual load times
- Identify bottlenecks
- Optimize as needed (may not be necessary)

---

## Library Reusability (Future Work)

This project should eventually be architected so others can use it for their own benchmark visualization needs without requiring Vortex. This is not a priority for initial implementation but should be kept in mind during development.

**Key abstraction points to consider:**

1. **Storage backend** — The DuckDB + Vortex combination should be swappable for plain DuckDB, PostgreSQL, or other databases
2. **Configuration-driven groups** — Benchmark group definitions should be data-driven (config file or similar) rather than hardcoded
3. **Unit conversion** — The `MeasurementUnit` system for converting u64 raw values to display units should be reusable

**Potential crate structure (future):**

```
bench-viz/                    # Library crate (reusable)
├── src/
│   ├── storage/             # Storage trait + implementations
│   ├── config/              # Group/series configuration
│   └── components/          # Leptos components

bench-vortex/                 # Vortex-specific binary
├── src/
│   ├── main.rs
│   └── groups.rs            # Vortex benchmark group definitions
```

For now, implement directly in the vortex repo. Refactor into a library later if there's demand.

---

## Future Considerations

### Scaling Beyond Current Needs

The current design handles ~125MB of data and ~200 charts comfortably. If data grows significantly:

- **More commits (10,000+)**: Consider time-based partitioning of Vortex files (e.g., one file per year)
- **More benchmark groups**: No architectural changes needed, just more files
- **Higher write throughput**: DuckDB handles concurrent writes well, but could add write batching if needed
- **Global low latency**: Deploy read replicas in multiple regions (sync Vortex files to S3, read from nearest region)

### Alternative Charting Libraries

If plotters or custom SVG proves insufficient:

- **uPlot**: Extremely fast (47KB), handles 100k+ points, but requires JS interop
- **ECharts via charming**: Full-featured, but large bundle (~1MB)
- **D3 via wasm-bindgen**: Maximum flexibility, moderate complexity

### Vortex-Specific Optimizations

Since we're dogfooding Vortex:

- **Compression**: Ensure Vortex files use appropriate encoding for benchmark data (u64 integers and strings)
- **Predicate pushdown**: DuckDB + Vortex should push down filters efficiently; verify with EXPLAIN
- **Column pruning**: Queries should only read needed columns; verify Vortex extension does this

---

## Appendix: Quick Reference

### Common Commands

```bash
# Local development
cd bench-website
cargo leptos watch

# Run tests
cargo test

# Build for production
cargo leptos build --release

# Query DuckDB directly (for debugging)
duckdb data/benchmarks.duckdb
> SELECT * FROM tpch_sf1 WHERE chart = 'q1' LIMIT 10;

# Ingest test data
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $BENCH_CI_TOKEN" \
  -d '{"group": "tpch_sf1", "commit_hash": "abc123", "commit_timestamp": 1705968000, "chart": "q1", "results": {"vortex_ns": 100000000}}'
```

### Key Files

| File                      | Purpose                         |
| ------------------------- | ------------------------------- |
| `src/db/connection.rs`    | DuckDB pool and query helpers   |
| `src/api/ingest.rs`       | CI ingest endpoint              |
| `src/api/charts.rs`       | Server functions for chart data |
| `src/components/chart.rs` | Interactive chart island        |
| `src/config/groups.rs`    | Benchmark group definitions     |
| `data/*.vortex`           | Vortex data files (gitignored)  |

### Data Sizes (Estimates)

| Group                  | Charts   | Commits | Rows     | ~Size      |
| ---------------------- | -------- | ------- | -------- | ---------- |
| compression            | 1        | 5000    | 5,000    | 2MB        |
| random_access          | 1        | 5000    | 5,000    | 2MB        |
| tpch_sf1               | 19       | 5000    | 95,000   | 15MB       |
| tpch_sf10              | 19       | 5000    | 95,000   | 15MB       |
| tpch_sf100             | 19       | 5000    | 95,000   | 15MB       |
| clickbench             | 44       | 5000    | 220,000  | 25MB       |
| other micro-benchmarks | ~100     | 5000    | ~500,000 | 50MB       |
| commits                | -        | 5000    | 5,000    | 1MB        |
| **Total**              | **~200** | -       | **~1M**  | **~125MB** |

Note: Actual current data is ~2000 commits (manually truncated). With full history restored, expect ~125MB total.