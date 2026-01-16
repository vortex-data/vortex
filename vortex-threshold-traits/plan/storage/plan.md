# Data Storage & Production

## Overview

The storage layer persists benchmark measurements and produces threshold data for consumption by Vortex at compile time (static dispatch tables) or runtime (configuration files).

## Current Implementation

### Done
- [x] `BenchmarkStorage` trait defining storage interface
- [x] `SqliteStorage` implementation with full query support
- [x] `MeasurementQuery` builder for filtering results
- [x] JSON export for CI artifact collection
- [x] Threshold aggregator that merges multi-arch results
- [x] Rust code generation for `LazyLock` dispatch tables

### Not Done
- [ ] Threshold history tracking for regression detection
- [ ] Cross-commit comparison queries
- [ ] Automatic threshold file updates in PRs
- [ ] Schema versioning and migration
- [ ] Compression for large result sets

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Flow                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Runner (per arch)          Aggregator           Output     │
│  ┌──────────────┐          ┌──────────┐      ┌──────────┐  │
│  │ Measurements │─────────>│  Merge   │─────>│ .rs file │  │
│  │   (JSON)     │          │ Analyze  │      │ thresholds│  │
│  └──────────────┘          └──────────┘      └──────────┘  │
│         │                        │                          │
│         v                        v                          │
│  ┌──────────────┐          ┌──────────┐                    │
│  │   SQLite     │<─────────│ History  │                    │
│  │   (local)    │          │ Tracking │                    │
│  └──────────────┘          └──────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Schema

### Measurements Table (SQLite)

```sql
CREATE TABLE measurements (
    id INTEGER PRIMARY KEY,
    algorithm TEXT NOT NULL,
    variant TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    parameter_value INTEGER NOT NULL,
    cpu_class TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    median_ns REAL NOT NULL,
    mean_ns REAL NOT NULL,
    stddev_ns REAL NOT NULL,
    min_ns REAL NOT NULL,
    max_ns REAL NOT NULL,
    sample_count INTEGER NOT NULL
);

CREATE INDEX idx_algorithm ON measurements(algorithm);
CREATE INDEX idx_cpu_class ON measurements(cpu_class);
CREATE INDEX idx_commit ON measurements(commit_hash);
CREATE INDEX idx_timestamp ON measurements(timestamp);
```

### Thresholds Table (SQLite)

```sql
CREATE TABLE thresholds (
    id INTEGER PRIMARY KEY,
    algorithm TEXT NOT NULL,
    variant_from TEXT NOT NULL,
    variant_to TEXT NOT NULL,
    cpu_class TEXT NOT NULL,
    threshold_value INTEGER NOT NULL,
    commit_hash TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    confidence REAL,  -- statistical confidence in this threshold
    UNIQUE(algorithm, variant_from, variant_to, cpu_class, commit_hash)
);

CREATE INDEX idx_threshold_algo ON thresholds(algorithm);
CREATE INDEX idx_threshold_cpu ON thresholds(cpu_class);
```

## JSON Export Format

```json
{
  "metadata": {
    "commit": "abc123",
    "timestamp": "2024-01-15T10:30:00Z",
    "cpu_class": "IntelSapphire",
    "runner": "intel-sapphire-01"
  },
  "measurements": [
    {
      "algorithm": "popcount",
      "variant": "naive",
      "parameter_name": "input_size",
      "parameter_value": 1024,
      "median_ns": 150.5,
      "mean_ns": 152.3,
      "stddev_ns": 5.2,
      "sample_count": 100
    }
  ],
  "thresholds": [
    {
      "algorithm": "popcount",
      "from_variant": "naive",
      "to_variant": "avx2",
      "threshold": 256,
      "confidence": 0.95
    }
  ]
}
```

## Generated Rust Code

```rust
//! Auto-generated threshold dispatch tables
//! Generated at: 2024-01-15T10:30:00Z
//! Commit: abc123

use std::sync::LazyLock;
use vortex_threshold_traits::CpuClass;

/// Thresholds for popcount algorithm
#[derive(Debug, Clone, Copy)]
pub struct PopcountThresholds {
    /// Switch from naive to avx2 at this input size
    pub naive_to_avx2: usize,
}

pub static POPCOUNT_THRESHOLDS: LazyLock<PopcountThresholds> = LazyLock::new(|| {
    match CpuClass::detect() {
        CpuClass::IntelSapphire => PopcountThresholds { naive_to_avx2: 256 },
        CpuClass::IntelIceLake => PopcountThresholds { naive_to_avx2: 256 },
        CpuClass::AmdGenoa => PopcountThresholds { naive_to_avx2: 512 },
        CpuClass::AmdMilan => PopcountThresholds { naive_to_avx2: 384 },
        CpuClass::Graviton3 => PopcountThresholds { naive_to_avx2: 128 },
        CpuClass::Graviton2 => PopcountThresholds { naive_to_avx2: 192 },
        _ => PopcountThresholds { naive_to_avx2: 256 }, // default
    }
});
```

## Planned Changes

### 1. Threshold History Tracking

Track how thresholds change over time for regression detection:

```rust
impl SqliteStorage {
    /// Get threshold history for regression detection
    fn get_threshold_history(
        &self,
        algorithm: &str,
        from_variant: &str,
        to_variant: &str,
        cpu_class: CpuClass,
        limit: usize,
    ) -> Result<Vec<ThresholdHistoryEntry>>;

    /// Detect if threshold changed significantly
    fn detect_regression(
        &self,
        current: &Threshold,
        history_window: usize,
        change_percent: f64,
    ) -> Option<ThresholdRegression>;
}
```

### 2. Cross-Commit Comparison

```rust
impl SqliteStorage {
    /// Compare thresholds between two commits
    fn compare_commits(
        &self,
        base_commit: &str,
        head_commit: &str,
    ) -> Result<Vec<ThresholdDiff>>;
}

struct ThresholdDiff {
    algorithm: String,
    cpu_class: CpuClass,
    base_threshold: usize,
    head_threshold: usize,
    change_percent: f64,
    is_regression: bool,
}
```

### 3. Schema Migration

```rust
impl SqliteStorage {
    fn migrate(&self) -> Result<()> {
        let version = self.get_schema_version()?;
        match version {
            0 => self.migrate_v0_to_v1()?,
            1 => self.migrate_v1_to_v2()?,
            _ => {}
        }
        Ok(())
    }
}
```

## Open Questions

1. **Storage location**: Where should SQLite DB live? Per-project? Global cache?
2. **Data retention**: How long to keep measurement history?
3. **Aggregation strategy**: How to combine measurements across multiple runs?
4. **Generated code location**: Where should threshold .rs files live?

## Files

- `vortex-threshold-traits/src/storage.rs` - BenchmarkStorage trait
- `vortex-threshold-runner/src/storage/sqlite.rs` - SQLite implementation
- `vortex-threshold-aggregator/src/main.rs` - Code generation

## Next Steps

1. Implement threshold history tracking
2. Add cross-commit comparison
3. Add schema versioning
4. Document storage location conventions
