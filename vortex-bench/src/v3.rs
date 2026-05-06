// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! v3 wire-format records emitted by `--gh-json-v3`.
//!
//! See `benchmarks-website/planning/02-contracts.md` for the discriminated record
//! format and `benchmarks-website/planning/01-schema.md` for the destination
//! tables. The records emitted here are bare: the post-ingest envelope
//! (`run_meta` + `commit`) is added by `scripts/post-ingest.py` before POSTing
//! to `/api/ingest`.
//!
//! This module is purely additive to the existing `gh-json` emission path.

use std::io::Write;
use std::sync::LazyLock;

use serde::Serialize;
use target_lexicon::Triple;

use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;
use crate::compress::CompressOp;
use crate::measurements::CompressionTimingMeasurement;
use crate::measurements::MemoryMeasurement;
use crate::measurements::QueryMeasurement;
use crate::measurements::TimingMeasurement;
use crate::utils::GIT_COMMIT_ID;

/// `(architecture, operating_system, environment)` triple for the host running the benchmark.
///
/// Cached for the lifetime of the process; the host triple does not change.
pub static ENV_TRIPLE: LazyLock<String> = LazyLock::new(|| {
    let host = Triple::host();
    format!(
        "{}-{}-{}",
        host.architecture, host.operating_system, host.environment
    )
});

/// Wire-format kind discriminator. One value per fact table.
///
/// Each variant flattens its inner record next to a `"kind"` field, matching the
/// shape consumed by `/api/ingest`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum V3Record {
    /// SQL query suite measurement (TPC-H/TPC-DS/ClickBench/...).
    QueryMeasurement(QueryMeasurementRecord),
    /// `compress-bench` encode/decode timing.
    CompressionTime(CompressionTimeRecord),
    /// `compress-bench` on-disk size.
    CompressionSize(CompressionSizeRecord),
    /// `random-access-bench` take timing.
    RandomAccessTime(RandomAccessTimeRecord),
    /// `vector-search-bench` cosine-similarity scan run.
    VectorSearchRun(VectorSearchRunRecord),
}

/// A single SQL-query measurement, fused from a [`QueryMeasurement`] and an
/// optional paired [`MemoryMeasurement`].
///
/// Memory fields are populated together (all four or none), matching the
/// `--track-memory` instrumentation.
#[derive(Debug, Clone, Serialize)]
pub struct QueryMeasurementRecord {
    /// 40-hex lowercase commit SHA.
    pub commit_sha: String,
    /// Top-level dataset name (`tpch`, `tpcds`, `clickbench`, ...).
    pub dataset: String,
    /// ClickBench flavor (`partitioned`/`single`) or Public-BI sub-dataset name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_variant: Option<String>,
    /// TPC scale factor (TPC-H / TPC-DS only). Other suites leave this `None`
    /// so live records merge with the migrated v2 history, which never carried
    /// a per-suite scale factor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_factor: Option<String>,
    /// 1-based query index within the suite.
    pub query_idx: u32,
    /// Storage backend the run targeted (`nvme` or `s3`).
    pub storage: String,
    /// Query engine (`datafusion`, `duckdb`, `vortex`, `arrow`).
    pub engine: String,
    /// On-disk format (`parquet`, `vortex-file-compressed`, `lance`, ...).
    pub format: String,
    /// Median per-iteration wall time in nanoseconds.
    pub value_ns: u64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<u64>,
    /// Peak resident-set bytes during the query, when memory tracking was on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_physical: Option<u64>,
    /// Peak virtual-memory bytes during the query, when memory tracking was on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_virtual: Option<u64>,
    /// Resident-set delta across the query, when memory tracking was on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_delta: Option<i64>,
    /// Virtual-memory delta across the query, when memory tracking was on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub virtual_delta: Option<i64>,
    /// Host environment triple (e.g. `x86_64-linux-gnu`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_triple: Option<String>,
}

/// A single encode-or-decode timing from `compress-bench`.
#[derive(Debug, Clone, Serialize)]
pub struct CompressionTimeRecord {
    /// 40-hex lowercase commit SHA.
    pub commit_sha: String,
    /// Compression dataset name.
    pub dataset: String,
    /// Optional dataset variant (reserved; unused at alpha).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_variant: Option<String>,
    /// On-disk format the timing applies to.
    pub format: String,
    /// `encode` or `decode`.
    pub op: String,
    /// Best-of-N wall time in nanoseconds.
    pub value_ns: u64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<u64>,
    /// Host environment triple.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_triple: Option<String>,
}

/// On-disk size of a compressed file from `compress-bench`.
#[derive(Debug, Clone, Serialize)]
pub struct CompressionSizeRecord {
    /// 40-hex lowercase commit SHA.
    pub commit_sha: String,
    /// Compression dataset name.
    pub dataset: String,
    /// Optional dataset variant (reserved; unused at alpha).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_variant: Option<String>,
    /// On-disk format the size applies to.
    pub format: String,
    /// Size in bytes.
    pub value_bytes: u64,
}

/// A single take-time timing from `random-access-bench`.
#[derive(Debug, Clone, Serialize)]
pub struct RandomAccessTimeRecord {
    /// 40-hex lowercase commit SHA.
    pub commit_sha: String,
    /// Random-access dataset name (different namespace from SQL suites).
    pub dataset: String,
    /// On-disk format the timing applies to.
    pub format: String,
    /// Median per-iteration wall time in nanoseconds.
    pub value_ns: u64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<u64>,
    /// Host environment triple.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_triple: Option<String>,
}

/// A single cosine-similarity scan from `vector-search-bench`.
///
/// Carries timing **and** the side counters in one row, mirroring the
/// `vector_search_runs` fact table.
#[derive(Debug, Clone, Serialize)]
pub struct VectorSearchRunRecord {
    /// 40-hex lowercase commit SHA.
    pub commit_sha: String,
    /// Vector dataset name (e.g. `cohere-large-10m`).
    pub dataset: String,
    /// Train-split layout label (e.g. `partitioned`).
    pub layout: String,
    /// Compression flavor label (e.g. `vortex-turboquant`).
    pub flavor: String,
    /// Cosine threshold passed to the scan filter.
    pub threshold: f64,
    /// Median per-iteration wall time in nanoseconds.
    pub value_ns: u64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<u64>,
    /// Number of rows that survived the cosine filter.
    pub matches: u64,
    /// Total rows scanned.
    pub rows_scanned: u64,
    /// Total on-disk bytes scanned.
    pub bytes_scanned: u64,
    /// Number of timed iterations.
    pub iterations: u32,
    /// Host environment triple.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_triple: Option<String>,
}

/// Canonicalize a TPC scale factor string for v3 dim emission.
///
/// Bench-orchestrator passes raw strings like `"1.0"` and `"100.0"`, but the
/// v2 → v3 migrate path canonicalizes integer-valued scale factors to `"1"` and
/// `"100"` (because v2 chart names carried integer-looking values). Live
/// records must use the same canonical form so they merge with migrated
/// history into a single chart group instead of forking off a sibling group
/// keyed on `SF=1.0` vs `SF=1`.
///
/// Falls back to the trimmed input on parse failure or non-finite values, so a
/// scale factor we cannot interpret as a number passes through unchanged
/// rather than being silently rewritten.
fn canonical_tpc_scale_factor(scale_factor: &str) -> String {
    let trimmed = scale_factor.trim();
    match trimmed.parse::<f64>() {
        Ok(value) if value.is_finite() => format!("{value}"),
        _ => scale_factor.to_string(),
    }
}

/// Map a [`BenchmarkDataset`] to the `(dataset, dataset_variant, scale_factor)`
/// triple emitted in `query_measurement` records.
///
/// Mirrors the `Per-suite dim values` table in
/// `benchmarks-website/planning/benchmark-mapping.md`.
pub fn benchmark_dataset_dims(d: &BenchmarkDataset) -> (String, Option<String>, Option<String>) {
    match d {
        BenchmarkDataset::TpcH { scale_factor } => (
            "tpch".to_string(),
            None,
            Some(canonical_tpc_scale_factor(scale_factor)),
        ),
        BenchmarkDataset::TpcDS { scale_factor } => (
            "tpcds".to_string(),
            None,
            Some(canonical_tpc_scale_factor(scale_factor)),
        ),
        // ClickBench: the migrate path leaves `dataset_variant` NULL because
        // v2 record names did not encode flavor, so the live emitter does the
        // same to keep historical and live records in one `clickbench` group.
        // Flavor is fixed per CI matrix entry and recoverable from there.
        BenchmarkDataset::ClickBench { .. } => ("clickbench".to_string(), None, None),
        BenchmarkDataset::PublicBi { name } => ("public-bi".to_string(), Some(name.clone()), None),
        // StatPopGen / PolarSignals: the migrate path (v2 → v3 backfill) does
        // not carry a per-record scale factor for these suites, so writing one
        // here would split each into two groups (sf=NULL historical vs. sf=N
        // live). Drop it to keep live ingests merging into the migrated
        // group. The dataset-level `n_rows` is recoverable from the bench
        // matrix if ever needed.
        BenchmarkDataset::StatPopGen { .. } => ("statpopgen".to_string(), None, None),
        BenchmarkDataset::PolarSignals { .. } => ("polarsignals".to_string(), None, None),
        BenchmarkDataset::Fineweb => ("fineweb".to_string(), None, None),
        BenchmarkDataset::GhArchive => ("gharchive".to_string(), None, None),
    }
}

/// Build a `query_measurement` record by collapsing a [`QueryMeasurement`] and
/// an optional paired [`MemoryMeasurement`] into one wire row.
///
/// The pair is matched by the caller; this function does not search.
pub fn query_measurement_record(
    qm: &QueryMeasurement,
    memory: Option<&MemoryMeasurement>,
) -> V3Record {
    let (dataset, dataset_variant, scale_factor) = benchmark_dataset_dims(&qm.benchmark_dataset);
    let value_ns = duration_as_ns(qm.median_run());
    let all_runtimes_ns = qm.runs.iter().copied().map(duration_as_ns).collect();
    let (peak_physical, peak_virtual, physical_delta, virtual_delta) = match memory {
        Some(m) => (
            Some(m.peak_physical_memory),
            Some(m.peak_virtual_memory),
            Some(m.physical_memory_delta),
            Some(m.virtual_memory_delta),
        ),
        None => (None, None, None, None),
    };
    V3Record::QueryMeasurement(QueryMeasurementRecord {
        commit_sha: GIT_COMMIT_ID.clone(),
        dataset,
        dataset_variant,
        scale_factor,
        query_idx: u32::try_from(qm.query_idx).unwrap_or(u32::MAX),
        storage: qm.storage.clone(),
        engine: engine_label(qm.target.engine).to_string(),
        format: qm.target.format.name().to_string(),
        value_ns,
        all_runtimes_ns,
        peak_physical,
        peak_virtual,
        physical_delta,
        virtual_delta,
        env_triple: Some(ENV_TRIPLE.clone()),
    })
}

/// Build a `compression_time` record from a [`CompressionTimingMeasurement`].
///
/// Caller passes `dataset` (the compress-bench dataset name) and the
/// `op`. `dataset_variant` is reserved and unused at alpha.
///
/// `dataset` is lowercased here to match the v2 → v3 migrate classifier,
/// which stores `dataset = series.to_lowercase()`. Callers like
/// `Dataset::v3_dataset_dims` may therefore return mixed-case names without
/// having to duplicate the case-folding rule.
pub fn compression_time_record(
    timing: &CompressionTimingMeasurement,
    dataset: &str,
    dataset_variant: Option<&str>,
    op: CompressOp,
    all_runtimes_ns: Vec<u64>,
) -> V3Record {
    V3Record::CompressionTime(CompressionTimeRecord {
        commit_sha: GIT_COMMIT_ID.clone(),
        dataset: dataset.to_lowercase(),
        dataset_variant: dataset_variant.map(str::to_string),
        format: timing.format.name().to_string(),
        op: compress_op_label(op).to_string(),
        value_ns: duration_as_ns(timing.time),
        all_runtimes_ns,
        env_triple: Some(ENV_TRIPLE.clone()),
    })
}

/// Build a `compression_size` record.
///
/// `dataset` is lowercased here for the same reason as
/// [`compression_time_record`].
pub fn compression_size_record(
    dataset: &str,
    dataset_variant: Option<&str>,
    format: Format,
    value_bytes: u64,
) -> V3Record {
    V3Record::CompressionSize(CompressionSizeRecord {
        commit_sha: GIT_COMMIT_ID.clone(),
        dataset: dataset.to_lowercase(),
        dataset_variant: dataset_variant.map(str::to_string),
        format: format.name().to_string(),
        value_bytes,
    })
}

/// Build a `random_access_time` record from a [`TimingMeasurement`].
pub fn random_access_record(timing: &TimingMeasurement, dataset: &str) -> V3Record {
    let value_ns = duration_as_ns(timing.median_time());
    let all_runtimes_ns = timing.runs.iter().copied().map(duration_as_ns).collect();
    V3Record::RandomAccessTime(RandomAccessTimeRecord {
        commit_sha: GIT_COMMIT_ID.clone(),
        dataset: dataset.to_string(),
        format: timing.target.format.name().to_string(),
        value_ns,
        all_runtimes_ns,
        env_triple: Some(ENV_TRIPLE.clone()),
    })
}

/// Inputs for [`vector_search_record`]. The caller supplies the per-scan
/// dimensions that don't live on `ScanTiming`.
pub struct VectorSearchDims<'a> {
    /// Vector dataset name (e.g. `cohere-large-10m`).
    pub dataset: &'a str,
    /// Train-split layout label (e.g. `partitioned`).
    pub layout: &'a str,
    /// Compression flavor label (e.g. `vortex-turboquant`).
    pub flavor: &'a str,
    /// Cosine threshold the scan was run with.
    pub threshold: f64,
}

/// Build a `vector_search_run` record. `iterations` is `all_runs.len()`; we keep
/// it explicit since the contract has it as a real column.
pub fn vector_search_record(
    dims: VectorSearchDims<'_>,
    median_ns: u64,
    all_runs_ns: Vec<u64>,
    matches: u64,
    rows_scanned: u64,
    bytes_scanned: u64,
) -> V3Record {
    let iterations = u32::try_from(all_runs_ns.len()).unwrap_or(u32::MAX);
    V3Record::VectorSearchRun(VectorSearchRunRecord {
        commit_sha: GIT_COMMIT_ID.clone(),
        dataset: dims.dataset.to_string(),
        layout: dims.layout.to_string(),
        flavor: dims.flavor.to_string(),
        threshold: dims.threshold,
        value_ns: median_ns,
        all_runtimes_ns: all_runs_ns,
        matches,
        rows_scanned,
        bytes_scanned,
        iterations,
        env_triple: Some(ENV_TRIPLE.clone()),
    })
}

/// Write `records` as JSONL (one JSON object per line) to `writer`.
///
/// JSONL is the on-disk format consumed by `scripts/post-ingest.py`.
pub fn write_jsonl<W: Write>(writer: &mut W, records: &[V3Record]) -> std::io::Result<()> {
    for record in records {
        let line = serde_json::to_string(record).map_err(std::io::Error::other)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Write `records` as JSONL to `path`, creating parent directories as needed.
pub fn write_jsonl_to_path(path: &std::path::Path, records: &[V3Record]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    write_jsonl(&mut file, records)
}

fn duration_as_ns(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

fn engine_label(engine: Engine) -> &'static str {
    match engine {
        Engine::Vortex => "vortex",
        Engine::Arrow => "arrow",
        Engine::DataFusion => "datafusion",
        Engine::DuckDB => "duckdb",
    }
}

fn compress_op_label(op: CompressOp) -> &'static str {
    match op {
        CompressOp::Compress => "encode",
        CompressOp::Decompress => "decode",
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use insta::assert_snapshot;
    use insta::with_settings;

    use super::*;
    use crate::Target;
    use crate::clickbench::Flavor;
    use crate::memory::MemoryMeasurementResult;

    fn redact_env(json: &str) -> String {
        json.replace(ENV_TRIPLE.as_str(), "<env-triple>")
            .replace(GIT_COMMIT_ID.as_str(), "<commit-sha>")
    }

    fn render(record: &V3Record) -> anyhow::Result<String> {
        let json = serde_json::to_string_pretty(record)?;
        Ok(redact_env(&json))
    }

    #[test]
    fn snapshot_query_measurement_with_memory() -> anyhow::Result<()> {
        let qm = QueryMeasurement {
            query_idx: 3,
            target: Target::new(Engine::DataFusion, Format::OnDiskVortex),
            benchmark_dataset: BenchmarkDataset::TpcH {
                scale_factor: "10".to_string(),
            },
            benchmark_runner: "test-runner".to_string(),
            storage: "nvme".to_string(),
            runs: vec![
                Duration::from_nanos(1_000_000),
                Duration::from_nanos(1_200_000),
                Duration::from_nanos(900_000),
            ],
        };
        let mm = MemoryMeasurement::new(
            qm.query_idx,
            qm.target,
            qm.benchmark_dataset.clone(),
            qm.benchmark_runner.clone(),
            qm.storage.clone(),
            MemoryMeasurementResult {
                physical_memory_delta: 1024,
                virtual_memory_delta: 4096,
                peak_physical_memory: 8192,
                peak_virtual_memory: 16384,
            },
        );
        let record = query_measurement_record(&qm, Some(&mm));
        let rendered = render(&record)?;
        with_settings!({snapshot_suffix => "with_memory"}, {
            assert_snapshot!(rendered);
        });
        Ok(())
    }

    #[test]
    fn snapshot_query_measurement_clickbench_no_memory() -> anyhow::Result<()> {
        let qm = QueryMeasurement {
            query_idx: 1,
            target: Target::new(Engine::DuckDB, Format::Parquet),
            benchmark_dataset: BenchmarkDataset::ClickBench {
                flavor: Flavor::Partitioned,
            },
            benchmark_runner: "ci-runner".to_string(),
            storage: "s3".to_string(),
            runs: vec![Duration::from_nanos(2_000_000)],
        };
        let record = query_measurement_record(&qm, None);
        let rendered = render(&record)?;
        with_settings!({snapshot_suffix => "clickbench"}, {
            assert_snapshot!(rendered);
        });
        Ok(())
    }

    #[test]
    fn snapshot_compression_time_encode() -> anyhow::Result<()> {
        let timing = CompressionTimingMeasurement {
            name: "compress time/taxi".to_string(),
            format: Format::OnDiskVortex,
            time: Duration::from_nanos(5_000_000),
        };
        let record = compression_time_record(
            &timing,
            "taxi",
            None,
            CompressOp::Compress,
            vec![5_500_000, 5_000_000, 5_200_000],
        );
        assert_snapshot!(render(&record)?);
        Ok(())
    }

    #[test]
    fn snapshot_compression_size() -> anyhow::Result<()> {
        let record = compression_size_record("taxi", None, Format::Lance, 12_345_678);
        assert_snapshot!(render(&record)?);
        Ok(())
    }

    #[test]
    fn snapshot_compression_time_public_bi() -> anyhow::Result<()> {
        // PBI compression records flatten the `public-bi` parent into the
        // dataset axis: a `PBIBenchmark` named `CMSprovider` emits
        // `dataset = "cmsprovider", dataset_variant = NULL`. Mixed-case input
        // here exercises both `PBIBenchmark::v3_dataset_dims` (no variant) and
        // `compression_time_record` (lowercase-folded dataset).
        let timing = CompressionTimingMeasurement {
            name: "compress time/CMSprovider".to_string(),
            format: Format::OnDiskVortex,
            time: Duration::from_nanos(5_000_000),
        };
        let record = compression_time_record(
            &timing,
            "CMSprovider",
            None,
            CompressOp::Compress,
            vec![5_500_000, 5_000_000, 5_200_000],
        );
        assert_snapshot!(render(&record)?);
        Ok(())
    }

    #[test]
    fn snapshot_random_access_time() -> anyhow::Result<()> {
        let timing = TimingMeasurement {
            name: "random-access/taxi/uniform/parquet-tokio-local-disk".to_string(),
            target: Target::new(Engine::Arrow, Format::Parquet),
            storage: "nvme".to_string(),
            runs: vec![
                Duration::from_nanos(800_000),
                Duration::from_nanos(900_000),
                Duration::from_nanos(850_000),
            ],
        };
        let record = random_access_record(&timing, "taxi");
        assert_snapshot!(render(&record)?);
        Ok(())
    }

    #[test]
    fn snapshot_vector_search_run() -> anyhow::Result<()> {
        let dims = VectorSearchDims {
            dataset: "cohere-large-10m",
            layout: "partitioned",
            flavor: "vortex-turboquant",
            threshold: 0.85,
        };
        let record = vector_search_record(
            dims,
            42_000_000,
            vec![45_000_000, 42_000_000, 41_000_000],
            123,
            10_000_000,
            512_000_000,
        );
        assert_snapshot!(render(&record)?);
        Ok(())
    }

    #[test]
    fn live_dims_match_migrate_for_non_fan_out_suites() {
        // The v2 → v3 migrate classifier leaves both `dataset_variant` and
        // `scale_factor` NULL for the non-fan-out SQL suites (clickbench,
        // polarsignals, statpopgen, fineweb, gharchive). The live emitter
        // must do the same so live ingests merge with migrated history into
        // a single group instead of forking off a sibling group keyed on a
        // dim the historical rows do not carry.
        for (case, expected) in [
            (
                BenchmarkDataset::ClickBench {
                    flavor: Flavor::Partitioned,
                },
                "clickbench",
            ),
            (
                BenchmarkDataset::ClickBench {
                    flavor: Flavor::Single,
                },
                "clickbench",
            ),
            (
                BenchmarkDataset::PolarSignals { n_rows: 1_000_000 },
                "polarsignals",
            ),
            (
                BenchmarkDataset::StatPopGen { n_rows: 100_000 },
                "statpopgen",
            ),
            (BenchmarkDataset::Fineweb, "fineweb"),
            (BenchmarkDataset::GhArchive, "gharchive"),
        ] {
            let (ds, variant, sf) = benchmark_dataset_dims(&case);
            assert_eq!(ds, expected, "dataset for {case:?}");
            assert_eq!(variant, None, "dataset_variant for {case:?}");
            assert_eq!(sf, None, "scale_factor for {case:?}");
        }
    }

    #[test]
    fn compression_records_lowercase_dataset_for_v2_history_match() {
        // The v2 → v3 migrate classifier stores `dataset = series.to_lowercase()`
        // for compress-bench records (see `benchmarks-website/migrate/src/classifier.rs`).
        // Datasets whose `Dataset::name()` returns mixed case
        // (`TPC-H l_comment chunked`, every PBI name like `Arade`/`CMSprovider`)
        // would otherwise emit live records that do not merge with their
        // migrated history. Lowercasing inside the v3 helpers keeps the trait
        // API simple for non-v3 callers while still matching migrate's shape.
        let timing = CompressionTimingMeasurement {
            name: "compress time/TPC-H l_comment chunked".to_string(),
            format: Format::OnDiskVortex,
            time: Duration::from_nanos(1_000_000),
        };
        let record = compression_time_record(
            &timing,
            "TPC-H l_comment chunked",
            None,
            CompressOp::Compress,
            vec![1_000_000],
        );
        let V3Record::CompressionTime(time) = &record else {
            panic!("expected CompressionTime variant, got {record:?}");
        };
        assert_eq!(time.dataset, "tpc-h l_comment chunked");

        let record = compression_size_record("CMSprovider", None, Format::OnDiskVortex, 42);
        let V3Record::CompressionSize(size) = &record else {
            panic!("expected CompressionSize variant, got {record:?}");
        };
        assert_eq!(size.dataset, "cmsprovider");
    }

    #[test]
    fn tpc_scale_factors_are_canonicalized_for_query_dims() {
        // Bench-orchestrator passes raw TPC scale factors like `"1.0"` and `"100.0"`,
        // but the v2 → v3 migrate path canonicalizes integer-valued scale factors
        // to `"1"` and `"100"` (because v2 chart names carried integer-looking
        // values). The live emitter must do the same so live ingests merge with
        // migrated history into a single chart group instead of forking off a
        // sibling group keyed on `SF=1.0` vs `SF=1`.
        let cases = [
            ("1.0", "1"),
            ("100.0", "100"),
            ("1", "1"),
            ("100", "100"),
            ("0.01", "0.01"),
        ];
        for (input, expected) in cases {
            let (_, _, sf) = benchmark_dataset_dims(&BenchmarkDataset::TpcH {
                scale_factor: input.to_string(),
            });
            assert_eq!(
                sf.as_deref(),
                Some(expected),
                "TpcH scale factor {input:?} should canonicalize to {expected:?}",
            );
            let (_, _, sf) = benchmark_dataset_dims(&BenchmarkDataset::TpcDS {
                scale_factor: input.to_string(),
            });
            assert_eq!(
                sf.as_deref(),
                Some(expected),
                "TpcDS scale factor {input:?} should canonicalize to {expected:?}",
            );
        }
    }

    #[test]
    fn jsonl_round_trips_one_record_per_line() -> anyhow::Result<()> {
        let record = compression_size_record("taxi", None, Format::Parquet, 100);
        let mut buf: Vec<u8> = Vec::new();
        write_jsonl(&mut buf, &[record.clone(), record])?;
        let s = String::from_utf8(buf)?;
        assert_eq!(s.lines().count(), 2);
        for line in s.lines() {
            let v: serde_json::Value = serde_json::from_str(line)?;
            assert_eq!(v["kind"], "compression_size");
        }
        Ok(())
    }
}
