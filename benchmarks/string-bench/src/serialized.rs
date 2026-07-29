// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialized Vortex benchmarks: write a string column to an in-memory Vortex
//! file using a selected encoder, with its children compressed by Vortex, then
//! read it back and canonicalize it. These measure Vortex file CPU costs, not
//! physical disk I/O.
//!
//! Each timed iteration writes a fresh file and then re-runs the read path. The
//! write and read paths are reported as separate workloads:
//!
//! * **scan** pays scan construction and scheduling, segment resolution, and
//!   array deserialization, and yields arrays still in their on-disk encoding;
//! * **canonicalize** turns those arrays into their canonical `VarBinViewArray`
//!   form, decompressing the encoder's Vortex-compressed children.
//!
//! The benchmark stages these phases deliberately: it drains all encoded arrays
//! before canonicalizing them serially. This isolates encoding-specific
//! canonicalization costs for comparison and profiling, but is not the fused,
//! potentially parallel production scan path.
//!
//! The benchmark uses Vortex's standard current-thread runtime for both writing
//! and reading.

use std::hint::black_box;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use bytes::Bytes;
use futures::TryStreamExt;
use futures::pin_mut;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::LayoutStrategy;
use vortex::session::VortexSession;
use vortex_bench::Format;
use vortex_bench::measurements::CustomUnitMeasurement;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::SchemeId;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::NullDominatedSparseScheme;
use vortex_btrblocks::schemes::string::OnPairScheme;
use vortex_btrblocks::schemes::string::StringDictScheme;
use vortex_onpair::DEFAULT_DICT12_CONFIG;

use crate::SESSION;
use crate::StringColumn;
use crate::StringEncoder;
use crate::duration_ms;
use crate::median;
use crate::prepare_column;
use crate::throughput;
use crate::verify_canonicalized;

/// The btrblocks string schemes that `BtrBlocksCompressorBuilder::default()` can
/// choose between. Forcing one encoder excludes every entry except its own
/// scheme, so this list must track the default scheme set: add a row whenever a
/// new string encoder becomes selectable by default (e.g. Zstd).
fn default_string_scheme_ids() -> Vec<SchemeId> {
    vec![
        StringDictScheme.id(),
        FSSTScheme.id(),
        OnPairScheme.id(),
        NullDominatedSparseScheme.id(),
    ]
}

impl StringEncoder {
    /// The btrblocks string scheme that produces this encoder's on-disk arrays.
    fn scheme_id(self) -> SchemeId {
        match self {
            Self::OnPair => OnPairScheme.id(),
            Self::Fsst => FSSTScheme.id(),
        }
    }
}

/// An in-memory Vortex file produced for one column and encoder.
struct SerializedFile {
    file_bytes: u64,
    chunk_count: usize,
    strategy: Arc<dyn LayoutStrategy>,
}

/// Timings for one staged read of an already serialized Vortex buffer.
struct ReadTimings {
    open: Duration,
    scan: Duration,
    canonicalize: Duration,
    staged_read: Duration,
}

/// Phase-broken-down timings for one serialized write and staged read of a
/// string column and encoder.
pub struct SerializedResult {
    /// Column identifier.
    pub name: String,
    /// Encoder forced for this file, e.g. `onpair-12` or `fsst`.
    pub encoder: String,
    /// Number of rows.
    pub rows: usize,
    /// Canonical uncompressed array bytes used to normalize size and throughput.
    pub uncompressed_bytes: u64,
    /// Serialized Vortex file size (bytes).
    pub file_bytes: u64,
    /// Per-iteration write times, including encoding, layout, compression, and
    /// serialization into an in-memory buffer.
    pub write_runs: Vec<Duration>,
    /// Per-iteration serialized-buffer open times.
    pub open_runs: Vec<Duration>,
    /// Per-iteration scan times, including scan construction and scheduling,
    /// segment resolution, and encoded-array materialization.
    pub scan_runs: Vec<Duration>,
    /// Per-iteration fresh-array canonicalization times, including child decompression.
    pub canonicalize_runs: Vec<Duration>,
    /// Per-iteration staged read times: open, scan all encoded arrays, then
    /// canonicalize them serially.
    pub staged_read_runs: Vec<Duration>,
}

impl SerializedResult {
    fn ms(runs: &[Duration]) -> f64 {
        duration_ms(median(runs))
    }

    /// Median serialized-write throughput in MB/s of canonical uncompressed
    /// array bytes.
    pub fn write_mbps(&self) -> f64 {
        throughput(self.uncompressed_bytes, median(&self.write_runs)) / 1e6
    }

    /// Median fresh-array canonicalization throughput in MB/s of canonical
    /// uncompressed array bytes.
    pub fn canonicalize_mbps(&self) -> f64 {
        throughput(self.uncompressed_bytes, median(&self.canonicalize_runs)) / 1e6
    }

    /// Median staged read throughput in MB/s of canonical uncompressed bytes.
    pub fn staged_read_mbps(&self) -> f64 {
        throughput(self.uncompressed_bytes, median(&self.staged_read_runs)) / 1e6
    }

    /// Median serialized-buffer open time in milliseconds.
    pub fn open_ms(&self) -> f64 {
        Self::ms(&self.open_runs)
    }

    /// Median serialized-buffer scan time in milliseconds.
    pub fn scan_ms(&self) -> f64 {
        Self::ms(&self.scan_runs)
    }

    /// Complete serialized file bytes as a percentage of canonical
    /// uncompressed array bytes. Lower is better.
    pub fn file_size_pct(&self) -> f64 {
        self.file_bytes as f64 / self.uncompressed_bytes as f64 * 100.0
    }

    /// Emit all serialized metrics as Vortex-format custom-unit metrics.
    pub fn measurements(&self) -> Vec<CustomUnitMeasurement> {
        let suffix = format!("{} {}", self.name, self.encoder);
        let ms = |name: String, runs: &[Duration]| CustomUnitMeasurement {
            name,
            format: Format::OnDiskVortex,
            unit: "ms".into(),
            value: Self::ms(runs),
        };
        vec![
            ms(format!("serialized write/{suffix}"), &self.write_runs),
            ms(format!("serialized read open/{suffix}"), &self.open_runs),
            ms(format!("serialized read scan/{suffix}"), &self.scan_runs),
            ms(
                format!("serialized read canonicalize/{suffix}"),
                &self.canonicalize_runs,
            ),
            ms(
                format!("serialized staged read/{suffix}"),
                &self.staged_read_runs,
            ),
            CustomUnitMeasurement {
                name: format!("serialized file size (% of canonical)/{suffix}"),
                format: Format::OnDiskVortex,
                unit: "%".into(),
                value: self.file_size_pct(),
            },
        ]
    }
}

/// Build the file writer strategy that forces one selected string scheme while
/// leaving non-string child compression enabled.
fn serialized_write_strategy(encoder: StringEncoder) -> Arc<dyn LayoutStrategy> {
    let forced = encoder.scheme_id();
    let compressor = BtrBlocksCompressorBuilder::default().exclude_schemes(
        default_string_scheme_ids()
            .into_iter()
            .filter(|&id| id != forced),
    );
    WriteStrategyBuilder::default()
        .with_btrblocks_builder(compressor)
        .build()
}

/// Write one canonical string column to an in-memory Vortex file, forcing the
/// requested string encoder.
async fn write_serialized_file(
    session: &VortexSession,
    input: &ArrayRef,
    strategy: &Arc<dyn LayoutStrategy>,
) -> Result<Bytes> {
    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        session
            .write_options()
            .with_strategy(Arc::clone(strategy))
            .write(&mut cursor, input.to_array_stream())
            .await?;
    }
    Ok(Bytes::from(buf))
}

/// Validate that a serialized file contains the selected root encoding and,
/// optionally, canonicalizes back to the original input.
async fn inspect_serialized_file(
    session: &VortexSession,
    data: Bytes,
    column: &StringColumn,
    encoder: StringEncoder,
    canonical: &VarBinViewArray,
    verify: bool,
    ctx: &mut vortex::array::ExecutionCtx,
) -> Result<usize> {
    let file = session.open_options().open_buffer(data)?;
    let arrays: Vec<ArrayRef> = file.scan()?.into_array_stream()?.try_collect().await?;
    if arrays.is_empty() {
        return Err(anyhow!("empty scan for column {}", column.name));
    }
    let chunk_count = arrays.len();
    if let Some(unexpected) = arrays.iter().find(|array| !encoder.matches(array)) {
        bail!(
            "serialized column {} contains {} instead of {} — the file was not \
             uniformly {}-encoded (is {} the smallest scheme?)",
            column.name,
            unexpected.encoding_id(),
            encoder,
            encoder,
            encoder,
        );
    }
    if verify {
        let canonicalized = ChunkedArray::from_iter(arrays)
            .into_array()
            .execute::<VarBinViewArray>(ctx)?;
        verify_canonicalized(
            &format!("{} [serialized read {encoder}]", column.name),
            canonical,
            &canonicalized,
            ctx,
        )?;
    }
    Ok(chunk_count)
}

/// Read one serialized Vortex buffer, keeping scan and canonicalization as
/// separately measured phases. The scan is drained before canonicalization so
/// this remains comparable with the existing serialized-read benchmark.
async fn read_serialized_buffer(
    session: &VortexSession,
    data: Bytes,
    chunk_count: usize,
) -> Result<ReadTimings> {
    let mut ctx = session.create_execution_ctx();
    let mut arrays = Vec::with_capacity(chunk_count);
    let mut canonical = Vec::with_capacity(chunk_count);

    let t0 = Instant::now();
    let file = session.open_options().open_buffer(data)?;
    let t1 = Instant::now();
    let stream = file.scan()?.into_array_stream()?;
    pin_mut!(stream);
    while let Some(array) = stream.try_next().await? {
        arrays.push(array);
    }
    let t2 = Instant::now();
    for array in arrays {
        canonical.push(array.execute::<VarBinViewArray>(&mut ctx)?);
    }
    black_box(&canonical);
    let t3 = Instant::now();

    Ok(ReadTimings {
        open: t1 - t0,
        scan: t2 - t1,
        canonicalize: t3 - t2,
        staged_read: t3 - t0,
    })
}

/// Write and validate one serialized file before timing the write and read paths.
async fn prepare_serialized_file(
    session: &VortexSession,
    column: &StringColumn,
    input: &ArrayRef,
    canonical: &VarBinViewArray,
    encoder: StringEncoder,
    verify: bool,
    ctx: &mut vortex::array::ExecutionCtx,
) -> Result<SerializedFile> {
    let strategy = serialized_write_strategy(encoder);
    let data = write_serialized_file(session, input, &strategy).await?;
    let file_bytes = data.len() as u64;
    let chunk_count = inspect_serialized_file(
        session,
        data.clone(),
        column,
        encoder,
        canonical,
        verify,
        ctx,
    )
    .await?;
    Ok(SerializedFile {
        file_bytes,
        chunk_count,
        strategy,
    })
}

/// Label the actual configuration used by the serialized file.
fn serialized_encoder_label(encoder: StringEncoder) -> String {
    match encoder {
        StringEncoder::OnPair => {
            format!("onpair-{}", DEFAULT_DICT12_CONFIG.max_dict_bits.value())
        }
        StringEncoder::Fsst => "fsst".to_string(),
    }
}

/// Time fresh in-memory serialized writes and staged reads for one string
/// column and encoder. The read phases are staged deliberately so the
/// benchmark reports both canonicalization and complete staged-read costs; it
/// is not production query throughput.
pub async fn bench_serialized(
    column: &StringColumn,
    iterations: usize,
    warmup: usize,
    verify: bool,
    encoder: StringEncoder,
) -> Result<SerializedResult> {
    bench_serialized_with_session(&SESSION, column, iterations, warmup, verify, encoder).await
}

/// Time the serialized Vortex path using an explicitly configured Vortex
/// session.
pub async fn bench_serialized_with_session(
    session: &VortexSession,
    column: &StringColumn,
    iterations: usize,
    warmup: usize,
    verify: bool,
    encoder: StringEncoder,
) -> Result<SerializedResult> {
    crate::validate_iterations(iterations)?;
    let mut ctx = session.create_execution_ctx();

    let (canonical, uncompressed_bytes) = prepare_column(column, &mut ctx)?;
    let input = canonical.clone().into_array();
    let rows = canonical.len();
    let serialized = prepare_serialized_file(
        session, column, &input, &canonical, encoder, verify, &mut ctx,
    )
    .await?;

    // Warm up the same complete path that will be timed. The reference file
    // above is used only for validation, size, and the known chunk count.
    for _ in 0..warmup.max(1) {
        let data = write_serialized_file(session, &input, &serialized.strategy).await?;
        let _ = read_serialized_buffer(session, data, serialized.chunk_count).await?;
    }

    let mut write_runs = Vec::with_capacity(iterations);
    let mut open_runs = Vec::with_capacity(iterations);
    let mut scan_runs = Vec::with_capacity(iterations);
    let mut canonicalize_runs = Vec::with_capacity(iterations);
    let mut staged_read_runs = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let data = write_serialized_file(session, &input, &serialized.strategy).await?;
        let after_write = Instant::now();
        let timings = read_serialized_buffer(session, data, serialized.chunk_count).await?;

        write_runs.push(after_write - start);
        open_runs.push(timings.open);
        scan_runs.push(timings.scan);
        canonicalize_runs.push(timings.canonicalize);
        staged_read_runs.push(timings.staged_read);
    }

    Ok(SerializedResult {
        name: column.name.clone(),
        encoder: serialized_encoder_label(encoder),
        rows,
        uncompressed_bytes,
        file_bytes: serialized.file_bytes,
        write_runs,
        open_runs,
        scan_runs,
        canonicalize_runs,
        staged_read_runs,
    })
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::Canonical;
    use vortex::array::VortexSessionExecute;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex_btrblocks::ALL_SCHEMES;
    use vortex_btrblocks::SchemeExt;

    use super::*;

    #[test]
    fn forced_scheme_inventory_matches_default_utf8_schemes() {
        // Every default scheme whose dtype gate accepts canonical Utf8 must be
        // excluded when another root string encoding is forced.
        let canonical = Canonical::VarBinView(VarBinViewArray::from_iter_str(["value"]));
        let mut actual = ALL_SCHEMES
            .iter()
            .filter(|scheme| scheme.matches(&canonical))
            .map(|scheme| scheme.id())
            .collect::<Vec<_>>();
        let mut expected = default_string_scheme_ids();
        actual.sort_unstable_by_key(|id| id.to_string());
        expected.sort_unstable_by_key(|id| id.to_string());

        assert_eq!(actual, expected);
    }

    #[test]
    fn machine_readable_metric_schema_is_stable() {
        let result = SerializedResult {
            name: "fixture".to_string(),
            encoder: "fsst".to_string(),
            rows: 1,
            uncompressed_bytes: 2,
            file_bytes: 1,
            write_runs: vec![Duration::from_millis(1)],
            open_runs: vec![Duration::from_millis(1)],
            scan_runs: vec![Duration::from_millis(2)],
            canonicalize_runs: vec![Duration::from_millis(3)],
            staged_read_runs: vec![Duration::from_millis(6)],
        };

        let measurements = result.measurements();
        assert_eq!(
            measurements
                .iter()
                .map(|measurement| {
                    (
                        measurement.name.as_str(),
                        measurement.unit.as_ref(),
                        measurement.value,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                ("serialized write/fixture fsst", "ms", 1.0),
                ("serialized read open/fixture fsst", "ms", 1.0),
                ("serialized read scan/fixture fsst", "ms", 2.0),
                ("serialized read canonicalize/fixture fsst", "ms", 3.0),
                ("serialized staged read/fixture fsst", "ms", 6.0),
                (
                    "serialized file size (% of canonical)/fixture fsst",
                    "%",
                    50.0,
                ),
            ]
        );
    }

    #[test]
    fn serialized_benchmark_smoke_tests_each_encoder() -> Result<()> {
        let (column, expected_uncompressed_bytes) = crate::repeated_fixture();
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());

        for (encoder, expected_label) in [
            (StringEncoder::OnPair, "onpair-12"),
            (StringEncoder::Fsst, "fsst"),
        ] {
            let result = runtime.block_on(bench_serialized_with_session(
                &session, &column, 1, 0, true, encoder,
            ))?;

            assert_eq!(result.encoder, expected_label);
            assert_eq!(result.rows, 128);
            assert_eq!(result.uncompressed_bytes, expected_uncompressed_bytes);
            assert!(result.file_bytes > 0);
            assert_eq!(result.write_runs.len(), 1);
            assert_eq!(result.open_runs.len(), 1);
            assert_eq!(result.scan_runs.len(), 1);
            assert_eq!(result.canonicalize_runs.len(), 1);
            assert_eq!(result.staged_read_runs.len(), 1);
        }
        Ok(())
    }

    #[test]
    fn serialized_verification_handles_multiple_chunks() -> Result<()> {
        let values = (0..50_000_u64)
            .map(|i| {
                let mixed = i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                format!(
                    "https://example.com/users/{i:016x}/events/{mixed:016x}/\
                     common/repeated/string/payload"
                )
            })
            .collect::<Vec<_>>();
        let expected_uncompressed_bytes =
            values.iter().map(|value| value.len() as u64).sum::<u64>() + 16 * values.len() as u64;
        let column = StringColumn {
            name: "multi-chunk".to_string(),
            array: VarBinViewArray::from_iter_str(values.iter()).into_array(),
        };
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        let mut ctx = session.create_execution_ctx();
        let (canonical, uncompressed_bytes) = prepare_column(&column, &mut ctx)?;
        let input = canonical.clone().into_array();

        let serialized = runtime.block_on(prepare_serialized_file(
            &session,
            &column,
            &input,
            &canonical,
            StringEncoder::Fsst,
            true,
            &mut ctx,
        ))?;

        assert!(serialized.chunk_count > 1);
        assert_eq!(uncompressed_bytes, expected_uncompressed_bytes);
        Ok(())
    }
}
