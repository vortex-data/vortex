// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialized Vortex benchmarks. Each timed iteration writes a fresh in-memory
//! file with one string encoder forced and then reads it back:
//!
//! * **write** runs the full default write pipeline — repartition into row
//!   blocks, zoned statistics, dictionary probe, coalesce, compress with the
//!   forced string scheme and its children, layout, serialize into a buffer;
//! * **read** opens that buffer and runs the scan, decoding each row split to
//!   canonical `VarBinViewArray` form inside its own scan task and dropping it
//!   before the next split runs — the shape production uses, where
//!   `into_record_batch_stream` fuses the Arrow conversion into the split task.
//!
//! Both run on a current-thread runtime, so these are single-threaded CPU costs
//! and exclude physical I/O.

use std::hint::black_box;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use bytes::Bytes;
use futures::TryStreamExt;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
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

use crate::StringColumn;
use crate::StringEncoder;
use crate::duration_ms;
use crate::median;
use crate::onpair_label;
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

/// Label for one encoder on the serialized path, carrying the configuration the
/// file was written with, e.g. `onpair-12` or `fsst`. Changing that config
/// renames the metric and so restarts its benchmark history, which is intended:
/// neither size nor read time is comparable across dictionary budgets.
fn serialized_encoder_label(encoder: StringEncoder) -> String {
    match encoder {
        // The config `OnPairScheme` compresses with. When btrblocks gains a
        // configurable budget, pass the benchmark's own config here.
        StringEncoder::OnPair => onpair_label(&DEFAULT_DICT12_CONFIG),
        StringEncoder::Fsst => encoder.label().to_string(),
    }
}

/// An in-memory Vortex file produced for one column and encoder.
struct SerializedFile {
    file_bytes: u64,
    /// Number of row splits the scan will produce, i.e. the number of encoded
    /// arrays the read decodes one at a time.
    chunk_count: usize,
    strategy: Arc<dyn LayoutStrategy>,
}

/// The three core metrics for one serialized write and read of a string column
/// under one encoder: `size`, `write`, and `read`.
pub struct SerializedResult {
    /// Column identifier.
    pub name: String,
    /// Encoder forced for this file, with its configuration, e.g. `onpair-12`
    /// or `fsst`.
    pub encoder: String,
    /// Number of rows.
    pub rows: usize,
    /// Canonical uncompressed array bytes used to normalize size and
    /// throughput: one 16-byte view per row plus the bytes of the strings too
    /// long to inline.
    pub uncompressed_bytes: u64,
    /// Serialized Vortex file size (bytes).
    pub file_bytes: u64,
    /// Per-iteration write times: the whole default write pipeline, from
    /// repartitioning and zoned statistics through compression, layout, and
    /// serialization into an in-memory buffer.
    pub write_runs: Vec<Duration>,
    /// Per-iteration read times: open the file, then run the scan with each row
    /// split decoded to canonical form inside its own task.
    pub read_runs: Vec<Duration>,
}

impl SerializedResult {
    /// Median serialized-write throughput in MB/s of canonical uncompressed
    /// array bytes.
    pub fn write_mbps(&self) -> f64 {
        throughput(self.uncompressed_bytes, median(&self.write_runs)) / 1e6
    }

    /// Median read throughput in MB/s of canonical uncompressed bytes.
    pub fn read_mbps(&self) -> f64 {
        throughput(self.uncompressed_bytes, median(&self.read_runs)) / 1e6
    }

    /// Complete serialized file bytes as a percentage of canonical
    /// uncompressed array bytes. Lower is better.
    pub fn size_pct(&self) -> f64 {
        self.file_bytes as f64 / self.uncompressed_bytes as f64 * 100.0
    }

    /// Emit the three core metrics as Vortex-format custom-unit metrics, named
    /// `<metric>/<input>/<encoder>`.
    pub fn measurements(&self) -> Vec<CustomUnitMeasurement> {
        let suffix = format!("{}/{}", self.name, self.encoder);
        let ms = |name: String, runs: &[Duration]| CustomUnitMeasurement {
            name,
            format: Format::OnDiskVortex,
            unit: "ms".into(),
            value: duration_ms(median(runs)),
        };
        vec![
            CustomUnitMeasurement {
                name: format!("size/{suffix}"),
                format: Format::OnDiskVortex,
                unit: "%".into(),
                value: self.size_pct(),
            },
            ms(format!("write/{suffix}"), &self.write_runs),
            ms(format!("read/{suffix}"), &self.read_runs),
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

/// Time one complete read of a serialized Vortex buffer: open the file, then run
/// the scan with the canonical decode fused into each row split's task, dropping
/// each decoded chunk before the next split runs.
///
/// The splits are awaited one at a time rather than through
/// `ScanBuilder::into_array_stream`, which spawns
/// `concurrency * available_parallelism()` of them at once. On a current-thread
/// runtime that read-ahead buys no parallelism; it only holds that many chunks in
/// memory and makes the result depend on the host's core count. Awaiting one at a
/// time keeps the per-split work identical to production while making the
/// measurement machine-independent.
async fn read_serialized_buffer(session: &VortexSession, data: Bytes) -> Result<Duration> {
    let decode_session = session.clone();

    let start = Instant::now();
    let file = session.open_options().open_buffer(data)?;
    let splits = file
        .scan()?
        .map(move |chunk: ArrayRef| {
            let mut ctx = decode_session.create_execution_ctx();
            chunk.execute::<VarBinViewArray>(&mut ctx)
        })
        .build()?;

    let mut rows = 0usize;
    for split in splits {
        if let Some(canonical) = split.await? {
            rows += canonical.len();
            drop(black_box(canonical));
        }
    }

    black_box(rows);

    Ok(start.elapsed())
}

/// Write one reference file, then check it is uniformly `encoder`-encoded and,
/// when `verify` is set, that it decodes back to the input. Runs once, outside
/// every timed region: its size is the reported `size` metric.
async fn prepare_serialized_file(
    session: &VortexSession,
    column: &StringColumn,
    input: &ArrayRef,
    canonical: &VarBinViewArray,
    encoder: StringEncoder,
    verify: bool,
    ctx: &mut ExecutionCtx,
) -> Result<SerializedFile> {
    let strategy = serialized_write_strategy(encoder);
    let data = write_serialized_file(session, input, &strategy).await?;
    let file_bytes = data.len() as u64;

    let file = session.open_options().open_buffer(data)?;
    let chunks: Vec<ArrayRef> = file.scan()?.into_array_stream()?.try_collect().await?;
    if chunks.is_empty() {
        bail!("empty scan for column {}", column.name);
    }
    if let Some(unexpected) = chunks.iter().find(|array| !encoder.matches(array)) {
        bail!(
            "serialized column {} contains {} instead of {encoder} — the file was not \
             uniformly {encoder}-encoded (is {encoder} the smallest scheme?)",
            column.name,
            unexpected.encoding_id(),
        );
    }
    let chunk_count = chunks.len();
    if verify {
        let canonicalized = ChunkedArray::from_iter(chunks)
            .into_array()
            .execute::<VarBinViewArray>(ctx)?;
        verify_canonicalized(
            &format!("{} [serialized read {encoder}]", column.name),
            canonical,
            &canonicalized,
            ctx,
        )?;
    }

    Ok(SerializedFile {
        file_bytes,
        chunk_count,
        strategy,
    })
}

/// Time fresh in-memory serialized writes and reads for one string column and
/// encoder, on the given session's runtime.
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

    let label = serialized_encoder_label(encoder);
    tracing::debug!(
        "{} [{label}]: {rows} rows, {} file bytes in {} row splits",
        column.name,
        serialized.file_bytes,
        serialized.chunk_count,
    );

    // Warm up the same complete path that will be timed. The reference file
    // above is used only for validation and size.
    for _ in 0..warmup.max(1) {
        let data = write_serialized_file(session, &input, &serialized.strategy).await?;
        let _ = read_serialized_buffer(session, data).await?;
    }

    let mut write_runs = Vec::with_capacity(iterations);
    let mut read_runs = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let data = write_serialized_file(session, &input, &serialized.strategy).await?;
        write_runs.push(start.elapsed());
        read_runs.push(read_serialized_buffer(session, data).await?);
    }

    Ok(SerializedResult {
        name: column.name.clone(),
        encoder: label,
        rows,
        uncompressed_bytes,
        file_bytes: serialized.file_bytes,
        write_runs,
        read_runs,
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

    /// The three core metric names are the tracked CI schema: renaming one
    /// silently restarts its benchmark history.
    #[test]
    fn machine_readable_metric_schema_is_stable() {
        let result = SerializedResult {
            name: "fixture".to_string(),
            encoder: "fsst".to_string(),
            rows: 1,
            uncompressed_bytes: 2,
            file_bytes: 1,
            write_runs: vec![Duration::from_millis(1)],
            read_runs: vec![Duration::from_millis(6)],
        };

        assert_eq!(
            crate::measurement_rows(&result.measurements()),
            [
                "size/fixture/fsst % 50",
                "write/fixture/fsst ms 1",
                "read/fixture/fsst ms 6",
            ]
        );
    }

    /// Also pins the encoder labels: a change to the btrblocks OnPair default
    /// renames the tracked metrics and restarts their history.
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
            assert_eq!(result.read_runs.len(), 1);
        }
        Ok(())
    }

    /// Enough poorly-compressible rows to span several row splits, so the
    /// pre-timing check covers a file the read decodes one split at a time.
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
        let column = StringColumn {
            name: "multi-chunk".to_string(),
            array: VarBinViewArray::from_iter_str(values.iter()).into_array(),
        };
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        let mut ctx = session.create_execution_ctx();
        let (canonical, _) = prepare_column(&column, &mut ctx)?;
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
        Ok(())
    }
}
