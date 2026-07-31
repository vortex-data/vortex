// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "unstable_encodings")]

//! String-compression benchmarks for Vortex.
//!
//! Two intentionally separate suites:
//!
//! * `serialized` is the default and the only one CI tracks. It writes a column
//!   to an in-memory Vortex file with one encoder forced, then reads it back,
//!   reporting `size`, `write`, and `read`.
//! * `codec` is a local diagnostic (`--suite codec`) that times one encoder's
//!   array-level train + compress call and measures the encoded array's buffer
//!   bytes, with no layout, child compression, or I/O in the way. Use it to tell
//!   an encoder-level change apart from a change in the file stack.
//!
//! Both suites force the requested encoder rather than letting the btrblocks
//! selector pick, so neither can silently measure a different encoding. Their
//! size metrics are still not interchangeable: the codec path trains one state
//! over the whole column and leaves children uncompressed, while the file path
//! trains one per compression chunk and compresses children. See the README
//! ("Codec diagnostic").

use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Result;
use anyhow::bail;
use clap::ValueEnum;
use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use tokio::fs::File;
use vortex::VortexSessionDefault;
use vortex::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_arrow::FromArrowArray;
use vortex_bench::Format;
use vortex_bench::IdempotentPath;
use vortex_bench::datasets::Dataset;
use vortex_bench::datasets::data_downloads::download_data;
use vortex_bench::datasets::tpch_l_comment::TPCHLCommentCanonical;
use vortex_fsst::FSST;
use vortex_onpair::Config;
use vortex_onpair::OnPair;

mod codec;
pub use codec::ColumnResult;
pub use codec::DirectCandidate;
pub use codec::bench_column;

const CLICKBENCH_SHARD_COUNT: u32 = 100;
const CLICKBENCH_URL_PREFIX: &str =
    "https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/";

/// Serialized write and read benchmark: write → open → scan, with each row
/// split decoded to canonical form inside its own scan task.
mod serialized;
pub use serialized::*;

/// Session with the available string encodings and their canonicalize kernels
/// registered for benchmarking.
pub static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_tokio());

/// A named string column, canonicalized to a `VarBinViewArray`-backed array,
/// ready for either benchmark path.
pub struct StringColumn {
    /// Human-readable column identifier used in measurement names.
    pub name: String,
    /// Canonical (`VarBinViewArray`) Utf8 data.
    pub array: ArrayRef,
}

/// String encoder selected by the benchmark.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum StringEncoder {
    /// The OnPair encoder.
    #[value(name = "onpair")]
    OnPair,
    /// The FSST encoder.
    #[value(name = "fsst")]
    Fsst,
}

impl StringEncoder {
    /// Stable short label used in benchmark output.
    pub fn label(self) -> &'static str {
        match self {
            Self::OnPair => "onpair",
            Self::Fsst => "fsst",
        }
    }

    /// Whether `array` is this encoder's Vortex encoding. Used to confirm a
    /// column was compressed (or serialized) with the requested scheme.
    pub fn matches(self, array: &ArrayRef) -> bool {
        match self {
            Self::OnPair => array.as_opt::<OnPair>().is_some(),
            Self::Fsst => array.as_opt::<FSST>().is_some(),
        }
    }
}

impl std::fmt::Display for StringEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Label for an OnPair configuration, e.g. `onpair-12`. The dictionary budget is
/// read from the config that was actually used, so a label can never claim a
/// budget the measured data was not encoded with.
pub(crate) fn onpair_label(config: &Config) -> String {
    format!("onpair-{}", config.max_dict_bits.value())
}

/// Bytes per second for `bytes` processed in `elapsed`.
pub(crate) fn throughput(bytes: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        bytes as f64 / secs
    }
}

/// Duration in milliseconds for lower-is-better machine-readable output.
pub(crate) fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn validate_iterations(iterations: usize) -> Result<()> {
    if iterations == 0 {
        bail!("iterations must be greater than zero");
    }
    Ok(())
}

/// Median of `runs` (the two middle values are averaged for an even count).
/// Empty → zero.
pub(crate) fn median(runs: &[Duration]) -> Duration {
    if runs.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = runs.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }
}

/// Return the non-empty, all-valid Utf8 input in compact canonical form, plus
/// the canonical baseline every metric normalizes against: one 16-byte view per
/// row plus the bytes of the strings too long to inline. Preparation and size
/// accounting are outside all timed regions.
pub(crate) fn prepare_column(
    column: &StringColumn,
    ctx: &mut ExecutionCtx,
) -> Result<(VarBinViewArray, u64)> {
    let canonical = column.array.clone().execute::<VarBinViewArray>(ctx)?;
    if !canonical.dtype().is_utf8() {
        bail!(
            "column {} has dtype {}; string-bench requires Utf8 input",
            column.name,
            canonical.dtype(),
        );
    }
    let valid_count = canonical
        .validity()?
        .execute_mask(canonical.len(), ctx)?
        .true_count();
    if valid_count != canonical.len() {
        bail!(
            "column {} contains {} null strings; string-bench requires all rows to be valid",
            column.name,
            canonical.len() - valid_count,
        );
    }

    // Use an aggressively compacted array so the baseline cannot include
    // retained backing-buffer regions that the compressors never see.
    let canonical = canonical.compact_with_threshold(1.0, ctx)?;
    let uncompressed_bytes = u64::try_from(uncompressed_size_in_bytes(canonical.as_ref(), ctx)?)?;
    if uncompressed_bytes == 0 {
        bail!("column {} has zero uncompressed bytes", column.name);
    }

    Ok((canonical, uncompressed_bytes))
}

/// Assert the canonicalized column matches the original row-for-row (dtype,
/// length, validity, and bytes). This runs once before timing.
pub(crate) fn verify_canonicalized(
    label: &str,
    expected: &VarBinViewArray,
    canonicalized: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> Result<()> {
    if canonicalized.dtype() != expected.dtype() {
        bail!(
            "{label}: canonicalized dtype {} != original {}",
            canonicalized.dtype(),
            expected.dtype(),
        );
    }

    let len = expected.len();
    if canonicalized.len() != len {
        bail!(
            "{label}: canonicalized row count {} != original {len}",
            canonicalized.len(),
        );
    }
    // Materialize both validity masks up front so the per-row `value(i)` and
    // `bytes_at(i)` below are O(1); this one-shot check stays off the timed path.
    let expected_valid = expected.validity()?.execute_mask(len, ctx)?;
    let canonicalized_valid = canonicalized.validity()?.execute_mask(len, ctx)?;
    for i in 0..len {
        let valid = expected_valid.value(i);
        if valid != canonicalized_valid.value(i) {
            bail!("{label}: validity mismatch at row {i}");
        }
        if valid && expected.bytes_at(i).as_slice() != canonicalized.bytes_at(i).as_slice() {
            bail!("{label}: canonicalized value differs from input at row {i}");
        }
    }
    tracing::debug!("{label}: canonicalized output verified ({len} rows)");
    Ok(())
}

/// Canonicalize `array` (a struct, possibly chunked) and pull out `field` as a
/// canonical `VarBinViewArray`-backed Utf8 column.
fn to_utf8_column(array: ArrayRef, field: &str, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    let structs = array.execute::<StructArray>(ctx)?;
    let column = structs.unmasked_field_by_name(field)?.clone();
    Ok(column.execute::<VarBinViewArray>(ctx)?.into_array())
}

/// Load the ClickBench `hits` `URL` column from shard `shard`, downloading the
/// shard first if it is not already present locally.
pub async fn load_clickbench_url(shard: u32, ctx: &mut ExecutionCtx) -> Result<StringColumn> {
    let path = download_clickbench_shard(shard).await?;
    let struct_array = read_parquet_projected(path, "URL").await?;
    Ok(StringColumn {
        name: format!("clickbench/URL/shard-{shard}"),
        array: to_utf8_column(struct_array, "URL", ctx)?,
    })
}

async fn download_clickbench_shard(shard: u32) -> Result<PathBuf> {
    if shard >= CLICKBENCH_SHARD_COUNT {
        bail!(
            "invalid ClickBench shard {shard} (want 0..{})",
            CLICKBENCH_SHARD_COUNT - 1
        );
    }
    let filename = format!("hits_{shard}.parquet");
    let path = "clickbench_partitioned"
        .to_data_path()
        .join(Format::Parquet.name())
        .join(&filename);
    download_data(path, format!("{CLICKBENCH_URL_PREFIX}{filename}")).await
}

/// Load the TPC-H `l_comment` column (from the first `lineitem` parquet shard),
/// generating the TPC-H data if needed.
pub async fn load_tpch_l_comment(ctx: &mut ExecutionCtx) -> Result<StringColumn> {
    let path = TPCHLCommentCanonical.to_parquet_path().await?;
    let struct_array = read_parquet_projected(path, "l_comment").await?;
    Ok(StringColumn {
        name: "tpch/l_comment".to_string(),
        array: to_utf8_column(struct_array, "l_comment", ctx)?,
    })
}

/// Read a single column from a parquet file (projected at the parquet level to
/// avoid decoding the other columns) into a chunked struct array.
async fn read_parquet_projected(path: PathBuf, column: &str) -> Result<ArrayRef> {
    let file = File::open(&path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let col_idx = builder.schema().index_of(column)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [col_idx]);
    let reader = builder.with_projection(mask).build()?;

    let chunks: Vec<ArrayRef> = reader
        .map(|batch| {
            batch
                .map_err(anyhow::Error::from)
                .and_then(|rb| ArrayRef::from_arrow(rb, false).map_err(anyhow::Error::from))
        })
        .try_collect()
        .await?;

    Ok(ChunkedArray::from_iter(chunks).into_array())
}

/// 128 rows cycling over 8 distinct outlined values, plus the canonical
/// uncompressed size they must account for (one 16-byte view per row).
#[cfg(test)]
fn repeated_fixture() -> (StringColumn, u64) {
    let values = (0..128)
        .map(|i| {
            format!(
                "https://example.com/path/to/a/repeated/string-value-{}",
                i % 8
            )
        })
        .collect::<Vec<_>>();
    let outlined_bytes = values.iter().map(|value| value.len() as u64).sum::<u64>();
    (
        StringColumn {
            name: "fixture".to_string(),
            array: VarBinViewArray::from_iter_str(values.iter()).into_array(),
        },
        128 * 16 + outlined_bytes,
    )
}

/// Compact `(name, unit, value)` rendering of emitted measurements, for the
/// metric-schema tests that guard the tracked CI names.
#[cfg(test)]
pub(crate) fn measurement_rows(
    measurements: &[vortex_bench::measurements::CustomUnitMeasurement],
) -> Vec<String> {
    measurements
        .iter()
        .map(|m| format!("{} {} {}", m.name, m.unit, m.value))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vortex::array::VortexSessionExecute;

    use super::*;

    /// The size baseline every metric normalizes against: one 16-byte view per
    /// row plus the bytes of the strings too long to inline, counting neither
    /// buffer regions a slice merely retains nor anything but the live rows.
    #[test]
    fn canonical_baseline_counts_views_and_live_outlined_bytes() -> Result<()> {
        let kept = "second outlined string kept by the slice";
        let chunk_tail = "an outlined string in the second chunk";
        let cases = [
            (
                "inline and outlined",
                VarBinViewArray::from_iter_str(["cat", "hello", "abcdefghijklmnop"]).into_array(),
                3,
                3 * 16 + 16,
            ),
            (
                "sliced",
                VarBinViewArray::from_iter_str([
                    "first outlined string dropped by the slice",
                    kept,
                ])
                .into_array()
                .slice(1..2)?,
                1,
                16 + kept.len() as u64,
            ),
            (
                "chunked",
                ChunkedArray::from_iter([
                    VarBinViewArray::from_iter_str(["alpha", ""]).into_array(),
                    VarBinViewArray::from_iter_str([chunk_tail]).into_array(),
                ])
                .into_array(),
                3,
                3 * 16 + chunk_tail.len() as u64,
            ),
        ];
        let mut ctx = SESSION.create_execution_ctx();

        for (name, array, rows, expected_bytes) in cases {
            let column = StringColumn {
                name: name.to_string(),
                array,
            };
            let (canonical, uncompressed_bytes) = prepare_column(&column, &mut ctx)?;

            assert_eq!(canonical.len(), rows, "{name} row count");
            assert_eq!(uncompressed_bytes, expected_bytes, "{name} size");
        }
        Ok(())
    }

    #[test]
    fn verify_canonicalized_rejects_mismatches() {
        let cases = [
            (
                "dtype",
                VarBinViewArray::from_iter_str(["alpha"]),
                VarBinViewArray::from_iter_bin(["alpha".as_bytes()]),
            ),
            (
                "length",
                VarBinViewArray::from_iter_str(["alpha", "beta"]),
                VarBinViewArray::from_iter_str(["alpha"]),
            ),
            (
                "validity",
                VarBinViewArray::from_iter_nullable_str([Some("alpha"), Some("beta")]),
                VarBinViewArray::from_iter_nullable_str([Some("alpha"), None]),
            ),
            (
                "value",
                VarBinViewArray::from_iter_str(["alpha", "beta"]),
                VarBinViewArray::from_iter_str(["alpha", "BETA"]),
            ),
        ];
        let mut ctx = SESSION.create_execution_ctx();

        for (kind, expected, actual) in cases {
            assert!(
                verify_canonicalized("fixture", &expected, &actual, &mut ctx).is_err(),
                "{kind} mismatch must be rejected"
            );
        }
    }

    #[test]
    fn median_is_stable_for_odd_and_even_runs() {
        let runs = [
            Duration::from_nanos(30),
            Duration::from_nanos(10),
            Duration::from_nanos(20),
        ];
        assert_eq!(median(&runs), Duration::from_nanos(20));

        let runs = [Duration::from_nanos(10), Duration::from_nanos(30)];
        assert_eq!(median(&runs), Duration::from_nanos(20));
        assert_eq!(median(&[]), Duration::ZERO);
    }

    #[test]
    fn iterations_must_be_positive() -> Result<()> {
        assert!(validate_iterations(0).is_err());
        validate_iterations(1)
    }

    #[tokio::test]
    async fn clickbench_shard_rejects_out_of_range() {
        assert!(
            download_clickbench_shard(CLICKBENCH_SHARD_COUNT)
                .await
                .is_err()
        );
    }

    #[test]
    fn prepare_column_rejects_invalid_input() {
        let cases = [
            (
                "null",
                VarBinViewArray::from_iter_nullable_str([Some("alpha"), None]).into_array(),
            ),
            (
                "empty",
                VarBinViewArray::from_iter_str(std::iter::empty::<&str>()).into_array(),
            ),
            (
                "binary",
                VarBinViewArray::from_iter_bin([b"alpha".as_slice()]).into_array(),
            ),
        ];
        let mut ctx = SESSION.create_execution_ctx();

        for (name, array) in cases {
            let column = StringColumn {
                name: name.to_string(),
                array,
            };
            assert!(
                prepare_column(&column, &mut ctx).is_err(),
                "{name} input must be rejected"
            );
        }
    }
}
