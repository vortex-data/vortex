// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPair chunked-array compression benchmark.
//!
//! For one string column this:
//!   1. samples roughly the first `sample_bytes` of raw string payload,
//!   2. splits that sample into chunks sized by an *uncompressed byte budget*
//!      (`chunk_bytes`), cut on equal-ish row boundaries,
//!   3. OnPair-compresses each chunk with its own dictionary (in parallel),
//!   4. assembles the chunks into a `ChunkedArray`,
//!   5. writes ~`file_target_bytes` Vortex files that preserve the OnPair
//!      encoding on disk,
//!   6. reads every file back and verifies the string round-trip,
//!   7. reports sizes, ratios and encode/decode throughput.
//!
//! The matrix swept by [`run_column`] is `bits × chunk_bytes × threshold`.
//! New datasets/columns are added by the caller (the Python orchestrator)
//! simply by pointing [`run_column`] at a different parquet file + column;
//! TPC-H generation is provided by [`ensure_tpch_all_parquet`].

use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use arrow_array::Array as _;
use arrow_array::LargeStringArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use futures::future::try_join_all;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use serde::Serialize;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::accessor::ArrayAccessor;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::validity::Validity;
use vortex::compressor::BtrBlocksCompressor;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::encodings::fastlanes::Delta;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::config_with_bits;
use vortex_onpair::onpair_compress_array_default;

use crate::SESSION;

const GIB: f64 = (1usize << 30) as f64;

/// One row of benchmark output: a single `(column, bits, chunk, threshold)`
/// cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellResult {
    /// Stable id of the source dataset (e.g. `tpch-sf10`).
    pub dataset_id: String,
    /// Column name within the dataset.
    pub column: String,
    /// OnPair dictionary code width.
    pub bits: u32,
    /// OnPair training threshold.
    pub threshold: f64,
    /// Per-chunk uncompressed byte budget.
    pub chunk_bytes: u64,
    /// Rows in the sampled prefix.
    pub rows: u64,
    /// Distinct string values in the sample. Compare against `rows`: a low
    /// `unique_count` means a whole-value dictionary (codes + values) may beat
    /// OnPair's token dictionary.
    pub unique_count: u64,
    /// Raw (uncompressed) string payload bytes in the sample.
    pub sample_bytes: u64,
    /// Number of OnPair chunks (one dictionary each).
    pub n_chunks: usize,
    /// In-memory size of the OnPair `ChunkedArray`.
    pub in_memory_bytes: u64,
    /// Total bytes of the dictionary blobs across all chunks.
    pub dict_bytes: u64,
    /// Total on-disk size of the written `.vortex` files.
    pub on_disk_bytes: u64,
    /// Number of `.vortex` files written.
    pub n_files: usize,
    /// Wall-clock OnPair compression time.
    pub encode_ms: f64,
    /// Wall-clock read-back + canonicalize time.
    pub decode_ms: f64,
    /// `sample_bytes / encode_time`.
    pub encode_gib_s: f64,
    /// `sample_bytes / decode_time`.
    pub decode_gib_s: f64,
    /// `sample_bytes / in_memory_bytes`.
    pub mem_ratio: f64,
    /// `sample_bytes / on_disk_bytes`.
    pub disk_ratio: f64,
    /// Whether the decoded strings matched the input exactly.
    pub verified: bool,
    /// Whether every chunk on disk is encoded purely as OnPair (no
    /// recompression to another scheme).
    pub onpair_only: bool,
    /// Directory holding this cell's `.vortex` files + `meta.json`.
    pub out_dir: String,
}

/// Encoding id of the OnPair array, used to assert on-disk encoding.
const ONPAIR_ENCODING: &str = "vortex.onpair";

/// A sampled string column held canonically in memory, reused across all
/// matrix cells for a column.
struct Sample {
    array: ArrayRef,
    rows: usize,
    raw_bytes: u64,
    unique_count: u64,
}

/// Read the first `sample_bytes` of raw string payload from `column` in
/// `parquet_path`, returning a canonical `Utf8` array.
fn build_sample(parquet_path: &Path, column: &str, sample_bytes: u64) -> Result<Sample> {
    let file = std::fs::File::open(parquet_path)
        .with_context(|| format!("opening parquet {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let arrow_schema = Arc::clone(builder.schema());
    let col_idx = arrow_schema
        .index_of(column)
        .with_context(|| format!("column '{column}' not found in {}", parquet_path.display()))?;
    let nullable = arrow_schema.field(col_idx).is_nullable();
    let mask = ProjectionMask::roots(builder.parquet_schema(), [col_idx]);

    let reader = builder
        .with_projection(mask)
        .with_batch_size(64 * 1024)
        .build()?;

    // Stop reading once we have enough payload. Keep only the batches we need;
    // the last batch is row-capped so the sample lands near `sample_bytes`.
    let mut kept: Vec<RecordBatch> = Vec::new();
    let mut bytes: u64 = 0;
    let mut rows: usize = 0;
    'outer: for batch in reader {
        let batch = batch?;
        let col = batch.column(0);
        let lens = string_byte_lengths(col)
            .with_context(|| format!("column '{column}' is not a string column"))?;
        let mut take = lens.len();
        for (i, l) in lens.iter().enumerate() {
            if bytes + *l > sample_bytes {
                take = i;
                break;
            }
            bytes += *l;
        }
        if take < lens.len() {
            if take > 0 {
                kept.push(batch.slice(0, take));
                rows += take;
            }
            break 'outer;
        }
        rows += lens.len();
        kept.push(batch);
    }

    let dtype = DType::Utf8(if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    });
    let view = VarBinViewArray::from_iter(StringIter::new(&kept), dtype);

    // Distinct whole-string values (64-bit hashed; exact at this scale). Lets
    // the report compare OnPair against a plain value-dictionary encoding.
    let mut seen: HashSet<u64> = HashSet::new();
    view.with_iterator(|it| {
        for v in it {
            let mut h = DefaultHasher::new();
            match v {
                Some(b) => {
                    0u8.hash(&mut h);
                    b.hash(&mut h);
                }
                None => 1u8.hash(&mut h),
            }
            seen.insert(h.finish());
        }
    });

    Ok(Sample {
        array: view.into_array(),
        rows,
        raw_bytes: bytes,
        unique_count: seen.len() as u64,
    })
}

/// Per-element byte lengths of a string Arrow array (any of the three string
/// layouts). `None` if the column is not a string column.
fn string_byte_lengths(col: &dyn arrow_array::Array) -> Option<Vec<u64>> {
    if let Some(s) = col.as_any().downcast_ref::<StringArray>() {
        Some((0..s.len()).map(|i| s.value(i).len() as u64).collect())
    } else if let Some(s) = col.as_any().downcast_ref::<LargeStringArray>() {
        Some((0..s.len()).map(|i| s.value(i).len() as u64).collect())
    } else {
        col.as_any()
            .downcast_ref::<StringViewArray>()
            .map(|s| (0..s.len()).map(|i| s.value(i).len() as u64).collect())
    }
}

/// Iterator over `Option<&[u8]>` across a slice of string record batches.
struct StringIter<'a> {
    batches: &'a [RecordBatch],
}

impl<'a> StringIter<'a> {
    fn new(batches: &'a [RecordBatch]) -> Self {
        Self { batches }
    }
}

impl<'a> IntoIterator for StringIter<'a> {
    type Item = Option<&'a [u8]>;
    type IntoIter = Box<dyn Iterator<Item = Option<&'a [u8]>> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.batches.iter().flat_map(|b| {
            let col = b.column(0);
            let len = col.len();
            (0..len).map(move |i| {
                if col.is_null(i) {
                    None
                } else if let Some(s) = col.as_any().downcast_ref::<StringArray>() {
                    Some(s.value(i).as_bytes())
                } else if let Some(s) = col.as_any().downcast_ref::<LargeStringArray>() {
                    Some(s.value(i).as_bytes())
                } else {
                    col.as_any()
                        .downcast_ref::<StringViewArray>()
                        .map(|s| s.value(i).as_bytes())
                }
            })
        }))
    }
}

/// Equal-ish row ranges so each chunk holds roughly `chunk_bytes` of payload.
fn chunk_ranges(rows: usize, raw_bytes: u64, chunk_bytes: u64) -> Vec<std::ops::Range<usize>> {
    if rows == 0 {
        return vec![];
    }
    let n_chunks = usize::try_from(raw_bytes.div_ceil(chunk_bytes).max(1))
        .unwrap_or(usize::MAX)
        .min(rows);
    let per = rows.div_ceil(n_chunks);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < rows {
        let end = (start + per).min(rows);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// A `LayoutStrategy` that serialises chunks as-is (no recompression), so the
/// OnPair encoding survives to disk. `FlatLayoutStrategy::default()` allows all
/// encodings during normalization.
fn preserve_strategy() -> Arc<ChunkedLayoutStrategy> {
    Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()))
}

/// Rebuild an [`OnPairArray`] with each of its integer children compressed by
/// BtrBlocks (the dict byte blob is left untouched). The OnPair node — and so
/// its sorted-dictionary pushdown — is preserved; the children are
/// transparently decompressed by the decode path's `execute::<PrimitiveArray>`.
/// This is the big win for low-cardinality columns, whose codes / offsets /
/// lengths are extremely compressible.
fn compress_onpair_children(op: &OnPairArray, ctx: &mut ExecutionCtx) -> Result<OnPairArray> {
    let compressor = BtrBlocksCompressor::default();
    // dict_offsets and codes_offsets are monotonic prefix sums. Try both plain
    // and FastLanes Delta forms for those children and keep the smaller result.
    let dict_offsets = compress_offsets(op.dict_offsets(), &compressor, ctx)?;
    let codes_offsets = compress_offsets(op.codes_offsets(), &compressor, ctx)?;
    let codes = compressor.compress(op.codes(), ctx)?;
    let lengths = compressor.compress(op.uncompressed_lengths(), ctx)?;
    Ok(OnPair::try_new(
        op.dtype().clone(),
        op.dict_bytes_handle().clone(),
        dict_offsets,
        codes,
        codes_offsets,
        lengths,
        op.array_validity(),
        op.bits(),
    )?)
}

/// Compress a monotonic offset child by comparing compressor-only against
/// FastLanes Delta plus compressed `bases`/`deltas`, then keeping the smaller
/// representation.
fn compress_offsets(
    child: &ArrayRef,
    compressor: &BtrBlocksCompressor,
    ctx: &mut ExecutionCtx,
) -> Result<ArrayRef> {
    let plain = compressor.compress(child, ctx)?;
    let prim = child.clone().execute::<PrimitiveArray>(ctx)?;
    let len = prim.len();
    let children = Delta::try_from_primitive_array(&prim, ctx)?
        .into_array()
        .children();
    let bases = compressor.compress(&children[0], ctx)?;
    let deltas = compressor.compress(&children[1], ctx)?;
    let delta_arr = Delta::try_new(bases, deltas, 0, len)?.into_array();
    // Keep whichever is smaller in memory (a faithful proxy for on-disk, since
    // the OnPair tree is written as-is).
    Ok(if delta_arr.nbytes() < plain.nbytes() {
        delta_arr
    } else {
        plain
    })
}

/// Run the full `bits × chunk_bytes × threshold` matrix for one column.
#[expect(clippy::too_many_arguments)]
pub async fn run_column(
    dataset_id: &str,
    parquet_path: &Path,
    column: &str,
    bits: &[u32],
    chunk_bytes: &[u64],
    thresholds: &[f64],
    sample_bytes: u64,
    file_target_bytes: u64,
    out_root: &Path,
) -> Result<Vec<CellResult>> {
    let sample = build_sample(parquet_path, column, sample_bytes)?;
    tracing::info!(
        rows = sample.rows,
        raw_bytes = sample.raw_bytes,
        "sampled column '{column}'"
    );

    let mut results = Vec::new();
    for &b in bits {
        for &cb in chunk_bytes {
            for &thr in thresholds {
                let res = run_cell(
                    dataset_id,
                    column,
                    &sample,
                    b,
                    thr,
                    cb,
                    file_target_bytes,
                    out_root,
                )
                .await?;
                results.push(res);
            }
        }
    }
    Ok(results)
}

#[expect(clippy::too_many_arguments)]
async fn run_cell(
    dataset_id: &str,
    column: &str,
    sample: &Sample,
    bits: u32,
    threshold: f64,
    chunk_bytes: u64,
    file_target_bytes: u64,
    out_root: &Path,
) -> Result<CellResult> {
    let out_dir = out_root.join(dataset_id).join(column).join(format!(
        "bits{bits}_chunk{}_thr{:.2}",
        human_bytes(chunk_bytes),
        threshold,
    ));
    std::fs::create_dir_all(&out_dir)?;

    let ranges = chunk_ranges(sample.rows, sample.raw_bytes, chunk_bytes);
    let mut config = config_with_bits(bits);
    config.threshold = threshold;

    // 1. Compress each chunk (own dictionary), then BtrBlocks-compress the
    //    OnPair children, in parallel.
    let t0 = Instant::now();
    let compress_tasks = ranges.iter().cloned().map(|r| {
        let slice = sample.array.slice(r)?;
        anyhow::Ok(tokio::task::spawn_blocking(
            move || -> Result<OnPairArray> {
                let op = onpair_compress_array_default(&slice, config)?;
                let mut ctx = SESSION.create_execution_ctx();
                compress_onpair_children(&op, &mut ctx)
            },
        ))
    });
    let handles = compress_tasks.collect::<Result<Vec<_>>>()?;
    let onpairs = try_join_all(handles)
        .await?
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let encode_secs = t0.elapsed().as_secs_f64();

    let in_memory_bytes: u64 = onpairs
        .iter()
        .map(|a| a.clone().into_array().nbytes())
        .sum();
    let dict_bytes: u64 = onpairs.iter().map(|a| a.dict_bytes().len() as u64).sum();
    let n_chunks = onpairs.len();

    // 2. Group consecutive chunks into ~file_target_bytes files, written so the
    //    OnPair encoding is preserved on disk.
    let mut files: Vec<PathBuf> = Vec::new();
    let mut on_disk_bytes: u64 = 0;
    let mut group: Vec<ArrayRef> = Vec::new();
    let mut group_bytes: u64 = 0;
    let mut file_idx = 0usize;

    for op in &onpairs {
        let len = op.len();
        let chunk = StructArray::new(
            FieldNames::from([column]),
            vec![op.clone().into_array()],
            len,
            Validity::NonNullable,
        )
        .into_array();
        group_bytes += op.clone().into_array().nbytes();
        group.push(chunk);
        if group_bytes >= file_target_bytes {
            let path = out_dir.join(format!("part_{file_idx:04}.vortex"));
            on_disk_bytes += write_group(std::mem::take(&mut group), &path).await?;
            files.push(path);
            group_bytes = 0;
            file_idx += 1;
        }
    }
    if !group.is_empty() {
        let path = out_dir.join(format!("part_{file_idx:04}.vortex"));
        on_disk_bytes += write_group(group, &path).await?;
        files.push(path);
    }

    // 3. Read every file back and verify the string round-trip + that each
    //    chunk on disk is encoded purely as OnPair.
    let t1 = Instant::now();
    let (verified, onpair_only) = verify_roundtrip(&files, column, &sample.array).await?;
    let decode_secs = t1.elapsed().as_secs_f64();

    let gib = sample.raw_bytes as f64 / GIB;
    let result = CellResult {
        dataset_id: dataset_id.to_string(),
        column: column.to_string(),
        bits,
        threshold,
        chunk_bytes,
        rows: sample.rows as u64,
        unique_count: sample.unique_count,
        sample_bytes: sample.raw_bytes,
        n_chunks,
        in_memory_bytes,
        dict_bytes,
        on_disk_bytes,
        n_files: files.len(),
        encode_ms: encode_secs * 1e3,
        decode_ms: decode_secs * 1e3,
        encode_gib_s: if encode_secs > 0.0 {
            gib / encode_secs
        } else {
            0.0
        },
        decode_gib_s: if decode_secs > 0.0 {
            gib / decode_secs
        } else {
            0.0
        },
        mem_ratio: ratio(sample.raw_bytes, in_memory_bytes),
        disk_ratio: ratio(sample.raw_bytes, on_disk_bytes),
        verified,
        onpair_only,
        out_dir: out_dir.to_string_lossy().into_owned(),
    };

    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}

/// Write one group of OnPair chunks (wrapped in single-field structs) to a
/// `.vortex` file, returning its on-disk size.
async fn write_group(group: Vec<ArrayRef>, path: &Path) -> Result<u64> {
    let chunked = ChunkedArray::from_iter(group).into_array();
    let mut file = tokio::fs::File::create(path).await?;
    SESSION
        .write_options()
        .with_strategy(preserve_strategy())
        .write(&mut file, chunked.to_array_stream())
        .await?;
    Ok(std::fs::metadata(path)?.len())
}

/// Read every file back, canonicalize the single string field of each chunk,
/// and check (1) it equals the corresponding prefix of `original` and (2) the
/// on-disk field encoding is purely OnPair. Returns `(strings_match,
/// onpair_only)`.
async fn verify_roundtrip(
    files: &[PathBuf],
    column: &str,
    original: &ArrayRef,
) -> Result<(bool, bool)> {
    use futures::StreamExt;

    let mut row = 0usize;
    let mut onpair_only = true;
    for path in files {
        let vxf = SESSION.open_options().open_path(path.clone()).await?;
        let mut stream = vxf.scan()?.into_array_stream()?;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let strct = chunk.execute::<StructArray>(&mut SESSION.create_execution_ctx())?;
            let field = strct
                .unmasked_field_by_name(column)
                .with_context(|| format!("missing field '{column}' on read-back"))?;
            if field.encoding_id().to_string() != ONPAIR_ENCODING {
                onpair_only = false;
            }
            let decoded = field
                .clone()
                .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
            let len = decoded.len();
            let expected = original
                .slice(row..row + len)?
                .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;

            let ok = decoded.with_iterator(|dec| {
                expected.with_iterator(|exp| dec.zip(exp).all(|(a, b)| a == b))
            });
            if !ok {
                return Ok((false, onpair_only));
            }
            row += len;
        }
    }
    Ok((row == original.len(), onpair_only))
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn human_bytes(bytes: u64) -> String {
    const MB: u64 = 1 << 20;
    const KB: u64 = 1 << 10;
    if bytes.is_multiple_of(MB) {
        format!("{}mb", bytes / MB)
    } else if bytes.is_multiple_of(KB) {
        format!("{}kb", bytes / KB)
    } else {
        format!("{bytes}b")
    }
}

/// Generate every TPC-H table at `sf` as a single Parquet file per table under
/// `out_dir/parquet/<table>_0.parquet` (idempotent — existing files are kept).
///
/// Delegates to the shared [`generate_tpch_tables`](crate::tpch::tpchgen::generate_tpch_tables)
/// generator, which writes one file per table at the default (unbounded) file size.
pub async fn ensure_tpch_all_parquet(sf: f64, out_dir: &Path) -> Result<()> {
    use crate::Format;
    use crate::tpch::tpchgen::TpchGenOptions;
    use crate::tpch::tpchgen::generate_tpch_tables;

    // `generate_tpch_tables` is itself per-file idempotent; the lineitem marker
    // lets us skip the (cheap) probe entirely once a full set exists.
    if out_dir.join("parquet").join("lineitem_0.parquet").exists() {
        return Ok(());
    }
    std::fs::create_dir_all(out_dir)?;
    let options = TpchGenOptions::new(format!("{sf}"), out_dir).with_format(Format::Parquet);
    generate_tpch_tables(options).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ranges_equal_ish() {
        // 100 rows, 1000 bytes, 250-byte chunks -> 4 chunks of 25 rows.
        let r = chunk_ranges(100, 1000, 250);
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], 0..25);
        assert_eq!(r[3], 75..100);
    }

    #[test]
    fn chunk_ranges_single_when_budget_large() {
        let r = chunk_ranges(100, 1000, 1 << 30);
        assert_eq!(r, vec![0..100]);
    }

    #[test]
    fn chunk_ranges_caps_at_rows() {
        // Tiny budget would ask for more chunks than rows.
        let r = chunk_ranges(3, 1000, 1);
        assert_eq!(r.len(), 3);
    }
}
