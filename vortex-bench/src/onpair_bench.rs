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
#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicU64;
#[cfg(feature = "cuda")]
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use arrow_array::Array as _;
use arrow_array::LargeStringArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaView;
#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtrMut;
#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "cuda")]
use cudarc::driver::result::memset_d8_async;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUevent_flags;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
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
#[cfg(feature = "cuda")]
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
#[cfg(feature = "cuda")]
use vortex::array::vtable::child_to_validity;
use vortex::compressor::BtrBlocksCompressor;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
#[cfg(feature = "cuda")]
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::encodings::fastlanes::Delta;
#[cfg(feature = "cuda")]
use vortex::encodings::zstd::Zstd;
#[cfg(feature = "cuda")]
use vortex::encodings::zstd::ZstdDataParts;
#[cfg(feature = "cuda")]
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::utils::aliases::hash_set::HashSet;
#[cfg(feature = "cuda")]
use vortex_cuda::CudaBufferExt;
#[cfg(feature = "cuda")]
use vortex_cuda::CudaExecutionCtx;
#[cfg(feature = "cuda")]
use vortex_cuda::CudaKernelEvents;
#[cfg(feature = "cuda")]
use vortex_cuda::CudaSession;
#[cfg(feature = "cuda")]
use vortex_cuda::LaunchStrategy;
#[cfg(feature = "cuda")]
use vortex_cuda::ZstdKernelPrep;
#[cfg(feature = "cuda")]
use vortex_cuda::nvcomp::zstd as nvcomp_zstd;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::config_with_bits;
use vortex_onpair::onpair_compress_array_default;

use crate::SESSION;

const GIB: f64 = (1usize << 30) as f64;
#[cfg(feature = "cuda")]
const NVCOMP_ZSTD_VALUES_PER_FRAME: usize = 2048;
#[cfg(feature = "cuda")]
const NVCOMP_ZSTD_LEVEL: i32 = -10;
#[cfg(feature = "cuda")]
const NVCOMP_ZSTD_LEVELS: &[i32] = &[-10, 1, 3];

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
    /// CUDA kernel-only timings for GPU OnPair decompression, if requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuCellResult>,
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

/// CUDA kernel-only OnPair decompression results for one benchmark cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCellResult {
    /// Timed kernel launches per kernel variant.
    pub iterations: u64,
    /// Total bytes decoded into raw UTF-8 output buffers per iteration.
    pub decoded_bytes: u64,
    /// Number of OnPair chunks, and therefore launches per iteration.
    pub chunks: usize,
    /// Auto-selected kernel based on dictionary lengths.
    pub auto_kernel: String,
    /// Fastest measured kernel among all applicable variants.
    pub best_kernel: String,
    /// Auto-selected kernel average time across all chunks.
    pub auto_decode_ms: f64,
    /// Fastest measured kernel average time across all chunks.
    pub best_decode_ms: f64,
    /// Whether output byte validation was requested.
    pub validated: bool,
    /// Whether every applicable kernel produced bytes equal to CPU decode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// `decoded_bytes / auto_decode_ms`.
    pub auto_decode_gib_s: f64,
    /// `decoded_bytes / best_decode_ms`.
    pub best_decode_gib_s: f64,
    /// Per-kernel timing rows.
    pub kernels: Vec<GpuKernelResult>,
    /// nvCOMP ZSTD hardware-backend GPU decompression comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvcomp_zstd_hw: Option<NvcompZstdGpuResult>,
    /// nvCOMP ZSTD GPU decompression comparison over several compression levels.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub nvcomp_zstd: Vec<NvcompZstdGpuResult>,
}

/// nvCOMP ZSTD hardware-backend GPU decompression comparison for the same strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvcompZstdGpuResult {
    /// Whether nvCOMP accepted and ran the requested hardware backend.
    pub supported: bool,
    /// Error returned while forcing the hardware backend, if unsupported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Timed nvCOMP launches.
    pub iterations: u64,
    /// nvCOMP backend requested.
    pub backend: String,
    /// ZSTD compression level used to create comparison frames.
    pub zstd_level: i32,
    /// String values per independent ZSTD frame.
    pub values_per_frame: usize,
    /// Raw string bytes represented by the frames.
    pub raw_bytes: u64,
    /// Sum of compressed ZSTD frame sizes.
    pub compressed_bytes: u64,
    /// Number of independent ZSTD frames.
    pub frames: usize,
    /// `raw_bytes / compressed_bytes`.
    pub compression_ratio: f64,
    /// Average CUDA event time for one batched nvCOMP hardware decompress pass.
    pub decode_ms: f64,
    /// Raw string bytes per second.
    pub decode_gib_s: f64,
    /// Compressed input bytes per second.
    pub compressed_gib_s: f64,
}

/// CUDA kernel-only OnPair decompression results loaded from existing Vortex files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuVortexDecodeResult {
    /// Existing Vortex files used as input.
    pub files: Vec<String>,
    /// Column extracted from each file.
    pub column: String,
    /// Total logical rows across all OnPair chunks.
    pub rows: u64,
    /// Total in-memory bytes of the loaded OnPair chunks.
    pub in_memory_bytes: u64,
    /// Total OnPair dictionary bytes across all chunks.
    pub dict_bytes: u64,
    /// CUDA kernel-only timings and optional validation.
    pub gpu: GpuCellResult,
}

/// One CUDA kernel timing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuKernelResult {
    /// CUDA function name.
    pub kernel: String,
    /// Average CUDA event time for decoding all chunks once.
    pub decode_ms: f64,
    /// Raw decoded bytes per second.
    pub decode_gib_s: f64,
    /// Whether this kernel was applicable to all chunks.
    pub applicable: bool,
    /// Whether this kernel's GPU bytes matched CPU bytes, if validation was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// If not applicable, the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// If validation failed, the first mismatch detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_error: Option<String>,
}

/// Configuration for optional CUDA kernel-only OnPair decompression timing.
#[derive(Debug, Clone, Copy)]
pub struct GpuBenchmarkConfig {
    /// Timed iterations for each kernel variant.
    pub iterations: u64,
    /// Copy each kernel's raw output back and compare against CPU-decoded bytes.
    pub validate: bool,
}

/// Load existing benchmark `.vortex` files, extract an OnPair column, and run
/// CUDA kernel-only decompression without rebuilding the files from Parquet.
pub async fn run_vortex_gpu_decode(
    files: &[PathBuf],
    column: &str,
    gpu_config: GpuBenchmarkConfig,
) -> Result<GpuVortexDecodeResult> {
    let onpairs = read_onpair_chunks(files, column).await?;
    let rows = onpairs.iter().map(|a| a.len() as u64).sum();
    let in_memory_bytes = onpairs
        .iter()
        .map(|a| a.clone().into_array().nbytes())
        .sum();
    let dict_bytes = onpairs.iter().map(|a| a.dict_bytes().len() as u64).sum();
    let gpu = run_gpu_kernel_bench(&onpairs, gpu_config).await?;

    Ok(GpuVortexDecodeResult {
        files: files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        column: column.to_string(),
        rows,
        in_memory_bytes,
        dict_bytes,
        gpu,
    })
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
    gpu_config: Option<GpuBenchmarkConfig>,
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
                    gpu_config,
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
    gpu_config: Option<GpuBenchmarkConfig>,
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
    let gpu = match gpu_config {
        Some(config) => Some(run_gpu_kernel_bench(&onpairs, config).await?),
        None => None,
    };

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
        gpu,
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

#[cfg(not(feature = "cuda"))]
async fn run_gpu_kernel_bench(
    _onpairs: &[OnPairArray],
    _config: GpuBenchmarkConfig,
) -> Result<GpuCellResult> {
    anyhow::bail!(
        "GPU OnPair benchmark requested, but vortex-bench was built without --features cuda"
    )
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
struct TimedLaunchStrategy {
    total_time_ns: Arc<AtomicU64>,
}

#[cfg(feature = "cuda")]
impl TimedLaunchStrategy {
    fn timer(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.total_time_ns)
    }
}

#[cfg(feature = "cuda")]
impl LaunchStrategy for TimedLaunchStrategy {
    fn event_flags(&self) -> CUevent_flags {
        CU_EVENT_BLOCKING_SYNC
    }

    fn on_complete(&self, events: &CudaKernelEvents, _len: usize) -> VortexResult<()> {
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_nanos = events.duration()?.as_nanos() as u64;
        self.total_time_ns
            .fetch_add(elapsed_nanos, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
struct GpuOnPairChunk {
    rows: usize,
    decoded_bytes: u64,
    total_tokens: usize,
    dict_max_len: u8,
    dict_mean_len: f32,
    all_len_1: bool,
    all_len_2: bool,
    codes: vortex::array::buffer::BufferHandle,
    codes_offsets: vortex::array::buffer::BufferHandle,
    dict_padded: vortex::array::buffer::BufferHandle,
    dict_s8: vortex::array::buffer::BufferHandle,
    dict_s4: vortex::array::buffer::BufferHandle,
    dict_const1: vortex::array::buffer::BufferHandle,
    dict_const2: vortex::array::buffer::BufferHandle,
    dict_table: vortex::array::buffer::BufferHandle,
    dict_bytes: vortex::array::buffer::BufferHandle,
    output_offsets: vortex::array::buffer::BufferHandle,
    validity: vortex::array::buffer::BufferHandle,
    lens: vortex::array::buffer::BufferHandle,
    chunk_offsets_32: vortex::array::buffer::BufferHandle,
    chunk_offsets_64: vortex::array::buffer::BufferHandle,
    chunk_offsets_128: vortex::array::buffer::BufferHandle,
    chunk_offsets_256: vortex::array::buffer::BufferHandle,
    chunk_offsets_512: vortex::array::buffer::BufferHandle,
    chunk_offsets_1024: vortex::array::buffer::BufferHandle,
    output: vortex::array::buffer::BufferHandle,
    expected_bytes: Vec<u8>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
enum KernelLayout {
    Ref,
    Stride16,
    Stride8,
    Stride4,
    Const1,
    Const2,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct KernelVariant {
    name: &'static str,
    layout: KernelLayout,
    chunk_size: usize,
    block_warps: u32,
}

#[cfg(feature = "cuda")]
const GPU_KERNELS: &[KernelVariant] = &[
    KernelVariant {
        name: "onpair",
        layout: KernelLayout::Ref,
        chunk_size: 0,
        block_warps: 0,
    },
    KernelVariant {
        name: "onpair_shmem",
        layout: KernelLayout::Stride16,
        chunk_size: 32,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_2tpt",
        layout: KernelLayout::Stride16,
        chunk_size: 64,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_4tpt",
        layout: KernelLayout::Stride16,
        chunk_size: 128,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_4tpt_wpb8",
        layout: KernelLayout::Stride16,
        chunk_size: 128,
        block_warps: 8,
    },
    KernelVariant {
        name: "onpair_shmem_4tpt_wpb8_occ",
        layout: KernelLayout::Stride16,
        chunk_size: 128,
        block_warps: 8,
    },
    KernelVariant {
        name: "onpair_shmem_4tpt_split8",
        layout: KernelLayout::Stride16,
        chunk_size: 128,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_4tpt_split8_wpb8",
        layout: KernelLayout::Stride16,
        chunk_size: 128,
        block_warps: 8,
    },
    KernelVariant {
        name: "onpair_shmem_4tpt_split8_wpb8_occ",
        layout: KernelLayout::Stride16,
        chunk_size: 128,
        block_warps: 8,
    },
    KernelVariant {
        name: "onpair_shmem_s8",
        layout: KernelLayout::Stride8,
        chunk_size: 32,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_s8_2tpt",
        layout: KernelLayout::Stride8,
        chunk_size: 64,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_s8_4tpt",
        layout: KernelLayout::Stride8,
        chunk_size: 128,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_s8_8tpt",
        layout: KernelLayout::Stride8,
        chunk_size: 256,
        block_warps: 12,
    },
    KernelVariant {
        name: "onpair_shmem_s4l1",
        layout: KernelLayout::Stride4,
        chunk_size: 32,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_s4l1_2tpt",
        layout: KernelLayout::Stride4,
        chunk_size: 64,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_s4l1_4tpt",
        layout: KernelLayout::Stride4,
        chunk_size: 128,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_s4l1_8tpt",
        layout: KernelLayout::Stride4,
        chunk_size: 256,
        block_warps: 12,
    },
    KernelVariant {
        name: "onpair_shmem_s4l1_16tpt",
        layout: KernelLayout::Stride4,
        chunk_size: 512,
        block_warps: 8,
    },
    KernelVariant {
        name: "onpair_shmem_s4l1_32tpt",
        layout: KernelLayout::Stride4,
        chunk_size: 1024,
        block_warps: 8,
    },
    KernelVariant {
        name: "onpair_shmem_const1",
        layout: KernelLayout::Const1,
        chunk_size: 512,
        block_warps: 16,
    },
    KernelVariant {
        name: "onpair_shmem_const2",
        layout: KernelLayout::Const2,
        chunk_size: 256,
        block_warps: 16,
    },
];

#[cfg(feature = "cuda")]
async fn run_gpu_kernel_bench(
    onpairs: &[OnPairArray],
    config: GpuBenchmarkConfig,
) -> Result<GpuCellResult> {
    let iterations = config.iterations.max(1);
    let mut setup_ctx = create_cuda_execution_ctx()?;
    let mut chunks = Vec::with_capacity(onpairs.len());
    for op in onpairs {
        chunks.push(stage_gpu_chunk(op, &mut setup_ctx).await?);
    }
    setup_ctx.synchronize_stream()?;

    let decoded_bytes = chunks.iter().map(|c| c.decoded_bytes).sum::<u64>();
    let auto_kernel = pick_auto_kernel(&chunks).to_string();
    let mut kernels = Vec::with_capacity(GPU_KERNELS.len());

    for variant in GPU_KERNELS {
        if let Some(reason) = inapplicable_reason(*variant, &chunks) {
            kernels.push(GpuKernelResult {
                kernel: variant.name.to_string(),
                decode_ms: 0.0,
                decode_gib_s: 0.0,
                applicable: false,
                verified: None,
                reason: Some(reason),
                validation_error: None,
            });
            continue;
        }

        let decode_ms = time_kernel_variant(*variant, &chunks, iterations)?;
        let validation_error = if config.validate {
            validate_kernel_variant(*variant, &chunks).await.err()
        } else {
            None
        };
        let verified = config.validate.then_some(validation_error.is_none());
        kernels.push(GpuKernelResult {
            kernel: variant.name.to_string(),
            decode_ms,
            decode_gib_s: gib_s(decoded_bytes, decode_ms),
            applicable: true,
            verified,
            reason: None,
            validation_error: validation_error.map(|e| e.to_string()),
        });
    }

    let auto = kernels
        .iter()
        .find(|r| r.kernel == auto_kernel && r.applicable)
        .with_context(|| format!("auto kernel {auto_kernel} was not timed"))?;
    let best = kernels
        .iter()
        .filter(|r| r.applicable && (!config.validate || r.verified == Some(true)))
        .min_by(|a, b| a.decode_ms.total_cmp(&b.decode_ms))
        .context("no applicable verified CUDA OnPair kernels")?;
    let nvcomp_zstd_hw = run_nvcomp_zstd_bench(
        onpairs,
        iterations,
        NVCOMP_ZSTD_LEVEL,
        nvcomp_zstd::DecompressBackend::Hardware,
    )
    .await
    .unwrap_or_else(|error| NvcompZstdGpuResult {
        supported: false,
        error: Some(format!("{error:#}")),
        iterations,
        backend: "hardware".to_string(),
        zstd_level: NVCOMP_ZSTD_LEVEL,
        values_per_frame: NVCOMP_ZSTD_VALUES_PER_FRAME,
        raw_bytes: decoded_bytes,
        compressed_bytes: 0,
        frames: 0,
        compression_ratio: 0.0,
        decode_ms: 0.0,
        decode_gib_s: 0.0,
        compressed_gib_s: 0.0,
    });
    let mut nvcomp_zstd = Vec::with_capacity(NVCOMP_ZSTD_LEVELS.len());
    for &level in NVCOMP_ZSTD_LEVELS {
        nvcomp_zstd.push(
            run_nvcomp_zstd_bench(
                onpairs,
                iterations,
                level,
                nvcomp_zstd::DecompressBackend::Default,
            )
            .await
            .unwrap_or_else(|error| NvcompZstdGpuResult {
                supported: false,
                error: Some(format!("{error:#}")),
                iterations,
                backend: "default".to_string(),
                zstd_level: level,
                values_per_frame: NVCOMP_ZSTD_VALUES_PER_FRAME,
                raw_bytes: decoded_bytes,
                compressed_bytes: 0,
                frames: 0,
                compression_ratio: 0.0,
                decode_ms: 0.0,
                decode_gib_s: 0.0,
                compressed_gib_s: 0.0,
            }),
        );
    }

    Ok(GpuCellResult {
        iterations,
        decoded_bytes,
        chunks: chunks.len(),
        auto_kernel,
        best_kernel: best.kernel.clone(),
        auto_decode_ms: auto.decode_ms,
        best_decode_ms: best.decode_ms,
        validated: config.validate,
        verified: config.validate.then(|| {
            kernels
                .iter()
                .filter(|r| r.applicable)
                .all(|r| r.verified == Some(true))
        }),
        auto_decode_gib_s: auto.decode_gib_s,
        best_decode_gib_s: best.decode_gib_s,
        kernels,
        nvcomp_zstd_hw: Some(nvcomp_zstd_hw),
        nvcomp_zstd,
    })
}

#[cfg(feature = "cuda")]
async fn stage_gpu_chunk(op: &OnPairArray, ctx: &mut CudaExecutionCtx) -> Result<GpuOnPairChunk> {
    let codes_arr = op
        .codes()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;
    let codes_offsets_arr = op
        .codes_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;
    let dict_offsets_arr = op
        .dict_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;
    let lens_arr = op
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;
    let decoded = op
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(ctx.execution_ctx())?;
    let mut expected_bytes = Vec::with_capacity(usize::try_from(decoded.nbytes()).unwrap_or(0));
    decoded.with_iterator(|values| {
        for value in values.flatten() {
            expected_bytes.extend_from_slice(value);
        }
    });

    let codes_u16: Vec<u16> = match_each_integer_ptype!(codes_arr.ptype(), |P| {
        codes_arr
            .as_slice::<P>()
            .iter()
            .map(|&v| v as u16)
            .collect()
    });

    let dict_bytes_host = op.dict_bytes().as_slice();
    let (dict_padded, lens_table) = match_each_integer_ptype!(dict_offsets_arr.ptype(), |P| {
        let offsets = dict_offsets_arr.as_slice::<P>();
        let dict_size = offsets.len().saturating_sub(1);
        let mut padded = vec![0u8; dict_size * vortex_onpair::MAX_TOKEN_SIZE];
        let mut lens = vec![0u8; dict_size];
        for i in 0..dict_size {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let len = end.saturating_sub(start);
            padded[i * vortex_onpair::MAX_TOKEN_SIZE..i * vortex_onpair::MAX_TOKEN_SIZE + len]
                .copy_from_slice(&dict_bytes_host[start..end]);
            lens[i] = u8::try_from(len).unwrap_or(u8::MAX);
        }
        (padded, lens)
    });

    let decoded_bytes = match_each_integer_ptype!(lens_arr.ptype(), |P| {
        lens_arr.as_slice::<P>().iter().map(|&v| v as u64).sum()
    });
    let total_tokens = codes_u16.len();
    let dict_max_len = *lens_table.iter().max().unwrap_or(&0);
    let dict_mean_len = if lens_table.is_empty() {
        0.0
    } else {
        lens_table.iter().map(|&v| v as u64).sum::<u64>() as f32 / lens_table.len() as f32
    };
    let all_len_1 = !lens_table.is_empty() && lens_table.iter().all(|&l| l == 1);
    let all_len_2 = !lens_table.is_empty() && lens_table.iter().all(|&l| l == 2);

    let dict_table: Vec<u64> = match_each_integer_ptype!(dict_offsets_arr.ptype(), |P| {
        let offsets = dict_offsets_arr.as_slice::<P>();
        (0..offsets.len().saturating_sub(1))
            .map(|i| {
                let off = offsets[i] as u64;
                let len = (offsets[i + 1] - offsets[i]) as u64;
                (off << 16) | len
            })
            .collect()
    });
    let mut dict_bytes_with_pad = Vec::with_capacity(dict_bytes_host.len() + 16);
    dict_bytes_with_pad.extend_from_slice(dict_bytes_host);
    dict_bytes_with_pad.extend(std::iter::repeat_n(0u8, 16));

    let mut output_offsets = Vec::with_capacity(op.len() + 1);
    output_offsets.push(0u64);
    let mut acc = 0u64;
    match_each_integer_ptype!(lens_arr.ptype(), |P| {
        for &l in lens_arr.as_slice::<P>() {
            acc += l as u64;
            output_offsets.push(acc);
        }
    });
    let validity = vec![0xFFu8; op.len().div_ceil(8)];

    let mut dict_s8 = vec![0u8; lens_table.len() * 8];
    let mut dict_s4 = vec![0u8; lens_table.len() * 4];
    let mut dict_const1 = vec![0u8; lens_table.len()];
    let mut dict_const2 = vec![0u8; lens_table.len() * 2];
    for (i, len) in lens_table.iter().copied().enumerate() {
        let src = i * vortex_onpair::MAX_TOKEN_SIZE;
        let n8 = usize::from(len).min(8);
        dict_s8[i * 8..i * 8 + n8].copy_from_slice(&dict_padded[src..src + n8]);
        let n4 = usize::from(len).min(4);
        dict_s4[i * 4..i * 4 + n4].copy_from_slice(&dict_padded[src..src + n4]);
        if len >= 1 {
            dict_const1[i] = dict_padded[src];
        }
        let n2 = usize::from(len).min(2);
        dict_const2[i * 2..i * 2 + n2].copy_from_slice(&dict_padded[src..src + n2]);
    }

    let chunk_offsets_32 = chunk_offsets(&codes_u16, &lens_table, 32, decoded_bytes);
    let chunk_offsets_64 = chunk_offsets(&codes_u16, &lens_table, 64, decoded_bytes);
    let chunk_offsets_128 = chunk_offsets(&codes_u16, &lens_table, 128, decoded_bytes);
    let chunk_offsets_256 = chunk_offsets(&codes_u16, &lens_table, 256, decoded_bytes);
    let chunk_offsets_512 = chunk_offsets(&codes_u16, &lens_table, 512, decoded_bytes);
    let chunk_offsets_1024 = chunk_offsets(&codes_u16, &lens_table, 1024, decoded_bytes);

    Ok(GpuOnPairChunk {
        rows: op.len(),
        decoded_bytes,
        total_tokens,
        dict_max_len,
        dict_mean_len,
        all_len_1,
        all_len_2,
        codes: ctx.copy_to_device::<u16, _>(codes_u16)?.await?,
        codes_offsets: ctx
            .copy_to_device::<u64, _>(codes_offsets_to_u64(&codes_offsets_arr))?
            .await?,
        dict_padded: ctx.copy_to_device::<u8, _>(dict_padded)?.await?,
        dict_s8: ctx.copy_to_device::<u8, _>(dict_s8)?.await?,
        dict_s4: ctx.copy_to_device::<u8, _>(dict_s4)?.await?,
        dict_const1: ctx.copy_to_device::<u8, _>(dict_const1)?.await?,
        dict_const2: ctx.copy_to_device::<u8, _>(dict_const2)?.await?,
        dict_table: ctx.copy_to_device::<u64, _>(dict_table)?.await?,
        dict_bytes: ctx.copy_to_device::<u8, _>(dict_bytes_with_pad)?.await?,
        output_offsets: ctx.copy_to_device::<u64, _>(output_offsets)?.await?,
        validity: ctx.copy_to_device::<u8, _>(validity)?.await?,
        lens: ctx.copy_to_device::<u8, _>(lens_table)?.await?,
        chunk_offsets_32: ctx.copy_to_device::<u64, _>(chunk_offsets_32)?.await?,
        chunk_offsets_64: ctx.copy_to_device::<u64, _>(chunk_offsets_64)?.await?,
        chunk_offsets_128: ctx.copy_to_device::<u64, _>(chunk_offsets_128)?.await?,
        chunk_offsets_256: ctx.copy_to_device::<u64, _>(chunk_offsets_256)?.await?,
        chunk_offsets_512: ctx.copy_to_device::<u64, _>(chunk_offsets_512)?.await?,
        chunk_offsets_1024: ctx.copy_to_device::<u64, _>(chunk_offsets_1024)?.await?,
        output: ctx
            .copy_to_device::<u8, _>(vec![0u8; decoded_bytes as usize + 16])?
            .await?,
        expected_bytes,
    })
}

#[cfg(feature = "cuda")]
async fn run_nvcomp_zstd_bench(
    onpairs: &[OnPairArray],
    iterations: u64,
    zstd_level: i32,
    backend: nvcomp_zstd::DecompressBackend,
) -> Result<NvcompZstdGpuResult> {
    let iterations = iterations.max(1);
    let mut setup_ctx = create_cuda_execution_ctx()?;
    let mut values = Vec::new();
    let mut raw_bytes = 0u64;

    for op in onpairs {
        let decoded = op
            .clone()
            .into_array()
            .execute::<VarBinViewArray>(setup_ctx.execution_ctx())?;
        decoded.with_iterator(|iter| {
            for value in iter.flatten() {
                raw_bytes += value.len() as u64;
                values.push(value.to_vec());
            }
        });
    }

    if values.is_empty() {
        anyhow::bail!("cannot run nvCOMP zstd comparison for empty input");
    }

    let vbv = VarBinViewArray::from_iter_bin(values.iter().map(Vec::as_slice));
    let zstd_array = Zstd::from_var_bin_view_without_dict(
        &vbv,
        zstd_level,
        NVCOMP_ZSTD_VALUES_PER_FRAME,
        setup_ctx.execution_ctx(),
    )?;
    let opts = nvcomp_zstd::ZstdDecompressOpts { backend };

    let (compressed_bytes, frames) = {
        let validity = child_to_validity(
            zstd_array.as_ref().slots()[0].as_ref(),
            zstd_array.dtype().nullability(),
        );
        let parts: ZstdDataParts = zstd_array.clone().into_data().into_parts(validity);
        let bytes = parts.frames.iter().map(|f| f.len() as u64).sum();
        (bytes, parts.frames.len())
    };

    let mut ctx = create_cuda_execution_ctx()?;
    for _ in 0..2 {
        let exec = prepare_zstd_exec(&zstd_array, &mut ctx, opts, backend).await?;
        execute_nvcomp_zstd(exec, &mut ctx, opts, backend)?;
    }

    let mut total_ms = 0.0;
    for _ in 0..iterations {
        let exec = prepare_zstd_exec(&zstd_array, &mut ctx, opts, backend).await?;
        total_ms += execute_nvcomp_zstd(exec, &mut ctx, opts, backend)?;
    }
    ctx.synchronize_stream()?;

    let decode_ms = total_ms / iterations as f64;
    Ok(NvcompZstdGpuResult {
        supported: true,
        error: None,
        iterations,
        backend: nvcomp_backend_name(backend).to_string(),
        zstd_level,
        values_per_frame: NVCOMP_ZSTD_VALUES_PER_FRAME,
        raw_bytes,
        compressed_bytes,
        frames,
        compression_ratio: ratio(raw_bytes, compressed_bytes),
        decode_ms,
        decode_gib_s: gib_s(raw_bytes, decode_ms),
        compressed_gib_s: gib_s(compressed_bytes, decode_ms),
    })
}

#[cfg(feature = "cuda")]
async fn prepare_zstd_exec(
    zstd_array: &vortex::encodings::zstd::ZstdArray,
    ctx: &mut CudaExecutionCtx,
    opts: nvcomp_zstd::ZstdDecompressOpts,
    backend: nvcomp_zstd::DecompressBackend,
) -> Result<ZstdKernelPrep> {
    let validity = child_to_validity(
        zstd_array.as_ref().slots()[0].as_ref(),
        zstd_array.dtype().nullability(),
    );
    let parts: ZstdDataParts = zstd_array.clone().into_data().into_parts(validity);
    let ZstdDataParts {
        frames, metadata, ..
    } = parts;
    vortex_cuda::zstd_kernel_prepare_with_opts(frames, &metadata, ctx, opts)
        .await
        .with_context(|| {
            format!(
                "failed to prepare nvCOMP zstd {} decode",
                nvcomp_backend_name(backend)
            )
        })
}

#[cfg(feature = "cuda")]
fn execute_nvcomp_zstd(
    mut exec: ZstdKernelPrep,
    ctx: &mut CudaExecutionCtx,
    opts: nvcomp_zstd::ZstdDecompressOpts,
    backend: nvcomp_zstd::DecompressBackend,
) -> Result<f64> {
    let stream = ctx.stream();
    let cuda_ctx = stream.context();
    let start_event = cuda_ctx
        .new_event(Some(CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| anyhow::anyhow!("failed to create nvCOMP start event: {e:?}"))?;
    let end_event = cuda_ctx
        .new_event(Some(CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| anyhow::anyhow!("failed to create nvCOMP end event: {e:?}"))?;

    start_event
        .record(stream)
        .map_err(|e| anyhow::anyhow!("failed to record nvCOMP start event: {e:?}"))?;

    let (device_actual_sizes_ptr, record_actual_sizes) =
        exec.device_actual_sizes.device_ptr_mut(stream);
    let (nvcomp_temp_buffer_ptr, record_temp) = exec.nvcomp_temp_buffer.device_ptr_mut(stream);
    let (device_statuses_ptr, record_statuses) = exec.device_statuses.device_ptr_mut(stream);

    unsafe {
        nvcomp_zstd::decompress_async_with_opts(
            exec.frame_ptrs_ptr as _,
            exec.frame_sizes_ptr as _,
            exec.output_sizes_ptr as _,
            device_actual_sizes_ptr as _,
            exec.num_frames,
            nvcomp_temp_buffer_ptr as _,
            exec.nvcomp_temp_buffer_size,
            exec.output_ptrs_ptr as _,
            device_statuses_ptr as _,
            stream.cu_stream().cast(),
            opts,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "nvCOMP zstd {} decompress failed: {e}",
                nvcomp_backend_name(backend)
            )
        })?;
    }
    drop((record_actual_sizes, record_temp, record_statuses));

    end_event
        .record(stream)
        .map_err(|e| anyhow::anyhow!("failed to record nvCOMP end event: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("failed to synchronize nvCOMP stream: {e:?}"))?;
    let elapsed_ms = start_event.elapsed_ms(&end_event).map_err(|e| {
        anyhow::anyhow!(
            "failed to time nvCOMP {} decode: {e:?}",
            nvcomp_backend_name(backend)
        )
    })?;
    Ok(f64::from(elapsed_ms))
}

#[cfg(feature = "cuda")]
fn nvcomp_backend_name(backend: nvcomp_zstd::DecompressBackend) -> &'static str {
    match backend {
        nvcomp_zstd::DecompressBackend::Default => "default",
        nvcomp_zstd::DecompressBackend::Hardware => "hardware",
        nvcomp_zstd::DecompressBackend::Cuda => "cuda",
    }
}

#[cfg(feature = "cuda")]
fn codes_offsets_to_u64(codes_offsets_arr: &PrimitiveArray) -> Vec<u64> {
    match_each_integer_ptype!(codes_offsets_arr.ptype(), |P| {
        codes_offsets_arr
            .as_slice::<P>()
            .iter()
            .map(|&v| v as u64)
            .collect()
    })
}

#[cfg(feature = "cuda")]
fn chunk_offsets(codes: &[u16], lens: &[u8], chunk_size: usize, expected_total: u64) -> Vec<u64> {
    let total_chunks = codes.len().div_ceil(chunk_size);
    let mut offsets = Vec::with_capacity(total_chunks + 1);
    offsets.push(0);
    let mut acc = 0u64;
    for chunk_idx in 0..total_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(codes.len());
        for code in &codes[start..end] {
            acc += u64::from(lens[usize::from(*code)]);
        }
        offsets.push(acc);
    }
    debug_assert_eq!(acc, expected_total);
    offsets
}

#[cfg(feature = "cuda")]
fn inapplicable_reason(variant: KernelVariant, chunks: &[GpuOnPairChunk]) -> Option<String> {
    match variant.layout {
        KernelLayout::Ref | KernelLayout::Stride16 => None,
        KernelLayout::Stride8 => chunks
            .iter()
            .any(|c| c.dict_max_len > 8)
            .then(|| "dictionary entry longer than 8 bytes".to_string()),
        KernelLayout::Stride4 => chunks
            .iter()
            .any(|c| c.dict_max_len > 4)
            .then(|| "dictionary entry longer than 4 bytes".to_string()),
        KernelLayout::Const1 => chunks
            .iter()
            .any(|c| !c.all_len_1)
            .then(|| "not every dictionary entry is exactly 1 byte".to_string()),
        KernelLayout::Const2 => chunks
            .iter()
            .any(|c| !c.all_len_2)
            .then(|| "not every dictionary entry is exactly 2 bytes".to_string()),
    }
}

#[cfg(feature = "cuda")]
fn pick_auto_kernel(chunks: &[GpuOnPairChunk]) -> &'static str {
    if chunks.iter().all(|c| c.all_len_1) {
        return "onpair_shmem_const1";
    }
    if chunks.iter().all(|c| c.all_len_2) {
        return "onpair_shmem_const2";
    }

    let dict_max_len = chunks.iter().map(|c| c.dict_max_len).max().unwrap_or(16);
    let mean_bpt = if chunks.is_empty() {
        0.0
    } else {
        chunks.iter().map(|c| c.dict_mean_len).sum::<f32>() / chunks.len() as f32
    };
    let _ = mean_bpt;

    if dict_max_len <= 4 {
        "onpair_shmem_s4l1_16tpt"
    } else if dict_max_len <= 8 {
        "onpair_shmem_s8_4tpt"
    } else {
        "onpair_shmem_2tpt"
    }
}

#[cfg(feature = "cuda")]
fn time_kernel_variant(
    variant: KernelVariant,
    chunks: &[GpuOnPairChunk],
    iterations: u64,
) -> Result<f64> {
    let timed = TimedLaunchStrategy::default();
    let timer = timed.timer();
    let mut ctx = create_cuda_execution_ctx()?.with_launch_strategy(Arc::new(timed));
    let function = if matches!(variant.layout, KernelLayout::Ref) {
        ctx.load_function("onpair", &[u64::PTYPE])?
    } else {
        ctx.load_function(variant.name, &[])?
    };

    for _ in 0..2 {
        for chunk in chunks {
            launch_variant(&mut ctx, &function, variant, chunk)?;
        }
    }
    timer.store(0, Ordering::Relaxed);
    for _ in 0..iterations {
        for chunk in chunks {
            launch_variant(&mut ctx, &function, variant, chunk)?;
        }
    }
    ctx.synchronize_stream()?;

    Ok(timer.load(Ordering::Relaxed) as f64 / 1_000_000.0 / iterations as f64)
}

#[cfg(feature = "cuda")]
async fn validate_kernel_variant(variant: KernelVariant, chunks: &[GpuOnPairChunk]) -> Result<()> {
    let mut ctx = create_cuda_execution_ctx()?;
    let function = if matches!(variant.layout, KernelLayout::Ref) {
        ctx.load_function("onpair", &[u64::PTYPE])?
    } else {
        ctx.load_function(variant.name, &[])?
    };

    poison_kernel_outputs(&ctx, chunks)?;
    for chunk in chunks {
        launch_variant(&mut ctx, &function, variant, chunk)?;
    }
    ctx.synchronize_stream()?;

    for (idx, chunk) in chunks.iter().enumerate() {
        let host = chunk.output.clone().into_host().await;
        let actual = &host.as_ref()[..chunk.expected_bytes.len()];
        if actual != chunk.expected_bytes.as_slice() {
            let mismatch = actual
                .iter()
                .zip(&chunk.expected_bytes)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| actual.len().min(chunk.expected_bytes.len()));
            anyhow::bail!(
                "{} chunk {} output mismatch at byte {}: gpu={:?} cpu={:?}",
                variant.name,
                idx,
                mismatch,
                actual.get(mismatch),
                chunk.expected_bytes.get(mismatch)
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn create_cuda_execution_ctx() -> Result<CudaExecutionCtx> {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CudaSession::create_execution_ctx(&SESSION)
    }));
    std::panic::set_hook(previous_hook);

    match result {
        Ok(ctx) => ctx.context("failed to create CUDA execution context"),
        Err(payload) => anyhow::bail!(
            "failed to initialize CUDA execution context: {}",
            panic_payload_message(payload.as_ref())
        ),
    }
}

#[cfg(feature = "cuda")]
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(feature = "cuda")]
fn poison_kernel_outputs(ctx: &CudaExecutionCtx, chunks: &[GpuOnPairChunk]) -> Result<()> {
    for chunk in chunks {
        let ptr = chunk.output.cuda_device_ptr()?;
        // Validation must not inherit correct bytes from a previous kernel.
        unsafe {
            memset_d8_async(ptr, 0xA5, chunk.output.len(), ctx.stream().cu_stream())
                .map_err(|e| anyhow::anyhow!("failed to poison GPU output buffer: {e}"))?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_variant(
    ctx: &mut CudaExecutionCtx,
    function: &cudarc::driver::CudaFunction,
    variant: KernelVariant,
    chunk: &GpuOnPairChunk,
) -> Result<()> {
    let codes = chunk.codes.cuda_view::<u16>()?;
    let output = chunk.output.cuda_view::<u8>()?;
    let total_tokens = chunk.total_tokens as u64;

    match variant.layout {
        KernelLayout::Ref => {
            let codes_offsets = chunk.codes_offsets.cuda_view::<u64>()?;
            let dict_table = chunk.dict_table.cuda_view::<u64>()?;
            let dict_bytes = chunk.dict_bytes.cuda_view::<u8>()?;
            let output_offsets = chunk.output_offsets.cuda_view::<u64>()?;
            let validity = chunk.validity.cuda_view::<u8>()?;
            let rows = chunk.rows as u64;
            ctx.launch_kernel(function, chunk.rows, |args| {
                args.arg(&codes)
                    .arg(&codes_offsets)
                    .arg(&dict_table)
                    .arg(&dict_bytes)
                    .arg(&output_offsets)
                    .arg(&validity)
                    .arg(&output)
                    .arg(&rows);
            })?;
        }
        KernelLayout::Stride16 | KernelLayout::Stride8 | KernelLayout::Stride4 => {
            let (dict, chunk_offsets) = match variant.layout {
                KernelLayout::Stride16 => (
                    chunk.dict_padded.cuda_view::<u8>()?,
                    chunk_offsets_for_variant(chunk, variant.chunk_size)?,
                ),
                KernelLayout::Stride8 => (
                    chunk.dict_s8.cuda_view::<u8>()?,
                    chunk_offsets_for_variant(chunk, variant.chunk_size)?,
                ),
                KernelLayout::Stride4 => (
                    chunk.dict_s4.cuda_view::<u8>()?,
                    chunk_offsets_for_variant(chunk, variant.chunk_size)?,
                ),
                _ => unreachable!(),
            };
            let lens = chunk.lens.cuda_view::<u8>()?;
            let cfg = launch_config(chunk.total_tokens, variant.chunk_size, variant.block_warps);
            ctx.launch_kernel_config(function, cfg, chunk.total_tokens, |args| {
                args.arg(&codes)
                    .arg(&chunk_offsets)
                    .arg(&dict)
                    .arg(&lens)
                    .arg(&output)
                    .arg(&total_tokens);
            })?;
        }
        KernelLayout::Const1 => {
            let dict = chunk.dict_const1.cuda_view::<u8>()?;
            let cfg = launch_config(chunk.total_tokens, variant.chunk_size, variant.block_warps);
            ctx.launch_kernel_config(function, cfg, chunk.total_tokens, |args| {
                args.arg(&codes).arg(&dict).arg(&output).arg(&total_tokens);
            })?;
        }
        KernelLayout::Const2 => {
            let dict = chunk.dict_const2.cuda_view::<u8>()?;
            let cfg = launch_config(chunk.total_tokens, variant.chunk_size, variant.block_warps);
            ctx.launch_kernel_config(function, cfg, chunk.total_tokens, |args| {
                args.arg(&codes).arg(&dict).arg(&output).arg(&total_tokens);
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn chunk_offsets_for_variant(
    chunk: &GpuOnPairChunk,
    chunk_size: usize,
) -> Result<CudaView<'_, u64>> {
    match chunk_size {
        32 => Ok(chunk.chunk_offsets_32.cuda_view::<u64>()?),
        64 => Ok(chunk.chunk_offsets_64.cuda_view::<u64>()?),
        128 => Ok(chunk.chunk_offsets_128.cuda_view::<u64>()?),
        256 => Ok(chunk.chunk_offsets_256.cuda_view::<u64>()?),
        512 => Ok(chunk.chunk_offsets_512.cuda_view::<u64>()?),
        1024 => Ok(chunk.chunk_offsets_1024.cuda_view::<u64>()?),
        _ => anyhow::bail!("unsupported OnPair chunk size {chunk_size}"),
    }
}

#[cfg(feature = "cuda")]
fn launch_config(total_tokens: usize, chunk_size: usize, block_warps: u32) -> LaunchConfig {
    let total_chunks = total_tokens.div_ceil(chunk_size);
    LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks.div_ceil(block_warps as usize)).unwrap_or(u32::MAX),
            1,
            1,
        ),
        block_dim: (block_warps * 32, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
fn gib_s(bytes: u64, ms: f64) -> f64 {
    if ms == 0.0 {
        0.0
    } else {
        (bytes as f64 / GIB) / (ms / 1_000.0)
    }
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

async fn read_onpair_chunks(files: &[PathBuf], column: &str) -> Result<Vec<OnPairArray>> {
    use futures::StreamExt;

    let mut onpairs = Vec::new();
    for path in files {
        let vxf = SESSION.open_options().open_path(path.clone()).await?;
        let mut stream = vxf.scan()?.into_array_stream()?;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let strct = chunk.execute::<StructArray>(&mut SESSION.create_execution_ctx())?;
            let field = strct
                .unmasked_field_by_name(column)
                .with_context(|| format!("missing field '{column}' in {}", path.display()))?;
            if field.encoding_id().to_string() != ONPAIR_ENCODING {
                anyhow::bail!(
                    "{} field '{column}' is {}, expected {ONPAIR_ENCODING}",
                    path.display(),
                    field.encoding_id()
                );
            }
            onpairs.push(field.clone().try_downcast::<OnPair>().map_err(|array| {
                anyhow::anyhow!(
                    "{} field '{column}' could not be downcast as OnPair: {}",
                    path.display(),
                    array.encoding_id()
                )
            })?);
        }
    }

    if onpairs.is_empty() {
        anyhow::bail!("no OnPair chunks found for column '{column}'");
    }
    Ok(onpairs)
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
