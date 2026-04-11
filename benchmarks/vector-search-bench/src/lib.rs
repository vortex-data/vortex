// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector similarity-search benchmark core.
//!
//! For each `(dataset, variant)` pair we report:
//!
//! - **In-memory size** — `ArrayRef::nbytes()` of the prepared variant tree. This is the
//!   memory footprint you'd pay to keep that encoding resident.
//! - **Compress time** — the wall time spent in the encoding-specific compression pass after
//!   any shared pre-processing has already run (0 for the uncompressed variant itself, the
//!   BtrBlocks pass for `vortex-default`, and the TurboQuant encode step for
//!   `vortex-turboquant`).
//! - **Decompress time** — the wall time to execute the variant tree back into a
//!   canonical `FixedSizeListArray` (≈0 for the already-canonical uncompressed variant,
//!   meaningful for the compressed variants).
//! - **Cosine time** — executing `CosineSimilarity(data, const_query)` to a materialized
//!   f32 primitive array.
//! - **Filter time** — executing `Binary(Gt, [cosine, threshold])` to a `BoolArray`.
//! - **Recall@10** (for the lossy TurboQuant variant only) against exact top-10 from the
//!   uncompressed variant.
//!
//! Before any timing begins, the benchmark also runs a **correctness verification** pass
//! via [`verify`]: for every variant it computes cosine scores for a single query and
//! compares them to the ground-truth scores from the uncompressed variant. Lossless
//! variants must match within [`verify::LOSSLESS_TOLERANCE`]; lossy variants must match
//! within [`verify::LOSSY_TOLERANCE`]. A correctness failure bails the run.
//!
//! Measurements are emitted via the existing `vortex_bench::measurements` types so
//! results flow through the standard `gh-json` pipeline and show up on the CI dashboard
//! alongside compress-bench / random-access-bench.

use std::time::Duration;
use std::time::Instant;

pub mod display;
pub mod handrolled_baseline;
pub mod recall;
pub mod verify;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::ValueEnum;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::Extension;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::array::extension::EmptyMetadata;
use vortex::array::scalar::PValue;
use vortex::array::validity::Validity;
use vortex::buffer::BufferMut;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex::error::vortex_panic;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::conversions::list_to_vector_ext;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::Dataset;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_tensor::encodings::turboquant::TurboQuantConfig;
use vortex_tensor::encodings::turboquant::turboquant_encode_unchecked;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use vortex_tensor::vector::Vector;
use vortex_tensor::vector_search::build_constant_query_vector;
use vortex_tensor::vector_search::build_similarity_search_tree;

/// Lossy downcast of an f64 query vector to f32. This is intentional -- TurboQuant and
/// BtrBlocks operate in f32, so the query must match.
pub(crate) fn f64_to_f32_vec(v: &[f64]) -> Vec<f32> {
    #[expect(clippy::cast_possible_truncation)]
    v.iter().map(|&x| x as f32).collect()
}

/// The threshold used when wrapping the similarity expression in a
/// `Binary(Gt, [cosine, threshold])` filter. Set to a value high enough that random pairs
/// from a ~1.0-norm distribution reject but self-query pairs match.
pub const DEFAULT_THRESHOLD: f32 = 0.8;

/// Row index used to pick a query vector from the dataset. Using a fixed row keeps queries
/// reproducible across runs and guarantees at least one match (since `cosine(x, x) == 1.0`).
pub const DEFAULT_QUERY_ROW: usize = 0;

/// A single data-preparation strategy that the benchmark exercises.
///
/// Each variant corresponds to one column on the "format" axis in downstream dashboards. The
/// `Format` mapping is what gets serialized into the `target.format` field of gh-json
/// output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Variant {
    /// Raw `Vector<dim, f32>` with no encoding-level compression applied.
    #[clap(name = "vortex-uncompressed")]
    VortexUncompressed,
    /// `BtrBlocksCompressor::default()` walks into the `Vector` extension and recursively
    /// compresses the FSL storage child. This is the "generic lossless" Vortex story for
    /// float vectors.
    #[clap(name = "vortex-default")]
    VortexDefault,
    /// TurboQuant encoded vectors wrapped as `L2Denorm(SorfTransform(FSL(Dict)))`. Lossy;
    /// dramatic size win; requires reporting recall alongside throughput for the comparison
    /// to be honest. The benchmark times only the encode step; shared L2 normalization runs
    /// before timing starts.
    #[clap(name = "vortex-turboquant")]
    VortexTurboQuant,
}

impl Variant {
    /// The Format enum value this variant reports itself as in emitted measurements.
    /// Uncompressed and BtrBlocks-default both surface as [`Format::OnDiskVortex`]; the
    /// TurboQuant variant surfaces as [`Format::VortexLossy`] — the general
    /// file-format bucket for any `.vortex` file that contains lossy encodings —
    /// so dashboards can distinguish lossy runs from the lossless baseline.
    pub fn as_format(&self) -> Format {
        match self {
            Variant::VortexUncompressed => Format::OnDiskVortex,
            Variant::VortexDefault => Format::OnDiskVortex,
            Variant::VortexTurboQuant => Format::VortexLossy,
        }
    }

    /// A stable, kebab-cased label used in metric names so dashboards can split apart
    /// variants that map to the same Format.
    pub fn label(&self) -> &'static str {
        match self {
            Variant::VortexUncompressed => "vortex-uncompressed",
            Variant::VortexDefault => "vortex-default",
            Variant::VortexTurboQuant => "vortex-turboquant",
        }
    }
}

/// The ingested form of a dataset, ready to be fed to [`prepare_variant`] and the
/// timing/verification pipeline.
pub struct PreparedDataset {
    /// Name used in metric strings -- usually the dataset's `Dataset::name()`.
    pub name: String,
    /// Uncompressed `Vector<dim, T>` array in the original element type (`f32` or `f64`).
    /// Used as the ground-truth basis for verification and as the array for the
    /// [`Variant::VortexUncompressed`] variant.
    pub uncompressed: ArrayRef,
    /// f32-cast version of the uncompressed data. Identity when the source is already
    /// `f32`; a lossy cast for `f64` sources. Used as input to normalization and both
    /// compressed variants (BtrBlocks default, TurboQuant).
    pub uncompressed_f32: ArrayRef,
    /// Unit-norm normalized f32 vectors produced by [`normalize_as_l2_denorm`]. Both
    /// [`Variant::VortexDefault`] and [`Variant::VortexTurboQuant`] compress this
    /// instead of the raw `uncompressed` array, so normalization cost is excluded
    /// from compression timing.
    pub normalized: ArrayRef,
    /// L2 norms extracted alongside `normalized`. Used to rewrap compressed variants
    /// in [`L2Denorm`] after encoding.
    pub norms: ArrayRef,
    /// Query vector in `f64` precision. Used for ground-truth cosine computation on the
    /// original-precision uncompressed array.
    pub query_f64: Vec<f64>,
    /// Query vector cast to f32. Used for variant cosine computation on compressed
    /// (f32) arrays.
    pub query_f32: Vec<f32>,
    /// Parquet file size on disk in bytes -- produced by the dataset download step
    /// and reused as the "handrolled size" measurement in main.rs.
    pub parquet_bytes: u64,
}

impl PreparedDataset {
    /// Dimension of the underlying vector column.
    ///
    /// # Panics
    ///
    /// Panics if `self.uncompressed` is not an `Extension<FixedSizeList<_, dim, _>>` —
    /// which should be impossible because [`prepare_dataset`] is the only constructor
    /// and it guarantees this shape.
    pub fn dim(&self) -> u32 {
        let fsl_dtype = match self.uncompressed.dtype() {
            DType::Extension(ext) => ext.storage_dtype(),
            other => vortex_panic!("expected Extension<Vector>, got {other}"),
        };
        match fsl_dtype {
            DType::FixedSizeList(_, dim, _) => *dim,
            other => vortex_panic!("expected FixedSizeList storage, got {other}"),
        }
    }

    /// Number of rows in the uncompressed dataset.
    pub fn num_rows(&self) -> usize {
        self.uncompressed.len()
    }
}

/// Prepare a dataset by downloading its parquet file, converting the `emb` column to a
/// `Vector<dim, T>` extension array, and extracting a single-row query vector.
///
/// The benchmark pipeline currently supports source element types `f32` and `f64`.
/// [`vortex_bench::conversions::list_to_vector_ext`] can wrap `f16` inputs as a
/// `Vector` extension array, but the benchmark's query extraction, f32 cast path, and
/// hand-rolled parquet baseline are not wired for `f16` yet.
pub async fn prepare_dataset(dataset: &VectorDataset) -> Result<PreparedDataset> {
    let parquet_path = dataset
        .to_parquet_path()
        .await
        .context("download vector dataset parquet")?;
    let parquet_bytes = std::fs::metadata(&parquet_path)
        .with_context(|| format!("stat parquet file {:?}", parquet_path))?
        .len();

    tracing::info!(
        "ingesting {} parquet from {:?} ({} bytes)",
        dataset.name(),
        parquet_path,
        parquet_bytes
    );

    let chunked = parquet_to_vortex_chunks(parquet_path).await?;

    let struct_array = chunked.into_array();
    let emb_column = extract_emb_column(&struct_array)?;
    let wrapped = list_to_vector_ext(emb_column)?;

    // `list_to_vector_ext` may return a chunked `Extension<Vector>` when the source was
    // a `ChunkedArray` of list columns (the usual shape after `parquet_to_vortex_chunks`).
    // Materialize it into a single non-chunked `ExtensionArray` so downstream code can
    // treat it uniformly.
    let mut ctx = SESSION.create_execution_ctx();
    let uncompressed = wrapped.execute::<ExtensionArray>(&mut ctx)?.into_array();

    let query_f64 = extract_query_row_f64(&uncompressed, DEFAULT_QUERY_ROW)?;
    let query_f32: Vec<f32> = f64_to_f32_vec(&query_f64);

    // Cast to f32 for compressed variants. Identity when the source is already f32.
    let uncompressed_f32 = cast_vector_to_f32(&uncompressed, &mut ctx)?;

    // Pre-compute normalization on f32 data once so both compressed variants can skip it
    // during their timed encoding step.
    let l2_denorm = normalize_as_l2_denorm(uncompressed_f32.clone(), &mut ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();

    Ok(PreparedDataset {
        name: dataset.name().to_string(),
        uncompressed,
        uncompressed_f32,
        normalized,
        norms,
        query_f64,
        query_f32,
        parquet_bytes,
    })
}

/// Project the `emb` column out of a chunked struct array. This rebuilds a chunked list
/// array with just that one column.
fn extract_emb_column(struct_array: &ArrayRef) -> Result<ArrayRef> {
    if let Some(chunked) = struct_array.as_opt::<Chunked>() {
        let mut emb_chunks: Vec<ArrayRef> = Vec::with_capacity(chunked.nchunks());
        for chunk in chunked.iter_chunks() {
            emb_chunks.push(extract_emb_column(chunk)?);
        }
        if emb_chunks.is_empty() {
            bail!("dataset has no chunks");
        }
        return Ok(ChunkedArray::from_iter(emb_chunks).into_array());
    }

    let Some(struct_view) = struct_array.as_opt::<Struct>() else {
        bail!(
            "expected dataset chunks to be Struct arrays, got {}",
            struct_array.dtype()
        );
    };

    let field = struct_view
        .unmasked_field_by_name("emb")
        .context("dataset parquet must have an `emb` column")?;
    Ok(field.clone())
}

/// Pull a single row out of a `Vector<dim, T>` extension array as a `Vec<f64>`.
///
/// The benchmark currently supports `f32` and `f64` source vector columns.
pub(crate) fn extract_query_row_f64(vector_ext: &ArrayRef, row: usize) -> Result<Vec<f64>> {
    if row >= vector_ext.len() {
        bail!(
            "query row {row} out of bounds for dataset of length {}",
            vector_ext.len()
        );
    }

    let ext_view = vector_ext
        .as_opt::<Extension>()
        .context("prepared dataset must be a Vector extension array")?;

    let mut ctx = SESSION.create_execution_ctx();

    // Execute storage array to its canonical FSL form.
    let fsl: FixedSizeListArray = ext_view.storage_array().clone().execute(&mut ctx)?;

    let dim_usize = match fsl.dtype() {
        DType::FixedSizeList(_, d, _) => *d as usize,
        other => bail!("storage dtype must be FixedSizeList, got {other}"),
    };

    let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
    let start = row * dim_usize;
    let end = start + dim_usize;

    match elements.ptype() {
        PType::F32 => Ok(elements.as_slice::<f32>()[start..end]
            .iter()
            .map(|&v| f64::from(v))
            .collect()),
        PType::F64 => Ok(elements.as_slice::<f64>()[start..end].to_vec()),
        other => bail!("extract_query_row_f64 only supports f32/f64 Vector columns, got {other:?}"),
    }
}

/// Cast a `Vector<dim, T>` extension array's elements to `f32`. Returns the array
/// unchanged if it is already `f32`. For `f64` this is a lossy narrowing cast.
///
/// `f16` inputs are not currently supported by the benchmark pipeline.
fn cast_vector_to_f32(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    let ext: ExtensionArray = array.clone().execute(ctx)?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;

    if elements.ptype() == PType::F32 {
        return Ok(array.clone());
    }

    let dim = fsl.list_size();
    let num_rows = fsl.len();

    #[expect(clippy::cast_possible_truncation)]
    let f32_values: Vec<f32> = match elements.ptype() {
        PType::F64 => elements
            .as_slice::<f64>()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        other => bail!("cast_vector_to_f32: unsupported element ptype {other:?}"),
    };

    let f32_buf = BufferMut::<f32>::from_iter(f32_values).freeze();
    let f32_elements = PrimitiveArray::new::<f32>(f32_buf, Validity::NonNullable);
    let f32_fsl = FixedSizeListArray::try_new(
        f32_elements.into_array(),
        dim,
        Validity::NonNullable,
        num_rows,
    )?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, f32_fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, f32_fsl.into_array()).into_array())
}

/// A prepared variant: the in-memory array tree plus the metadata we want to report
/// alongside it (size and construction cost).
#[derive(Debug, Clone)]
pub struct PreparedVariant {
    /// The variant's in-memory array tree. For the uncompressed variant this is the same
    /// canonical `Extension<Vector>` pulled out of `prepare_dataset`; for the others it's
    /// the output of the respective compression pipeline.
    pub array: ArrayRef,
    /// Summed byte footprint of the variant tree — `ArrayRef::nbytes()`. This is the
    /// in-memory cost of keeping the variant resident, not a disk size.
    pub nbytes: u64,
    /// Wall time spent in the encoding-specific compression step after any shared
    /// pre-processing has already run. 0 for [`Variant::VortexUncompressed`];
    /// meaningful for the compressed variants.
    pub compress_duration: Duration,
}

/// Apply a `Variant`'s preparation strategy to the materialized uncompressed source and
/// return the resulting tree together with its reported in-memory size and construction
/// time. For compressed variants, shared normalization happens before timing starts so
/// the reported duration covers only encoding-specific work. Uses the global
/// [`vortex_bench::SESSION`] for any execution-context work; the benchmark has no reason
/// to support multiple concurrent sessions.
///
/// **Why nbytes instead of on-disk size?** The Vortex file writer applies BtrBlocks
/// compression as part of its default write strategy regardless of the in-memory tree
/// shape, so serializing an "uncompressed" tree and measuring the resulting `.vortex`
/// file produces the same bytes as serializing a `BtrBlocksCompressor::default()`-
/// compressed tree — the disk-size comparison collapses two conceptually different
/// things into one number. Reporting `nbytes()` of the in-memory tree keeps the size
/// measurement consistent with what the *compute* measurements operate on.
///
/// **Forward migration to disk-backed runs.** Today this benchmark keeps each
/// variant in memory because the TurboQuant pipeline's `L2Denorm` and
/// `SorfTransform` scalar functions do not yet implement `ScalarFnVTable::serialize`,
/// so writing a TurboQuant variant to a `.vortex` file and reading it back is not
/// round-trippable. Once those serialize impls land, this benchmark can switch to
/// disk-backed runs additively — there is no lossy-variant-specific code to unwind
/// here, because the `Format::VortexLossy` bucket was deliberately kept generic.
/// Variants continue to report their `Format` via `as_format()` exactly as they do
/// today; only the path from `PreparedVariant` to execution gains an optional
/// write/read hop.
pub fn prepare_variant(prepared: &PreparedDataset, variant: Variant) -> Result<PreparedVariant> {
    match variant {
        Variant::VortexUncompressed => {
            // Use the f32-cast data so all variants operate on f32 and timing
            // comparisons are meaningful.
            let start = Instant::now();
            let array = prepared.uncompressed_f32.clone();
            let compress_duration = start.elapsed();
            let nbytes = array.nbytes();
            Ok(PreparedVariant {
                array,
                nbytes,
                compress_duration,
            })
        }
        Variant::VortexDefault => {
            let num_rows = prepared.num_rows();

            let start = Instant::now();
            let compressed = BtrBlocksCompressor::default().compress(&prepared.normalized)?;
            let compress_duration = start.elapsed();

            let array = unsafe {
                L2Denorm::new_array_unchecked(compressed, prepared.norms.clone(), num_rows)
            }?
            .into_array();
            let nbytes = array.nbytes();
            Ok(PreparedVariant {
                array,
                nbytes,
                compress_duration,
            })
        }
        Variant::VortexTurboQuant => {
            let mut ctx = SESSION.create_execution_ctx();
            let num_rows = prepared.num_rows();
            let normalized_ext = prepared
                .normalized
                .as_opt::<Extension>()
                .context("normalized child must be an Extension array")?;
            let config = TurboQuantConfig::default();

            let start = Instant::now();
            // SAFETY: `normalize_as_l2_denorm` guarantees every row is unit-norm (or
            // zero), which is the invariant `turboquant_encode_unchecked` expects.
            let tq = unsafe { turboquant_encode_unchecked(normalized_ext, &config, &mut ctx) }?;
            let compress_duration = start.elapsed();

            let array =
                unsafe { L2Denorm::new_array_unchecked(tq, prepared.norms.clone(), num_rows) }?
                    .into_array();
            let nbytes = array.nbytes();
            Ok(PreparedVariant {
                array,
                nbytes,
                compress_duration,
            })
        }
    }
}

/// Run the decompress / cosine / filter microbenchmarks against a prepared variant
/// array and return the best-of-`iterations` wall times for each measurement.
///
/// The three stages are **interleaved** inside a single outer loop rather than run
/// as three separate back-to-back loops. Interleaving keeps each stage's cache /
/// branch-predictor / allocator state symmetric across iterations — a pathology of
/// the back-to-back shape is that iteration `N+1` of the cosine stage runs on
/// warmed caches left behind by iteration `N` of the cosine stage, while iteration
/// `N+1` of the filter stage runs on caches left behind by the *cosine* stage. The
/// interleaved form makes each stage see roughly the same cache state every
/// iteration.
///
/// Each stage still gets a fresh `ExecutionCtx` (from the global
/// [`vortex_bench::SESSION`]), so no cached scalar-fn state leaks between stages
/// within a single iteration.
pub fn run_timings<T: NativePType + Into<PValue>>(
    variant_array: &ArrayRef,
    query: &[T],
    threshold: T,
    iterations: usize,
) -> Result<VariantTimings> {
    if iterations == 0 {
        bail!("run_timings requires iterations >= 1");
    }

    let mut decompress = Duration::MAX;
    let mut cosine = Duration::MAX;
    let mut filter = Duration::MAX;

    for _ in 0..iterations {
        {
            let mut ctx = SESSION.create_execution_ctx();
            let start = Instant::now();
            let decoded: FixedSizeListArray = decompress_full_scan(variant_array, &mut ctx)?;
            decompress = decompress.min(start.elapsed());
            drop(decoded);
        }
        {
            let mut ctx = SESSION.create_execution_ctx();
            let start = Instant::now();
            let scores: PrimitiveArray = execute_cosine(variant_array, query, &mut ctx)?;
            cosine = cosine.min(start.elapsed());
            drop(scores);
        }
        {
            let mut ctx = SESSION.create_execution_ctx();
            let start = Instant::now();
            let matches: BoolArray = execute_filter(variant_array, query, threshold, &mut ctx)?;
            filter = filter.min(start.elapsed());
            drop(matches);
        }
    }

    Ok(VariantTimings {
        decompress,
        cosine,
        filter,
    })
}

/// Timing summary for one `(dataset, variant)` pair.
#[derive(Debug, Clone, Copy)]
pub struct VariantTimings {
    /// Wall time to execute the variant's array tree back into a canonical
    /// `FixedSizeListArray`. ~0 for [`Variant::VortexUncompressed`] (the tree is already
    /// canonical), meaningful for the two compressed variants.
    pub decompress: Duration,
    /// Wall time for the cosine_similarity scalar-function execution over the whole
    /// column (materialized into an `f32` [`PrimitiveArray`]).
    pub cosine: Duration,
    /// Wall time for the full `Binary(Gt, [cosine, threshold])` expression executed
    /// into a [`BoolArray`].
    pub filter: Duration,
}

/// Fully materialize the input column so the measurement captures *all* decompression
/// work — the extension shell, the FSL storage, **and the inner f32 element buffer**.
///
/// Forcing the element buffer to materialize as a canonical `PrimitiveArray<f32>` is
/// what distinguishes this from a no-op cache hit. Executing the `ExtensionArray` or
/// `FixedSizeListArray` alone only unwraps the container shells — if the FSL's
/// `elements` child is (e.g.) an `alprd` tree, the bit-unpacking is lazy and only
/// happens when something reads the values. The `execute::<PrimitiveArray>` call below
/// forces that read.
///
/// For the Vortex-uncompressed variant this is cheap (bitwise copy / no-op). For
/// BtrBlocks-default it includes the ALP-RD decoding pass. For TurboQuant it includes
/// running the inverse SORF rotation + dictionary lookup through the scalar-fn
/// pipeline.
pub fn decompress_full_scan(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Result<FixedSizeListArray> {
    let ext: ExtensionArray = array.clone().execute(ctx)?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    // Force the element buffer all the way down to a canonical PrimitiveArray so the
    // timing captures any lazy decode work (ALP-RD bit unpacking, dict lookups, SORF
    // inverse rotation, etc.).
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    drop(elements);
    Ok(fsl)
}

/// Execute `CosineSimilarity(data, broadcast(query))` to a materialized
/// [`PrimitiveArray`]. Shared between the timing loop and the correctness-verification
/// path so both exercise the exact same expression tree.
///
/// The element type `T` must match the element type of `data`'s [`Vector`] extension
/// dtype (e.g. pass `&[f32]` for f32 data, `&[f64]` for f64 data).
///
/// # Errors
///
/// Returns an error if `data` is not a [`vortex_tensor::vector::Vector`] extension array,
/// if `query`'s length doesn't match the database vector dimension, or if the execution
/// context rejects the expression.
pub fn execute_cosine<T: NativePType + Into<PValue>>(
    data: &ArrayRef,
    query: &[T],
    ctx: &mut ExecutionCtx,
) -> Result<PrimitiveArray> {
    let num_rows = data.len();
    let query_vec = build_constant_query_vector(query, num_rows)?;
    let cosine = CosineSimilarity::try_new_array(data.clone(), query_vec, num_rows)?.into_array();
    Ok(cosine.execute(ctx)?)
}

fn execute_filter<T: NativePType + Into<PValue>>(
    data: &ArrayRef,
    query: &[T],
    threshold: T,
    ctx: &mut ExecutionCtx,
) -> Result<BoolArray> {
    let tree = build_similarity_search_tree(data.clone(), query, threshold)?;
    Ok(tree.execute(ctx)?)
}

/// Test-only helpers shared between the unit tests in this crate's submodules.
#[cfg(test)]
pub(crate) mod test_utils {
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::extension::EmptyMetadata;
    use vortex::array::validity::Validity;
    use vortex::buffer::BufferMut;
    use vortex::dtype::extension::ExtDType;
    use vortex_tensor::vector::Vector;

    /// Build a deterministic `Vector<dim, f32>` extension array of `num_rows` rows for
    /// tests. The PRNG is a trivial xorshift keyed by `seed`; we don't care about the
    /// distribution beyond "not all zeros".
    pub fn synthetic_vector(dim: u32, num_rows: usize, seed: u64) -> ArrayRef {
        let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim as usize);
        let mut state = seed;
        for _ in 0..(num_rows * dim as usize) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state & 0xFFFF) as f32 / 32768.0) - 1.0;
            buf.push(v);
        }
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable).into_array();
        let fsl =
            FixedSizeListArray::try_new(elements, dim, Validity::NonNullable, num_rows).unwrap();
        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
            .unwrap()
            .erased();
        ExtensionArray::new(ext_dtype, fsl.into_array()).into_array()
    }
}

#[cfg(test)]
mod tests {
    use super::test_utils::synthetic_vector;
    use super::*;

    /// Build a test `PreparedDataset` from synthetic f32 data.
    fn test_prepared(dim: u32, num_rows: usize, seed: u64) -> PreparedDataset {
        let uncompressed = synthetic_vector(dim, num_rows, seed);
        let query_f64 = extract_query_row_f64(&uncompressed, 0).unwrap();
        let query_f32: Vec<f32> = f64_to_f32_vec(&query_f64);

        let mut ctx = SESSION.create_execution_ctx();
        // Synthetic data is f32, so uncompressed_f32 == uncompressed.
        let uncompressed_f32 = uncompressed.clone();
        let l2_denorm = normalize_as_l2_denorm(uncompressed_f32.clone(), &mut ctx).unwrap();
        let normalized = l2_denorm.child_at(0).clone();
        let norms = l2_denorm.child_at(1).clone();

        PreparedDataset {
            name: "synthetic".to_string(),
            uncompressed,
            uncompressed_f32,
            normalized,
            norms,
            query_f64,
            query_f32,
            parquet_bytes: 0,
        }
    }

    #[test]
    fn extract_query_row_f64_returns_the_right_slice() {
        let dim = 8u32;
        let num_rows = 4usize;
        let prepared = test_prepared(dim, num_rows, 0xDEADBEEF);

        let row0 = extract_query_row_f64(&prepared.uncompressed, 0).unwrap();
        let row3 = extract_query_row_f64(&prepared.uncompressed, 3).unwrap();
        assert_eq!(row0, prepared.query_f64);
        assert_eq!(row0.len(), dim as usize);
        assert_eq!(row3.len(), dim as usize);
        assert_ne!(row0, row3, "different rows must differ for this seed");
    }

    #[test]
    fn extract_query_row_f64_rejects_out_of_bounds_row() {
        let dim = 8u32;
        let num_rows = 4usize;
        let prepared = test_prepared(dim, num_rows, 0xC0FFEE);

        let err = extract_query_row_f64(&prepared.uncompressed, 4)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("query row 4 out of bounds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn prepare_variant_produces_non_empty_array_for_all_variants() {
        let dim = 128u32;
        let num_rows = 64usize;
        let prepared = test_prepared(dim, num_rows, 0xC0FFEE);

        for variant in [
            Variant::VortexUncompressed,
            Variant::VortexDefault,
            Variant::VortexTurboQuant,
        ] {
            let prep = prepare_variant(&prepared, variant).unwrap();
            assert_eq!(
                prep.array.len(),
                num_rows,
                "variant {variant:?} changed row count"
            );
            assert!(prep.nbytes > 0, "variant {variant:?} reported zero size");

            let timings =
                run_timings(&prep.array, &prepared.query_f32, DEFAULT_THRESHOLD, 2).unwrap();
            // TurboQuant + default must do real work; uncompressed's decompress is a
            // no-op and can plausibly time as zero.
            assert!(timings.cosine > Duration::ZERO);
            assert!(timings.filter > Duration::ZERO);
        }
    }

    #[test]
    fn run_timings_rejects_zero_iterations() {
        let prepared = test_prepared(128, 64, 0xC0FFEE);
        let prep = prepare_variant(&prepared, Variant::VortexUncompressed).unwrap();

        let err = run_timings(&prep.array, &prepared.query_f32, DEFAULT_THRESHOLD, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("iterations >= 1"), "unexpected error: {err}");
    }

    /// The **uncompressed** variant's decompress pass must be a no-op (the tree is
    /// already canonical), while TurboQuant must do real work. This is a regression
    /// guard for a future change accidentally making the uncompressed variant take the
    /// slow path.
    #[test]
    fn uncompressed_decompress_is_fast() {
        let dim = 128u32;
        let num_rows = 256usize;
        let prepared = test_prepared(dim, num_rows, 0xDEADBEEF);

        let uncompressed_prep = prepare_variant(&prepared, Variant::VortexUncompressed).unwrap();
        let turboquant_prep = prepare_variant(&prepared, Variant::VortexTurboQuant).unwrap();

        let unc_timings = run_timings(
            &uncompressed_prep.array,
            &prepared.query_f32,
            DEFAULT_THRESHOLD,
            3,
        )
        .unwrap();
        let tq_timings = run_timings(
            &turboquant_prep.array,
            &prepared.query_f32,
            DEFAULT_THRESHOLD,
            3,
        )
        .unwrap();

        // The uncompressed decompress should be at least an order of magnitude faster
        // than TurboQuant's (usually many orders of magnitude). 5x is a loose lower
        // bound that won't flake on a noisy CI runner.
        assert!(
            tq_timings.decompress > unc_timings.decompress * 5,
            "expected TurboQuant decompress ({:?}) to be >5x uncompressed ({:?})",
            tq_timings.decompress,
            unc_timings.decompress
        );
    }

    /// Diagnostic: print the in-memory tree shape for each variant so we can see
    /// exactly what BtrBlocks and TurboQuant do to the FSL storage.
    ///
    /// Run with:
    /// ```bash
    /// cargo test -p vector-search-bench --release -- \
    ///     --ignored --nocapture print_variant_trees
    /// ```
    #[test]
    #[ignore]
    #[expect(clippy::use_debug, reason = "human-readable diagnostic output")]
    fn print_variant_trees() {
        let dim = 768u32;
        let num_rows = 500usize;
        let prepared = test_prepared(dim, num_rows, 0xC0FFEE);

        for variant in [
            Variant::VortexUncompressed,
            Variant::VortexDefault,
            Variant::VortexTurboQuant,
        ] {
            let prep = prepare_variant(&prepared, variant).unwrap();
            println!("=== {variant:?} ===");
            println!("  len              : {}", prep.array.len());
            println!("  nbytes           : {}", prep.nbytes);
            println!("  compress_duration: {:?}", prep.compress_duration);
            println!(
                "  encoding tree    : {}",
                prep.array.display_tree_encodings_only()
            );
        }
    }
}
