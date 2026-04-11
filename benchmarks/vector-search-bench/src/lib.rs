// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector similarity-search benchmark core.
//!
//! For each `(dataset, variant)` pair we report:
//!
//! - **In-memory size** — `ArrayRef::nbytes()` of the prepared variant tree. This is the
//!   memory footprint you'd pay to keep that encoding resident.
//! - **Compress time** — the wall time to build the variant tree from the materialized
//!   uncompressed source (0 for the uncompressed variant itself, the BtrBlocks pass for
//!   `vortex-default`, the full L2Denorm+SORF+quantize pipeline for `vortex-turboquant`).
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
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::vortex_panic;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::conversions::list_to_vector_ext;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::Dataset;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector_search::build_constant_query_vector;
use vortex_tensor::vector_search::build_similarity_search_tree;
use vortex_tensor::vector_search::compress_turboquant;

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
    /// The full TurboQuant pipeline: `L2Denorm(SorfTransform(FSL(Dict)))`. Lossy; dramatic
    /// size win; requires reporting recall alongside throughput for the comparison to be
    /// honest. See [`vortex_tensor::vector_search::compress_turboquant`].
    #[clap(name = "vortex-turboquant")]
    VortexTurboQuant,
}

impl Variant {
    /// The Format enum value this variant reports itself as in emitted measurements.
    /// Uncompressed and BtrBlocks-default both surface as [`Format::OnDiskVortex`]; the
    /// TurboQuant variant gets its own [`Format::VortexTurboQuant`] so dashboards can
    /// distinguish them.
    pub fn as_format(&self) -> Format {
        match self {
            Variant::VortexUncompressed => Format::OnDiskVortex,
            Variant::VortexDefault => Format::OnDiskVortex,
            Variant::VortexTurboQuant => Format::VortexTurboQuant,
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
    /// Name used in metric strings — usually the dataset's `Dataset::name()`.
    pub name: String,
    /// Uncompressed `Vector<dim, f32>` array (canonical form). Doubles as the
    /// ground-truth basis for the correctness-verification pass and for TurboQuant's
    /// Recall@K quality measurement.
    pub uncompressed: ArrayRef,
    /// The query vector to use (a single row pulled from the dataset).
    pub query: Vec<f32>,
    /// Parquet file size on disk in bytes — produced by the dataset download step
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
/// `Vector<dim, f32>` extension array, and extracting a single-row query vector.
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

    let query = extract_query_row(&uncompressed, DEFAULT_QUERY_ROW)?;

    Ok(PreparedDataset {
        name: dataset.name().to_string(),
        uncompressed,
        query,
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

/// Pull a single row out of a `Vector<dim, f32>` extension array as a plain `Vec<f32>`.
///
/// Only `f32`-typed `Vector` arrays are supported today — the benchmark deliberately
/// restricts itself to `f32` vectors, so we assert the element type rather than
/// quietly returning a mis-cast slice.
pub(crate) fn extract_query_row(vector_ext: &ArrayRef, row: usize) -> Result<Vec<f32>> {
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
    if elements.ptype() != PType::F32 {
        bail!(
            "extract_query_row currently only supports f32 Vector columns, got {:?}",
            elements.ptype()
        );
    }
    let slice = elements.as_slice::<f32>();
    let start = row * dim_usize;
    Ok(slice[start..start + dim_usize].to_vec())
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
    /// Wall time spent constructing the variant tree from the already-materialized
    /// uncompressed source. 0 for [`Variant::VortexUncompressed`]; meaningful for the
    /// compressed variants.
    pub compress_duration: Duration,
}

/// Apply a `Variant`'s preparation strategy to the materialized uncompressed source and
/// return the resulting tree together with its reported in-memory size and construction
/// time. Uses the global [`vortex_bench::SESSION`] for any execution-context work; the
/// benchmark has no reason to support multiple concurrent sessions.
///
/// **Why nbytes instead of on-disk size?** The Vortex file writer applies BtrBlocks
/// compression as part of its default write strategy regardless of the in-memory tree
/// shape, so serializing an "uncompressed" tree and measuring the resulting `.vortex`
/// file produces the same bytes as serializing a `BtrBlocksCompressor::default()`-
/// compressed tree — the disk-size comparison collapses two conceptually different
/// things into one number. Reporting `nbytes()` of the in-memory tree keeps the size
/// measurement consistent with what the *compute* measurements operate on.
pub fn prepare_variant(prepared: &PreparedDataset, variant: Variant) -> Result<PreparedVariant> {
    match variant {
        Variant::VortexUncompressed => {
            // Identity: the uncompressed Extension<Vector> is already materialized. Still
            // record a dummy Instant so the timing point has a well-defined value even
            // if it's effectively zero.
            let start = Instant::now();
            let array = prepared.uncompressed.clone();
            let compress_duration = start.elapsed();
            let nbytes = array.nbytes();
            Ok(PreparedVariant {
                array,
                nbytes,
                compress_duration,
            })
        }
        Variant::VortexDefault => {
            let start = Instant::now();
            let array = BtrBlocksCompressor::default().compress(&prepared.uncompressed)?;
            let compress_duration = start.elapsed();
            let nbytes = array.nbytes();
            Ok(PreparedVariant {
                array,
                nbytes,
                compress_duration,
            })
        }
        Variant::VortexTurboQuant => {
            let mut ctx = SESSION.create_execution_ctx();
            let start = Instant::now();
            let array = compress_turboquant(prepared.uncompressed.clone(), &mut ctx)?;
            let compress_duration = start.elapsed();
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
pub fn run_timings(
    variant_array: &ArrayRef,
    query: &[f32],
    iterations: usize,
) -> Result<VariantTimings> {
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
            let matches: BoolArray =
                execute_filter(variant_array, query, DEFAULT_THRESHOLD, &mut ctx)?;
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

/// Execute `CosineSimilarity(data, broadcast(query))` to a materialized `f32`
/// [`PrimitiveArray`]. Shared between the timing loop and the correctness-verification
/// path so both exercise the exact same expression tree.
///
/// # Errors
///
/// Returns an error if `data` is not a [`vortex_tensor::vector::Vector`] extension array,
/// if `query`'s length doesn't match the database vector dimension, or if the execution
/// context rejects the expression.
pub fn execute_cosine(
    data: &ArrayRef,
    query: &[f32],
    ctx: &mut ExecutionCtx,
) -> Result<PrimitiveArray> {
    let num_rows = data.len();
    let query_vec = build_constant_query_vector(query, num_rows)?;
    let cosine = CosineSimilarity::try_new_array(data.clone(), query_vec, num_rows)?.into_array();
    Ok(cosine.execute(ctx)?)
}

fn execute_filter(
    data: &ArrayRef,
    query: &[f32],
    threshold: f32,
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

    /// Build a test `PreparedDataset` from synthetic data, pulling the query from
    /// row 0 via the shared `extract_query_row` helper so all tests exercise the
    /// ptype-assertion path the benchmark hot path uses.
    fn test_prepared(dim: u32, num_rows: usize, seed: u64) -> PreparedDataset {
        let uncompressed = synthetic_vector(dim, num_rows, seed);
        let query = extract_query_row(&uncompressed, 0).unwrap();
        PreparedDataset {
            name: "synthetic".to_string(),
            uncompressed,
            query,
            parquet_bytes: 0,
        }
    }

    #[test]
    fn extract_query_row_returns_the_right_slice() {
        let dim = 8u32;
        let num_rows = 4usize;
        let prepared = test_prepared(dim, num_rows, 0xDEADBEEF);

        // Row 0 extraction was already used to populate `prepared.query`; check it
        // agrees with a second extraction for row 0, and that row 3 (last) is
        // different (as it should be for distinct synthetic vectors).
        let row0 = extract_query_row(&prepared.uncompressed, 0).unwrap();
        let row3 = extract_query_row(&prepared.uncompressed, 3).unwrap();
        assert_eq!(row0, prepared.query);
        assert_eq!(row0.len(), dim as usize);
        assert_eq!(row3.len(), dim as usize);
        assert_ne!(row0, row3, "different rows must differ for this seed");
    }

    #[test]
    fn extract_query_row_rejects_out_of_bounds_row() {
        let dim = 8u32;
        let num_rows = 4usize;
        let prepared = test_prepared(dim, num_rows, 0xC0FFEE);

        let err = extract_query_row(&prepared.uncompressed, 4)
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

            let timings = run_timings(&prep.array, &prepared.query, 2).unwrap();
            // TurboQuant + default must do real work; uncompressed's decompress is a
            // no-op and can plausibly time as zero.
            assert!(timings.cosine > Duration::ZERO);
            assert!(timings.filter > Duration::ZERO);
        }
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

        let unc_timings = run_timings(&uncompressed_prep.array, &prepared.query, 3).unwrap();
        let tq_timings = run_timings(&turboquant_prep.array, &prepared.query, 3).unwrap();

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
