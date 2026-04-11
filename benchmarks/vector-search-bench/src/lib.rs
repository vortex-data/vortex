// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector similarity-search benchmark core.
//!
//! This crate measures four quantities for each `(dataset, variant)` pair:
//!
//! 1. **Compressed storage size** (bytes on disk, or in-memory `.nbytes()` for variants that
//!    don't yet serialize — currently just [`Variant::VortexTurboQuant`]).
//! 2. **Full-scan decode time** — executing the `Vector<dim, f32>` column into a
//!    materialized [`vortex::array::arrays::FixedSizeListArray`].
//! 3. **Cosine-similarity execute time** — executing
//!    `CosineSimilarity(data, const_query)` into a materialized f32 primitive array.
//! 4. **Filter execute time** — executing
//!    `Binary(Gt, [CosineSimilarity, threshold])` into a
//!    [`vortex::array::arrays::BoolArray`].
//!
//! Measurements are emitted via the existing `vortex_bench::measurements` types so that
//! the benchmark results flow through the standard `gh-json` pipeline and appear in the
//! CI dashboard alongside compress-bench / random-access-bench results.

use std::time::Duration;
use std::time::Instant;

pub mod parquet_baseline;
pub mod recall;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::ValueEnum;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::conversions::list_to_vector_ext;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::Dataset;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_btrblocks::BtrBlocksCompressor;
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

/// Number of rows in the query vector — matches the database so `ScalarFnArray`'s
/// equal-length contract is satisfied. This type alias exists to make the broadcast
/// semantics obvious at call sites.
type QueryLen = usize;

/// A materialized Vortex array and its associated execution session / context.
pub struct PreparedDataset {
    /// Name used in metric strings — usually the dataset's `Dataset::name()`.
    pub name: String,
    /// Uncompressed `Vector<dim, f32>` array (canonical form). This is reused as the
    /// ground-truth basis for TurboQuant recall checks in future commits.
    pub uncompressed: ArrayRef,
    /// The query vector to use (a single row pulled from the dataset).
    pub query: Vec<f32>,
    /// Parquet file size on disk in bytes — produced by the dataset download step.
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
            vortex::dtype::DType::Extension(ext) => ext.storage_dtype(),
            other => {
                vortex::error::vortex_panic!("expected Extension<Vector>, got {other}")
            }
        };
        match fsl_dtype {
            vortex::dtype::DType::FixedSizeList(_, dim, _) => *dim,
            other => {
                vortex::error::vortex_panic!("expected FixedSizeList storage, got {other}")
            }
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
    use vortex::array::arrays::ExtensionArray;

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
    use vortex::array::arrays::Chunked;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::Struct;
    use vortex::array::arrays::chunked::ChunkedArrayExt;
    use vortex::array::arrays::struct_::StructArrayExt as _;

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
fn extract_query_row(vector_ext: &ArrayRef, row: usize) -> Result<Vec<f32>> {
    use vortex::array::arrays::Extension;
    use vortex::array::arrays::extension::ExtensionArrayExt;
    use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;

    let mut ctx = SESSION.create_execution_ctx();

    let ext_view = vector_ext
        .as_opt::<Extension>()
        .context("prepared dataset must be a Vector extension array")?;

    // Execute storage array to its canonical FSL form.
    let fsl: FixedSizeListArray = ext_view.storage_array().clone().execute(&mut ctx)?;

    let dim_usize = {
        let vortex::dtype::DType::FixedSizeList(_, d, _) = fsl.dtype() else {
            bail!("storage dtype must be FixedSizeList");
        };
        *d as usize
    };

    if row * dim_usize + dim_usize > vector_ext.len() * dim_usize {
        bail!(
            "query row {row} out of bounds for dataset of length {}",
            vector_ext.len()
        );
    }

    let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
    let slice = elements.as_slice::<f32>();
    let start = row * dim_usize;
    Ok(slice[start..start + dim_usize].to_vec())
}

/// Apply a `Variant`'s preparation strategy to the uncompressed Vortex array and return the
/// prepared array together with its reported size in bytes. For serializable variants the
/// size is the number of bytes written to a `.vortex` file; for in-memory-only variants
/// (TurboQuant) it's the live `.nbytes()` footprint.
pub async fn prepare_variant(
    prepared: &PreparedDataset,
    variant: Variant,
    session: &VortexSession,
) -> Result<(ArrayRef, u64)> {
    match variant {
        Variant::VortexUncompressed => {
            let array = prepared.uncompressed.clone();
            let size =
                measure_on_disk_size(&array, session, &prepared.name, "uncompressed").await?;
            Ok((array, size))
        }
        Variant::VortexDefault => {
            let array = BtrBlocksCompressor::default().compress(&prepared.uncompressed)?;
            let size = measure_on_disk_size(&array, session, &prepared.name, "default").await?;
            Ok((array, size))
        }
        Variant::VortexTurboQuant => {
            let mut ctx = session.create_execution_ctx();
            let array = compress_turboquant(prepared.uncompressed.clone(), &mut ctx)?;
            // TurboQuant cannot yet round-trip through a Vortex file (L2Denorm metadata
            // serialization is not implemented). Report the in-memory `.nbytes()` footprint
            // as a proxy. Document this in the benchmark output so consumers of the
            // dashboard aren't misled.
            let size = array.nbytes() as u64;
            Ok((array, size))
        }
    }
}

/// Serialize a prepared Vortex array to a temporary `.vortex` file and return its length.
/// This is what we report as the "compressed size" for serializable variants; it matches
/// the semantics of `compress-bench` which reports the on-disk parquet/vortex file size.
async fn measure_on_disk_size(
    array: &ArrayRef,
    session: &VortexSession,
    dataset_name: &str,
    variant_label: &str,
) -> Result<u64> {
    use vortex::file::WriteOptionsSessionExt;

    let tmp_dir = std::env::temp_dir().join("vortex-vector-search-bench");
    tokio::fs::create_dir_all(&tmp_dir).await?;
    let tmp_path = tmp_dir.join(format!("{dataset_name}-{variant_label}.vortex"));

    let mut file = tokio::fs::File::create(&tmp_path).await?;
    session
        .write_options()
        .write(&mut file, array.clone().to_array_stream())
        .await?;

    let metadata = tokio::fs::metadata(&tmp_path).await?;
    Ok(metadata.len())
}

/// Run the decode / cosine / filter microbenchmarks against a prepared variant array and
/// return the best-of-`iterations` wall times for each measurement.
pub fn run_timings(
    variant_array: &ArrayRef,
    query: &[f32],
    iterations: usize,
    session: &VortexSession,
) -> Result<VariantTimings> {
    let _ = QueryLen::default; // touch the type alias so rustc doesn't warn

    let mut decode = Duration::MAX;
    let mut cosine = Duration::MAX;
    let mut filter = Duration::MAX;

    for _ in 0..iterations {
        let mut ctx = session.create_execution_ctx();
        let start = Instant::now();
        let decoded: FixedSizeListArray = decode_full_scan(variant_array, &mut ctx)?;
        decode = decode.min(start.elapsed());
        drop(decoded);
    }

    for _ in 0..iterations {
        let mut ctx = session.create_execution_ctx();
        let start = Instant::now();
        let scores: PrimitiveArray = execute_cosine(variant_array, query, &mut ctx)?;
        cosine = cosine.min(start.elapsed());
        drop(scores);
    }

    for _ in 0..iterations {
        let mut ctx = session.create_execution_ctx();
        let start = Instant::now();
        let matches: BoolArray = execute_filter(variant_array, query, DEFAULT_THRESHOLD, &mut ctx)?;
        filter = filter.min(start.elapsed());
        drop(matches);
    }

    Ok(VariantTimings {
        decode,
        cosine,
        filter,
    })
}

/// Timing summary for one `(dataset, variant)` pair.
#[derive(Debug, Clone, Copy)]
pub struct VariantTimings {
    /// Wall time for a full column decode.
    pub decode: Duration,
    /// Wall time for the cosine_similarity scalar-function execution.
    pub cosine: Duration,
    /// Wall time for the full `Binary(Gt, [cosine, threshold])` expression.
    pub filter: Duration,
}

/// Fully materialize the input column so the measurement captures *all* decompression
/// work — the extension shell, the FSL storage, and the inner element buffer.
///
/// For the Vortex-uncompressed variant this is cheap (bitwise copy / no-op). For
/// BtrBlocks-default it includes FSL decompression. For TurboQuant it includes running
/// the inverse SORF rotation + dictionary lookup through the scalar-fn pipeline.
fn decode_full_scan(
    array: &ArrayRef,
    ctx: &mut vortex::array::ExecutionCtx,
) -> Result<FixedSizeListArray> {
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::extension::ExtensionArrayExt;

    let ext: ExtensionArray = array.clone().execute(ctx)?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    Ok(fsl)
}

fn execute_cosine(
    data: &ArrayRef,
    query: &[f32],
    ctx: &mut vortex::array::ExecutionCtx,
) -> Result<PrimitiveArray> {
    use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
    use vortex_tensor::vector_search::build_constant_query_vector;

    let num_rows = data.len();
    let query_vec = build_constant_query_vector(query, num_rows)?;
    let cosine = CosineSimilarity::try_new_array(data.clone(), query_vec, num_rows)
        .vortex_expect("cosine similarity accepts matching Vector inputs")
        .into_array();
    Ok(cosine.execute(ctx)?)
}

fn execute_filter(
    data: &ArrayRef,
    query: &[f32],
    threshold: f32,
    ctx: &mut vortex::array::ExecutionCtx,
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
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::extension::ExtensionArrayExt;
    use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
    use vortex_bench::SESSION;

    use super::test_utils::synthetic_vector;
    use super::*;

    #[test]
    fn prepare_variant_produces_non_empty_array_for_all_variants() {
        let dim = 128u32;
        let num_rows = 64usize;
        let uncompressed = synthetic_vector(dim, num_rows, 0xC0FFEE);

        let ext = uncompressed
            .as_opt::<vortex::array::arrays::Extension>()
            .unwrap();
        let mut ctx = SESSION.create_execution_ctx();
        let fsl: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx).unwrap();
        let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx).unwrap();
        let slice = elements.as_slice::<f32>();
        let query = slice[..dim as usize].to_vec();

        let prepared = PreparedDataset {
            name: "synthetic".to_string(),
            uncompressed: uncompressed.clone(),
            query,
            parquet_bytes: 0,
        };

        for variant in [
            Variant::VortexUncompressed,
            Variant::VortexDefault,
            Variant::VortexTurboQuant,
        ] {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (array, size) = rt
                .block_on(prepare_variant(&prepared, variant, &SESSION))
                .unwrap();
            assert_eq!(
                array.len(),
                num_rows,
                "variant {variant:?} changed row count"
            );
            assert!(size > 0, "variant {variant:?} reported zero size");

            let timings = run_timings(&array, &prepared.query, 2, &SESSION).unwrap();
            assert!(timings.decode > Duration::ZERO);
            assert!(timings.cosine > Duration::ZERO);
            assert!(timings.filter > Duration::ZERO);
        }
    }
}
