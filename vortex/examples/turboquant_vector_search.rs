// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector-search roundtrip on a vector-embedding dataset.
//!
//! Load a parquet dataset (cohere-small), wrap the `emb` column as a `Vector<f32, DIM>`
//! extension, compress with BtrBlocks + TurboQuant, write to an in-memory Vortex file, then read
//! the file back twice:
//!
//!   1. plain scan — decode to canonical `FixedSizeList<f32, DIM>` and verify the per-element diff
//!      against the original. TurboQuant is lossy, so we only check the reconstructed values are
//!      within a tolerance.
//!   2. scan with a pushed-down cosine-similarity filter `cosine_similarity(emb, query) > thresh`.
//!      The `CosineSimilarity` scalar fn is expressed directly as a filter `Expression`, so row
//!      selection happens inside the scan rather than after materialization.
//!
//! The parquet file is cached under `vortex-bench/data/<dataset>/` after the first download. Run
//! with:
//!
//! ```sh
//! cargo run --example turboquant_vector_search \
//!     -p vortex --features unstable_encodings --release
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use futures::TryStreamExt;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::EmptyMetadata;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::expr::col;
use vortex::array::expr::gt;
use vortex::array::expr::lit;
use vortex::array::scalar::Scalar;
use vortex::array::scalar_fn::EmptyOptions;
use vortex::array::scalar_fn::ScalarFnVTable;
use vortex::array::scalar_fn::ScalarFnVTableExt;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::file::ALLOWED_ENCODINGS;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_array::ExecutionCtx;
use vortex_array::builtins::ArrayBuiltins;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::vector_dataset;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_bench::vector_dataset::list_to_vector_ext;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_error::VortexExpect;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::sorf_transform::SorfTransform;
use vortex_tensor::vector::AnyVector;
use vortex_tensor::vector::Vector;

/// Cosine threshold for the demo filter. The query comes from the test split, so it may or may not
/// have nearby rows in the train split.
const COSINE_THRESHOLD: f32 = 0.85;

/// Slack for checking decoded rows against a predicate that was evaluated on TurboQuant's lossy
/// readthrough representation.
const COSINE_THRESHOLD_TOL: f32 = 0.02;

/// Regression ceiling on the decoded vs original max-abs-diff for 8-bit TurboQuant on 768-dim f32
/// embeddings. Observed on cohere-small: ~0.10. Pinned with slack so the check catches large
/// quality regressions without flapping on normal run-to-run variation.
const MAX_ABS_DIFF_TOL: f32 = 0.2;

#[tokio::main]
async fn main() -> Result<()> {
    // Opt in to registering the tensor scalar-fn array plugins before building the session.
    // Without this, the TurboQuant-compressed `emb` column cannot be serialized into the Vortex
    // file or deserialized on read.
    //
    // SAFETY: single-threaded setup before any other thread exists.
    unsafe { std::env::set_var(vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV, "1") };

    let session = VortexSession::default().with_tokio();
    vortex_tensor::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    println!("session initialized with tensor plugins");

    let dataset = VectorDataset::CohereSmall100k; // This is one of the smaller datasets.

    // Download the source parquet files.
    let dataset_paths = vector_dataset::download(dataset, TrainLayout::Single).await?;
    let (_id, query_vector) = get_query_vector(dataset_paths.test, &mut ctx).await?;
    println!(
        "query vector selected (id = {_id}, dim = {})",
        query_vector.len()
    );

    // Bring the parquet file into memory so that we can write it as a vortex file (after prep).
    let single_train_file = dataset_paths
        .train_files
        .first()
        .vortex_expect("we know that there must be a file here")
        .clone();

    println!("reading parquet into chunked array...");
    let chunked_table = parquet_to_vortex_chunks(single_train_file)
        .await?
        .into_array();
    let len = chunked_table.len();
    println!("parquet loaded: {len} rows");

    let id = chunked_table.get_item("id")?;
    let emb = chunked_table.get_item("emb")?;

    println!("converting emb column to Vector extension type...");
    let vector_array = list_to_vector_ext(emb)?;

    let fields = [("id", id), ("emb", vector_array)];
    let struct_array = StructArray::from_fields(&fields)?.into_array();

    println!("compressing with TurboQuant and writing to in-memory Vortex file...");
    let bytes = write_turboquant(&session, struct_array.clone().into_array()).await?;
    println!("vortex file written: {} bytes", bytes.len());

    println!("verifying roundtrip fidelity...");
    verify_roundtrip(&session, &bytes, struct_array.clone()).await?;

    println!("verifying filter pushdown with cosine similarity...");
    verify_filter_pushdown(&session, &bytes, &query_vector, struct_array, &mut ctx).await?;

    println!("all checks passed!");
    Ok(())
}

async fn get_query_vector(
    query_vectors_path: PathBuf,
    ctx: &mut ExecutionCtx,
) -> Result<(usize, Vec<f32>)> {
    let test_vectors = parquet_to_vortex_chunks(query_vectors_path).await?;

    // Get a random query vector.
    let idx = rand::random_range(0..test_vectors.len());
    let struct_scalar = test_vectors.execute_scalar(idx, ctx)?;
    let id_scalar = struct_scalar
        .as_struct()
        .field("id")
        .vortex_expect("test parquet file missing `id` field");

    ensure!(
        id_scalar
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("id was not a i64")
            == idx as i64
    );

    let emb_scalar = struct_scalar
        .as_struct()
        .field("emb")
        .vortex_expect("test parquet file missing `emb` field");

    // Pack into a `Vec<f32>`.
    let query_vector: Vec<f32> = emb_scalar
        .as_list()
        .elements()
        .vortex_expect("somehow had a null test vector")
        .iter()
        .map(|element| {
            element
                .as_primitive()
                .as_::<f32>()
                .vortex_expect("value was not a f32")
        })
        .collect();

    Ok((idx, query_vector))
}

async fn write_turboquant(session: &VortexSession, array: ArrayRef) -> Result<ByteBuffer> {
    let compressor = BtrBlocksCompressorBuilder::default()
        .with_turboquant()
        .build();

    // TurboQuant produces `L2Denorm(SorfTransform(FSL(Dict(...))), norms)`. The default write
    // allow-list only covers canonical/compressed array encodings, so the tensor scalar-fn
    // encodings it emits get rejected during normalization. Extend the set with the two encoding
    // IDs this scheme actually uses.
    let mut allowed = ALLOWED_ENCODINGS.clone();
    allowed.insert(L2Denorm.id());
    allowed.insert(SorfTransform.id());

    let strategy = WriteStrategyBuilder::default()
        .with_compressor(compressor)
        .with_allow_encodings(allowed)
        .build();

    let mut buf = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut buf, array.to_array_stream())
        .await?;
    Ok(buf.freeze())
}

async fn verify_roundtrip(
    session: &VortexSession,
    bytes: &ByteBuffer,
    original: ArrayRef,
) -> Result<()> {
    let chunks: Vec<ArrayRef> = session
        .open_options()
        .open_buffer(bytes.clone())?
        .scan()?
        .into_array_stream()?
        .try_collect()
        .await?;

    let mut ctx = session.create_execution_ctx();

    let read: StructArray = ChunkedArray::try_new(chunks, original.dtype().clone())?
        .into_array()
        .execute(&mut ctx)?;
    let original: StructArray = original.execute(&mut ctx)?;
    ensure!(read.len() == original.len());

    let read_emb = read.unmasked_field_by_name("emb")?.clone();
    let original_emb = original.unmasked_field_by_name("emb")?.clone();

    let decoded = flatten_vector_column(read_emb, &mut ctx)?;
    let original_decoded = flatten_vector_column(original_emb, &mut ctx)?;

    let (max_abs, mean_abs) = diff_stats(&original_decoded, &decoded);
    println!(
        "roundtrip fidelity: max_abs_diff = {max_abs:.6}, mean_abs_diff = {mean_abs:.6} \
         (tol = {MAX_ABS_DIFF_TOL})"
    );
    if max_abs > MAX_ABS_DIFF_TOL {
        bail!("TurboQuant max_abs_diff {max_abs} exceeds tolerance {MAX_ABS_DIFF_TOL}");
    }

    Ok(())
}

async fn verify_filter_pushdown(
    session: &VortexSession,
    bytes: &ByteBuffer,
    query: &[f32],
    original: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Result<()> {
    // Build the filter as `cosine_similarity(emb, <query>) > threshold`. The RHS of
    // `CosineSimilarity` is a `lit(...)` wrapping a `Vector<f32, DIM>` scalar; during scan
    // evaluation the Literal expands to a ConstantArray whose row count matches the current batch,
    // satisfying `CosineSimilarity`'s same-length requirement. The entire expression is pushed
    // through `with_filter`, so row selection happens inside the scan rather than after the whole
    // column is materialized.
    println!("query: {}", preview_vector(query));

    let query_scalar = build_query_vector_scalar(query)?;
    let cosine_expr = CosineSimilarity.new_expr(EmptyOptions, [col("emb"), lit(query_scalar)]);
    let filter = gt(cosine_expr, lit(COSINE_THRESHOLD));

    let scan_start = Instant::now();
    let chunks: Vec<ArrayRef> = session
        .open_options()
        .open_buffer(bytes.clone())?
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .try_collect()
        .await?;
    let scan_ms = scan_start.elapsed().as_secs_f64() * 1e3;

    let hits: usize = chunks.iter().map(|c| c.len()).sum();
    println!(
        "pushed down `cosine_similarity(emb, query) > {COSINE_THRESHOLD}`: {hits} rows survived \
         in {scan_ms:.2} ms"
    );
    if hits == 0 {
        println!("  no rows survived the filter for this random query");
        return Ok(());
    }

    // Materialize the matching rows and dump each `emb` vector so the reader can see what the
    // pushed-down filter actually selected. Vectors are truncated to the first few elements since
    // DIM is typically large.
    let filtered: StructArray = ChunkedArray::try_new(chunks, original.dtype().clone())?
        .into_array()
        .execute(ctx)?;

    let emb = filtered.unmasked_field_by_name("emb")?.clone();
    let flat = flatten_vector_column(emb, ctx)?;

    let dim = query.len();
    for (i, row) in flat.chunks_exact(dim).enumerate() {
        let cos = cosine_similarity(query, row);
        ensure!(
            cos >= COSINE_THRESHOLD - COSINE_THRESHOLD_TOL,
            "filtered row {i} had decoded cosine {cos:+.6}, below threshold {COSINE_THRESHOLD} \
             by more than tolerance {COSINE_THRESHOLD_TOL}"
        );
        println!("  match {i}: cos = {cos:+.6} {}", preview_vector(row));
    }

    Ok(())
}

/// Plain `dot(a, b) / (||a|| * ||b||)` over two equal-length f32 slices. Used purely for reporting
/// — the actual row selection is done inside the scan by the pushed-down `CosineSimilarity`
/// expression. This lets the reader cross-check that the surviving rows really do clear the
/// threshold once decoded.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut a_sq = 0.0f32;
    let mut b_sq = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        a_sq += x * x;
        b_sq += y * y;
    }
    dot / (a_sq.sqrt() * b_sq.sqrt())
}

/// Render a vector as `[v0, v1, ..., vN-1, vN]` with the first 4 and last 1 elements at 4-decimal
/// precision. Keeps the output compact for high-dim embeddings while still giving the reader
/// something concrete to eyeball.
fn preview_vector(row: &[f32]) -> String {
    let dim = row.len();
    if dim <= 5 {
        return format!("[{}] (dim = {dim})", fmt_slice(row));
    }
    format!(
        "[{}, ..., {}] (dim = {dim})",
        fmt_slice(&row[..4]),
        fmt_slice(&row[dim - 1..])
    )
}

fn fmt_slice(s: &[f32]) -> String {
    s.iter()
        .map(|v| format!("{v:+.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Wrap a query vector in a `Vector<f32, len>` extension scalar suitable for use as the RHS of a
/// `CosineSimilarity` filter expression via `lit(...)`.
fn build_query_vector_scalar(query: &[f32]) -> Result<Scalar> {
    let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();
    let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    Ok(Scalar::extension::<Vector>(EmptyMetadata, fsl_scalar))
}

/// Decode a `Vector<f32, _>` extension array's storage down to its flat f32 buffer.
fn flatten_vector_column(emb: ArrayRef, ctx: &mut ExecutionCtx) -> Result<Vec<f32>> {
    let ext: ExtensionArray = emb.execute(ctx)?;
    ensure!(ext.ext_dtype().is::<AnyVector>());

    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    Ok(elements.as_slice::<f32>().to_vec())
}

fn diff_stats(original: &[f32], decoded: &[f32]) -> (f32, f32) {
    assert_eq!(original.len(), decoded.len());
    let (sum_abs, max_abs) =
        original
            .iter()
            .zip(decoded)
            .fold((0.0f32, 0.0f32), |(sum, peak), (&orig, &dec)| {
                let diff = (orig - dec).abs();
                (sum + diff, peak.max(diff))
            });
    let mean_abs = sum_abs / original.len() as f32;
    (max_abs, mean_abs)
}
