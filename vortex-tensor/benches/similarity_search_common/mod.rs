// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for the similarity-search benchmark and example.
//!
//! This module is included from both `vortex-tensor/benches/similarity_search.rs` and
//! `vortex-tensor/examples/similarity_search.rs` via an explicit `#[path = ...]` so both targets
//! use the exact same array-tree builder.
//!
//! The three main entry points are:
//!
//! - [`generate_random_vectors`] to build a deterministic random [`Vector`] extension array.
//! - [`build_variant`] to take a raw vector array and apply the requested compression strategy
//!   (uncompressed, default BtrBlocks, or TurboQuant).
//! - [`build_similarity_search_tree`] to wire a cosine-similarity + threshold expression on top of
//!   a prepared data array and a single-row query vector.
//!
//! [`Vector`]: vortex_tensor::vector::Vector

#![allow(dead_code)]

use std::fmt;
use std::sync::LazyLock;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_tensor::encodings::turboquant::TurboQuantConfig;
use vortex_tensor::encodings::turboquant::turboquant_encode_unchecked;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use vortex_tensor::vector::Vector;

/// A shared [`VortexSession`] pre-loaded with the builtin [`ArraySession`] so both bench and
/// example can create execution contexts cheaply.
pub static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// The three compression strategies the benchmark and example exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Raw `Vector<dim, f32>` with no compression applied.
    Uncompressed,
    /// `BtrBlocksCompressor::default()` walks into the extension array and compresses the
    /// underlying FSL storage child with the default scheme set (no TurboQuant).
    DefaultCompression,
    /// TurboQuant: normalize, quantize to `FSL(Dict)`, wrap in SORF + `L2Denorm`.
    TurboQuant,
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uncompressed => f.write_str("Uncompressed"),
            Self::DefaultCompression => f.write_str("DefaultCompression"),
            Self::TurboQuant => f.write_str("TurboQuant"),
        }
    }
}

/// Generate `num_rows` random f32 vectors of dimension `dim`, wrapped in a [`Vector`] extension
/// array. The values are drawn from a standard normal distribution seeded by `seed` so results
/// are reproducible across runs.
///
/// [`Vector`]: vortex_tensor::vector::Vector
pub fn generate_random_vectors(num_rows: usize, dim: u32, seed: u64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(seed);
    // `Normal::new(0, 1)` is infallible for these parameters. `rand_distr::NormalError` does
    // not implement `Into<VortexError>`, so we cannot use `vortex_expect` here; fall back to
    // `vortex_panic!` on the (impossible) error path instead.
    let normal =
        Normal::new(0.0f32, 1.0).unwrap_or_else(|_| vortex_panic!("Normal(0, 1) is well-defined"));

    let dim_usize = dim as usize;
    let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim_usize);
    for _ in 0..(num_rows * dim_usize) {
        buf.push(normal.sample(&mut rng));
    }

    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl =
        FixedSizeListArray::try_new(elements.into_array(), dim, Validity::NonNullable, num_rows)
            .vortex_expect("FSL with valid shape and matching children length");

    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
        .vortex_expect("Vector extension dtype is valid for an f32 FSL")
        .erased();
    ExtensionArray::new(ext_dtype, fsl.into_array()).into_array()
}

/// Pull the `row`-th vector out of a `Vector<dim, f32>` extension array as a plain `Vec<f32>`.
///
/// Used to extract a single query vector from a batch of generated data. The input must already
/// be fully materialized (no lazy scalar-fn wrappers); pass a raw array from
/// [`generate_random_vectors`], not a compressed variant.
pub fn extract_row_as_query(vectors: &ArrayRef, row: usize, dim: u32) -> Vec<f32> {
    let ext = vectors
        .as_opt::<Extension>()
        .vortex_expect("data must be a Vector extension array");

    let mut ctx = SESSION.create_execution_ctx();
    let fsl: FixedSizeListArray = ext
        .storage_array()
        .clone()
        .execute(&mut ctx)
        .vortex_expect("storage array executes to an FSL");
    let elements: PrimitiveArray = fsl
        .elements()
        .clone()
        .execute(&mut ctx)
        .vortex_expect("FSL elements execute to a PrimitiveArray");

    let slice = elements.as_slice::<f32>();
    let dim_usize = dim as usize;
    let start = row * dim_usize;
    slice[start..start + dim_usize].to_vec()
}

/// Build a `Vector<dim, f32>` extension array whose storage is a [`ConstantArray`] broadcasting a
/// single query vector across `num_rows` rows. This is how we hand a single query vector to
/// `CosineSimilarity` on the `rhs` side -- `ScalarFnArray` requires both children to have the
/// same length, so we broadcast the query instead of hand-rolling a 1-row input.
fn build_constant_query_vector(query: &[f32], num_rows: usize) -> VortexResult<ArrayRef> {
    let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);

    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();
    let storage_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);

    let storage = ConstantArray::new(storage_scalar, num_rows).into_array();

    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, storage).into_array())
}

/// Compresses a raw `Vector<dim, f32>` array with the default BtrBlocks pipeline.
///
/// [`BtrBlocksCompressor`] walks into the extension array and recursively compresses the
/// underlying FSL storage child. TurboQuant is *not* exercised by this path -- it is not
/// registered in the default scheme set -- so this measures "generic" lossless compression
/// applied to float vectors.
pub fn compress_default(data: ArrayRef) -> VortexResult<ArrayRef> {
    BtrBlocksCompressor::default().compress(&data)
}

/// Compresses a raw `Vector<dim, f32>` array with the TurboQuant pipeline by hand, producing the
/// same tree shape that
/// [`vortex_tensor::encodings::turboquant::TurboQuantScheme`] would:
///
/// ```text
/// L2Denorm(SorfTransform(FSL(Dict(codes, centroids))), norms)
/// ```
///
/// Calling the encode helpers directly (instead of going through
/// `BtrBlocksCompressorBuilder::with_turboquant()`) lets this example avoid depending on the
/// `unstable_encodings` feature flag.
///
/// See `vortex-tensor/src/encodings/turboquant/tests/mod.rs::normalize_and_encode` for the same
/// canonical recipe.
pub fn compress_turboquant(data: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let l2_denorm = normalize_as_l2_denorm(data, ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();
    let num_rows = l2_denorm.len();

    let normalized_ext = normalized
        .as_opt::<Extension>()
        .vortex_expect("normalized child should be an Extension array");

    let config = TurboQuantConfig::default();
    // SAFETY: `normalize_as_l2_denorm` guarantees every row is unit-norm (or zero), which is the
    // invariant `turboquant_encode_unchecked` expects.
    let tq = unsafe { turboquant_encode_unchecked(normalized_ext, &config, ctx) }?;

    Ok(unsafe { L2Denorm::new_array_unchecked(tq, norms, num_rows) }?.into_array())
}

/// Dispatch helper that builds the data array for the requested [`Variant`], starting from a
/// single random-vector generation. Always returns an `ArrayRef` whose logical dtype is
/// `Vector<dim, f32>`.
pub fn build_variant(
    variant: Variant,
    num_rows: usize,
    dim: u32,
    seed: u64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let raw = generate_random_vectors(num_rows, dim, seed);
    match variant {
        Variant::Uncompressed => Ok(raw),
        Variant::DefaultCompression => compress_default(raw),
        Variant::TurboQuant => compress_turboquant(raw, ctx),
    }
}

/// Build the lazy similarity-search array tree for a prepared data array and a single query
/// vector. The returned tree is a boolean array of length `data.len()` where position `i` is
/// `true` iff `cosine_similarity(data[i], query) > threshold`.
///
/// The tree shape is:
///
/// ```text
/// Binary(Gt, [
///     CosineSimilarity([data, ConstantArray(query_vec, n)]),
///     ConstantArray(threshold, n),
/// ])
/// ```
///
/// This function does no execution; it is safe to call inside a benchmark setup closure.
pub fn build_similarity_search_tree(
    data: ArrayRef,
    query: &[f32],
    threshold: f32,
) -> VortexResult<ArrayRef> {
    let num_rows = data.len();
    let query_vec = build_constant_query_vector(query, num_rows)?;

    let cosine = CosineSimilarity::try_new_array(data, query_vec, num_rows)?.into_array();

    let threshold_scalar = Scalar::primitive(threshold, Nullability::NonNullable);
    let threshold_array = ConstantArray::new(threshold_scalar, num_rows).into_array();

    cosine.binary(threshold_array, Operator::Gt)
}
