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
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_tensor::vector::Vector;
pub use vortex_tensor::vector_search::build_similarity_search_tree;
pub use vortex_tensor::vector_search::compress_turboquant;

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

/// Compresses a raw `Vector<dim, f32>` array with the default BtrBlocks pipeline.
///
/// [`BtrBlocksCompressor`] walks into the extension array and recursively compresses the
/// underlying FSL storage child. TurboQuant is *not* exercised by this path -- it is not
/// registered in the default scheme set -- so this measures "generic" lossless compression
/// applied to float vectors.
///
/// Stays in this bench-only module because `BtrBlocksCompressor` is a dev-dependency of
/// `vortex-tensor`, so promoting it to the public `vector_search` module would drag the
/// `vortex-btrblocks` dep into `vortex-tensor`'s main dependency list.
pub fn compress_default(data: ArrayRef) -> VortexResult<ArrayRef> {
    BtrBlocksCompressor::default().compress(&data)
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
