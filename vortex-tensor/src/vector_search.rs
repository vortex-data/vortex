// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reusable helpers for building brute-force vector similarity search expressions over
//! [`Vector`] extension arrays.
//!
//! This module exposes three small building blocks that together make it straightforward to
//! stand up a cosine-similarity-plus-threshold scan on top of a prepared data array:
//!
//! - [`compress_turboquant`] applies the canonical TurboQuant encoding pipeline
//!   (`L2Denorm(SorfTransform(FSL(Dict(codes, centroids))), norms)`) to a raw
//!   `Vector<dim, f32>` array without requiring the caller to plumb the
//!   `unstable_encodings` feature flag on the `vortex` facade.
//! - [`build_constant_query_vector`] wraps a single query vector into a
//!   [`Vector`] extension array whose storage is a [`ConstantArray`] broadcast
//!   across `num_rows` rows. This is the shape expected by
//!   [`CosineSimilarity::try_new_array`] for the RHS of a database-vs-query scan.
//! - [`build_similarity_search_tree`] wires everything together into a lazy
//!   `Binary(Gt, [CosineSimilarity(data, query), threshold])` expression.
//!
//! Executing the tree from [`build_similarity_search_tree`] into a
//! [`BoolArray`](vortex_array::arrays::BoolArray) yields one boolean per row indicating whether
//! that row's cosine similarity to the query exceeds `threshold`.
//!
//! # Example
//!
//! ```ignore
//! use vortex_array::{ArrayRef, VortexSessionExecute};
//! use vortex_array::arrays::BoolArray;
//! use vortex_session::VortexSession;
//! use vortex_tensor::vector_search::{build_similarity_search_tree, compress_turboquant};
//!
//! fn run(session: &VortexSession, data: ArrayRef, query: &[f32]) -> anyhow::Result<()> {
//!     let mut ctx = session.create_execution_ctx();
//!     let data = compress_turboquant(data, &mut ctx)?;
//!     let tree = build_similarity_search_tree(data, query, 0.8)?;
//!     let _matches: BoolArray = tree.execute(&mut ctx)?;
//!     Ok(())
//! }
//! ```
//!
//! [`Vector`]: crate::vector::Vector
//! [`CosineSimilarity::try_new_array`]: crate::scalar_fns::cosine_similarity::CosineSimilarity::try_new_array

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::encodings::turboquant::TurboQuantConfig;
use crate::encodings::turboquant::turboquant_encode_unchecked;
use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use crate::vector::Vector;

/// Apply the canonical TurboQuant encoding pipeline to a `Vector<dim, f32>` array.
///
/// The returned array has the shape
/// `L2Denorm(SorfTransform(FSL(Dict(codes, centroids))), norms)` — exactly what
/// [`crate::encodings::turboquant::TurboQuantScheme`] produces when invoked through
/// `BtrBlocksCompressorBuilder::with_turboquant()`, but without requiring callers to enable
/// the `unstable_encodings` feature on the `vortex` facade.
///
/// The input `data` must be a [`Vector`] extension array whose element type is `f32` and whose
/// dimensionality is at least
/// [`turboquant::MIN_DIMENSION`](crate::encodings::turboquant::MIN_DIMENSION). The TurboQuant
/// configuration used is [`TurboQuantConfig::default()`] (8-bit codes, 3 SORF rounds, seed 42).
///
/// # Errors
///
/// Returns an error if `data` is not a [`Vector`] extension array, if normalization fails, or
/// if the underlying TurboQuant encoder rejects the input shape.
pub fn compress_turboquant(data: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let l2_denorm = normalize_as_l2_denorm(data, ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();
    let num_rows = l2_denorm.len();

    let Some(normalized_ext) = normalized.as_opt::<Extension>() else {
        vortex_bail!("normalize_as_l2_denorm must produce an Extension array child");
    };

    let config = TurboQuantConfig::default();
    // SAFETY: `normalize_as_l2_denorm` guarantees every row is unit-norm (or zero), which is
    // the invariant `turboquant_encode_unchecked` expects.
    let tq = unsafe { turboquant_encode_unchecked(normalized_ext, &config, ctx) }?;

    Ok(unsafe { L2Denorm::new_array_unchecked(tq, norms, num_rows) }?.into_array())
}

/// Build a [`Vector`] extension array whose storage is a [`ConstantArray`] broadcasting a single
/// query vector across `num_rows` rows.
///
/// The element type is inferred from `T` (e.g. `f32` or `f64`). This is the shape expected for
/// the RHS of a database-vs-query [`CosineSimilarity`] scan: the `ScalarFnArray` contract
/// requires both children to have the same length, so rather than hand-rolling a 1-row input we
/// broadcast the query across the whole database.
///
/// # Errors
///
/// Returns an error if the [`Vector`] extension dtype rejects the constructed storage dtype.
pub fn build_constant_query_vector<T: NativePType + Into<PValue>>(
    query: &[T],
    num_rows: usize,
) -> VortexResult<ArrayRef> {
    let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);

    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();
    let storage_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);

    let storage = ConstantArray::new(storage_scalar, num_rows).into_array();

    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, storage).into_array())
}

/// Build the lazy similarity-search expression tree for a prepared database array and a
/// single query vector.
///
/// The returned array is a lazy boolean expression of length `data.len()` whose position `i`
/// is `true` iff `cosine_similarity(data[i], query) > threshold`. Executing it into a
/// [`BoolArray`](vortex_array::arrays::BoolArray) runs the full scan.
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
/// The element type is inferred from `T` and must match the element type of `data`'s
/// [`Vector`] extension dtype.
///
/// This function performs no execution; it is safe to call inside a benchmark setup closure.
///
/// # Errors
///
/// Returns an error if `query` has a length incompatible with `data`'s vector dimension, or
/// if any of the intermediate array constructors fails.
pub fn build_similarity_search_tree<T: NativePType + Into<PValue>>(
    data: ArrayRef,
    query: &[T],
    threshold: T,
) -> VortexResult<ArrayRef> {
    let num_rows = data.len();
    let query_vec = build_constant_query_vector(query, num_rows)?;

    let cosine = CosineSimilarity::try_new_array(data, query_vec, num_rows)?.into_array();

    let threshold_scalar = Scalar::primitive(threshold, Nullability::NonNullable);
    let threshold_array = ConstantArray::new(threshold_scalar, num_rows).into_array();

    cosine.binary(threshold_array, Operator::Gt)
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Extension;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::build_constant_query_vector;
    use super::build_similarity_search_tree;
    use super::compress_turboquant;
    use crate::vector::Vector;

    /// Build a `Vector<DIM, f32>` extension array from a flat f32 slice. Each contiguous
    /// group of `DIM` values becomes one row.
    fn vector_array(dim: u32, values: &[f32]) -> VortexResult<ArrayRef> {
        let dim_usize = dim as usize;
        assert_eq!(values.len() % dim_usize, 0);
        let num_rows = values.len() / dim_usize;

        let mut buf = BufferMut::<f32>::with_capacity(values.len());
        for &v in values {
            buf.push(v);
        }
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            dim,
            Validity::NonNullable,
            num_rows,
        )?;

        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    fn test_session() -> VortexSession {
        VortexSession::empty().with::<ArraySession>()
    }

    #[test]
    fn constant_query_vector_has_vector_extension_dtype() -> VortexResult<()> {
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let rhs = build_constant_query_vector(&query, 5)?;

        assert_eq!(rhs.len(), 5);
        assert!(rhs.as_opt::<Extension>().is_some());
        Ok(())
    }

    #[test]
    fn similarity_search_tree_executes_to_bool_array() -> VortexResult<()> {
        // 4 rows of 3-dim vectors; the first and last match the query [1, 0, 0].
        let data = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, //
            ],
        )?;
        let query = [1.0f32, 0.0, 0.0];

        let tree = build_similarity_search_tree(data, &query, 0.5)?;
        let mut ctx = test_session().create_execution_ctx();
        let result: BoolArray = tree.execute(&mut ctx)?;

        let bits = result.to_bit_buffer();
        assert_eq!(bits.len(), 4);
        assert!(bits.value(0));
        assert!(!bits.value(1));
        assert!(!bits.value(2));
        assert!(bits.value(3));
        Ok(())
    }

    #[test]
    fn turboquant_roundtrip_preserves_ranking() -> VortexResult<()> {
        // Build 6 rows of 128-dim vectors where row 0 is highly correlated with the query.
        // TurboQuant should preserve the "row 0 is the best match" ordering.
        const DIM: u32 = 128;
        const NUM_ROWS: usize = 6;

        let mut values = Vec::<f32>::with_capacity(NUM_ROWS * DIM as usize);
        let query: Vec<f32> = (0..DIM as usize)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();

        // Row 0: identical to query (cosine=1.0)
        values.extend_from_slice(&query);
        // Row 1: query + noise
        for (i, q) in query.iter().enumerate() {
            values.push(q + 0.05 * ((i as f32) * 0.03).cos());
        }
        // Rows 2..6: unrelated patterns
        for row in 2..NUM_ROWS {
            for i in 0..DIM as usize {
                values.push(((row as f32 * 1.3 + i as f32) * 0.07).sin());
            }
        }

        let data = vector_array(DIM, &values)?;
        let mut ctx = test_session().create_execution_ctx();
        let compressed = compress_turboquant(data, &mut ctx)?;
        assert_eq!(compressed.len(), NUM_ROWS);

        // Build a tree with a low threshold so row 0 (cosine=1.0 exact) matches.
        let tree = build_similarity_search_tree(compressed, &query, 0.95)?;
        let result: BoolArray = tree.execute(&mut ctx)?;
        let bits = result.to_bit_buffer();
        assert_eq!(bits.len(), NUM_ROWS);
        assert!(
            bits.value(0),
            "row 0 (identical to query) must match at threshold 0.95 even after TurboQuant"
        );
        Ok(())
    }
}
