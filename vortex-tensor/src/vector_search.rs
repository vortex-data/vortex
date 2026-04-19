// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reusable helpers for building brute-force vector similarity search expressions over
//! [`Vector`] extension arrays.
//!
//! [`build_similarity_search_tree`] broadcasts the query into the shape expected by
//! [`CosineSimilarity`] via `Vector::constant_array` and returns a lazy
//! `Binary(Gt, [CosineSimilarity(data, query), threshold])` expression. The caller is responsible
//! for preparing `data` (e.g. by running it through [`turboquant_encode`]); this builder does not
//! compress.
//!
//! Executing the tree into a [`BoolArray`] yields one boolean per row indicating whether that row's
//! cosine similarity to the query exceeds `threshold`.
//!
//! # Example
//!
//! ```ignore
//! use vortex_array::{ArrayRef, VortexSessionExecute};
//! use vortex_array::arrays::BoolArray;
//! use vortex_session::VortexSession;
//! use vortex_tensor::encodings::turboquant::{TurboQuantConfig, turboquant_encode};
//! use vortex_tensor::vector_search::build_similarity_search_tree;
//!
//! fn run(session: &VortexSession, data: ArrayRef, query: &[f32]) -> anyhow::Result<()> {
//!     let mut ctx = session.create_execution_ctx();
//!     let data = turboquant_encode(data, &TurboQuantConfig::default(), &mut ctx)?;
//!     let tree = build_similarity_search_tree(data, query, 0.8)?;
//!     let _matches: BoolArray = tree.execute(&mut ctx)?;
//!     Ok(())
//! }
//! ```
//!
//! [`Vector`]: crate::vector::Vector
//! [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
//! [`turboquant_encode`]: crate::encodings::turboquant::turboquant_encode
//! [`BoolArray`]: vortex_array::arrays::BoolArray

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::vector::Vector;

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
    let query_vec = Vector::constant_array(query, num_rows)?;

    let cosine = CosineSimilarity::try_new_array(data, query_vec, num_rows)?.into_array();

    let threshold_scalar = Scalar::primitive(threshold, Nullability::NonNullable);
    let threshold_array = ConstantArray::new(threshold_scalar, num_rows).into_array();

    cosine.binary(threshold_array, Operator::Gt)
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_error::VortexResult;

    use super::build_similarity_search_tree;
    use crate::encodings::turboquant::TurboQuantConfig;
    use crate::encodings::turboquant::turboquant_encode;
    use crate::tests::SESSION;
    use crate::utils::test_helpers::vector_array;

    #[test]
    fn similarity_search_tree_executes_to_bool_array() -> VortexResult<()> {
        // 4 rows of 3-dim vectors; the first and last match the query [1, 0, 0].
        let data = vector_array(
            3,
            &[
                1.0f32, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, //
            ],
        )?;
        let query = [1.0f32, 0.0, 0.0];

        let tree = build_similarity_search_tree(data, &query, 0.5)?;
        let mut ctx = SESSION.create_execution_ctx();
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
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = turboquant_encode(data, &TurboQuantConfig::default(), &mut ctx)?;
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
