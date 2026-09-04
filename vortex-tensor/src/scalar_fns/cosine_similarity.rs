// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity for tensor-like columns.
//!
//! [`CosineSimilarity`] derives each result from the decoded input coordinates and preserves the
//! established floating-point operation order.

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::unstable::row::InitializedElement;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::UninitElementSink;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::scalar_fns::arithmetic::inner_product_row;
use crate::scalar_fns::arithmetic::l2_norm_row;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::BinaryTensorOpMetadata;

/// Cosine similarity between two columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor or vector.
/// The shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement.
///
/// Both inputs must be tensor-like extension arrays (`FixedShapeTensor` or `Vector`) with the
/// same dtype and a float element type. The output is a float column of the same float type.
#[derive(Clone, Debug, Default)]
pub struct CosineSimilarity;

impl CosineSimilarity {
    /// Constructs a [`ScalarFnArray`] that lazily computes the cosine similarity between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error unless both inputs are float tensors with the same dtype, ignoring
    /// top-level nullability.
    pub fn try_new(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(CosineSimilarity.bind(EmptyOptions), vec![lhs, rhs])
    }
}

impl RowFn for CosineSimilarity {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.cosine_similarity");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        match_each_float_ptype!(tensor_element_ptype(args)?, |T| {
            visitor.visit_into::<(TensorRow<T>, TensorRow<T>), UninitElementSink<T>, _>(
                (),
                |(lhs, rhs), output| {
                    // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
                    unsafe { InitializedElement::write(output, cosine_similarity_row(lhs, rhs)) }
                },
            )
        })
    }
}

impl ScalarFnArrayVTable for CosineSimilarity {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(BinaryTensorOpMetadata::encode_from_view(view)?))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let reconstructed =
            BinaryTensorOpMetadata::decode_children(metadata, len, children, session)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: reconstructed,
        })
    }
}

/// Computes `dot(lhs, rhs) / (||lhs|| * ||rhs||)` using the scalar functions' established
/// arithmetic order. A zero denominator produces `0.0`.
fn cosine_similarity_row<T: Float + NativePType>(lhs: &[T], rhs: &[T]) -> T {
    let dot = inner_product_row(lhs, rhs);
    let denominator = l2_norm_row(lhs) * l2_norm_row(rhs);

    if denominator == T::zero() {
        T::zero()
    } else {
        dot / denominator
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::cosine_similarity::CosineSimilarity;
    use crate::tests::SESSION;
    use crate::types::vector::Vector;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::constant_tensor_array;
    use crate::utils::test_helpers::literal_vector_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;
    use crate::utils::test_helpers::zero_width_vector_array;

    fn evaluate_cosine_similarity<T: NativePType>(
        lhs: ArrayRef,
        rhs: ArrayRef,
    ) -> VortexResult<Vec<T>> {
        let result = CosineSimilarity::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let output: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        Ok(output.as_slice::<T>().to_vec())
    }

    #[test]
    fn test_zero_width_and_empty_inputs() -> VortexResult<()> {
        let lhs = zero_width_vector_array::<f64>(3)?;
        let rhs = zero_width_vector_array::<f64>(3)?;
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[0.0, 0.0, 0.0]);

        let lhs = vector_array::<f64>(2, &[])?;
        let rhs = vector_array::<f64>(2, &[])?;
        assert!(evaluate_cosine_similarity::<f64>(lhs, rhs)?.is_empty());

        Ok(())
    }

    #[test]
    fn test_ieee_edges_follow_direct_arithmetic() -> VortexResult<()> {
        let overflow = vector_array(2, &[f32::MAX, 1.0])?;
        let overflow_result = evaluate_cosine_similarity::<f32>(overflow.clone(), overflow)?;
        assert!(overflow_result[0].is_nan());

        let underflow = vector_array(2, &[f32::MIN_POSITIVE, f32::MIN_POSITIVE])?;
        let underflow_result = evaluate_cosine_similarity::<f32>(underflow.clone(), underflow)?;
        assert_eq!(underflow_result[0].to_bits(), 0.0f32.to_bits());

        Ok(())
    }

    #[test]
    fn test_encoded_constants_match_materialized_rows_bitwise() -> VortexResult<()> {
        let lhs_row = [1.0e10f32, 1.0, -1.0e10];
        let rhs_row = [1.0f32, 2.0, 3.0];
        let materialized_lhs = vector_array(3, &[lhs_row, lhs_row, lhs_row].concat())?;
        let materialized_rhs = vector_array(3, &[rhs_row, rhs_row, rhs_row].concat())?;
        let expected: Vec<_> =
            evaluate_cosine_similarity::<f32>(materialized_lhs.clone(), materialized_rhs.clone())?
                .into_iter()
                .map(f32::to_bits)
                .collect();

        let cases = [
            (
                Vector::constant_array(&lhs_row, 3)?,
                materialized_rhs.clone(),
            ),
            (literal_vector_array(&lhs_row, 3), materialized_rhs),
            (
                materialized_lhs.clone(),
                Vector::constant_array(&rhs_row, 3)?,
            ),
            (materialized_lhs, literal_vector_array(&rhs_row, 3)),
        ];
        for (lhs, rhs) in cases {
            let actual: Vec<_> = evaluate_cosine_similarity::<f32>(lhs, rhs)?
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(actual, expected);
        }

        Ok(())
    }

    #[test]
    fn unit_vectors_1d() -> VortexResult<()> {
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // Tensor 1
                0.0, 1.0, 0.0, // Tensor 2
            ],
        )?;
        let rhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // Tensor 1
                1.0, 0.0, 0.0, // Tensor 2
            ],
        )?;

        // Row 0: identical -> 1.0, row 1: orthogonal -> 0.0.
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
        Ok(())
    }

    /// Single-row cosine similarity for various vector pairs.
    #[rstest]
    // Antiparallel -> -1.0.
    #[case::opposite(&[3], &[1.0, 0.0, 0.0],  &[-1.0, 0.0, 0.0], &[-1.0])]
    // dot=24, both magnitudes=5 -> 24/25 = 0.96.
    #[case::non_unit(&[2], &[3.0, 4.0],        &[4.0, 3.0],       &[0.96])]
    // Zero vector -> guarded to 0.0.
    #[case::zero_norm(&[2], &[0.0, 0.0],       &[1.0, 0.0],       &[0.0])]
    fn single_row(
        #[case] shape: &[usize],
        #[case] lhs_elems: &[f64],
        #[case] rhs_elems: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let lhs = tensor_array(shape, lhs_elems)?;
        let rhs = tensor_array(shape, rhs_elems)?;
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, expected);
        Ok(())
    }

    /// Self-similarity across various tensor shapes should always produce 1.0.
    #[rstest]
    // 2x3 matrix, flattened to 6 elements.
    #[case::matrix_2d(
        &[2, 3],
        &[
            1.0, 0.0, 0.0, // row 0
            0.0, 0.0, 0.0, // row 1
        ],
    )]
    // 2x2x2 tensor, 8 elements.
    #[case::tensor_3d(&[2, 2, 2], &[1.0; 8])]
    fn self_similarity(#[case] shape: &[usize], #[case] elements: &[f64]) -> VortexResult<()> {
        let lhs = tensor_array(shape, elements)?;
        let rhs = tensor_array(shape, elements)?;
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[1.0]);
        Ok(())
    }

    #[test]
    fn scalar_0d() -> VortexResult<()> {
        // 0-dimensional tensor: each "tensor" is a single scalar value.
        let lhs = tensor_array(&[], &[5.0, 3.0])?;
        let rhs = tensor_array(&[], &[5.0, -3.0])?;

        // Same sign -> 1.0, opposite sign -> -1.0.
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn many_rows() -> VortexResult<()> {
        // 5 tensors of shape [4] compared against themselves -> all 1.0.
        let lhs = tensor_array(
            &[4],
            &[
                1.0, 2.0, 3.0, 4.0, // tensor 0
                0.0, 1.0, 0.0, 0.0, // tensor 1
                5.0, 0.0, 5.0, 0.0, // tensor 2
                1.0, 1.0, 1.0, 1.0, // tensor 3
                0.0, 0.0, 0.0, 7.0, // tensor 4
            ],
        )?;
        let rhs = lhs.clone();

        assert_close(
            &evaluate_cosine_similarity(lhs, rhs)?,
            &[1.0, 1.0, 1.0, 1.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn constant_query_tensor() -> VortexResult<()> {
        // Compare 4 tensors of shape [3] against a single constant query tensor [1,0,0].
        let data = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // tensor 0
                0.0, 1.0, 0.0, // tensor 1
                0.0, 0.0, 1.0, // tensor 2
                1.0, 0.0, 0.0, // tensor 3
            ],
        )?;
        let query = constant_tensor_array(&[3], &[1.0, 0.0, 0.0], 4)?;

        assert_close(
            &evaluate_cosine_similarity(data, query)?,
            &[1.0, 0.0, 0.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn vector_unit_vectors() -> VortexResult<()> {
        let lhs = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // vector 0
                0.0, 1.0, 0.0, // vector 1
            ],
        )?;
        let rhs = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // vector 0
                1.0, 0.0, 0.0, // vector 1
            ],
        )?;

        // Row 0: identical -> 1.0, row 1: orthogonal -> 0.0.
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn vector_constant_query() -> VortexResult<()> {
        let data = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // vector 0
                0.0, 1.0, 0.0, // vector 1
                0.0, 0.0, 1.0, // vector 2
                1.0, 0.0, 0.0, // vector 3
            ],
        )?;
        let query = Vector::constant_array(&[1.0, 0.0, 0.0], 4)?;

        assert_close(
            &evaluate_cosine_similarity(data, query)?,
            &[1.0, 0.0, 0.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 2 rows of dim-2 vectors. Row 1 of rhs is masked as null.
        let lhs = tensor_array(&[2], &[3.0, 4.0, 1.0, 0.0])?;
        let rhs = tensor_array(&[2], &[3.0, 4.0, 0.0, 1.0])?;
        let rhs = MaskedArray::try_new(rhs, Validity::from_iter([true, false]))?.into_array();

        let result = CosineSimilarity::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: self-similarity = 1.0, row 1: null.
        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[1.0]);
        Ok(())
    }

    #[test]
    fn constant_lhs_matches_plain_tensor() -> VortexResult<()> {
        // The constant query `[1, 2, 2]` has norm 3, so its normalized form is `[1/3, 2/3, 2/3]`.
        // Expected cosine similarity against each row is `dot([1, 2, 2], row) / (3 * ||row||)`.
        let lhs = constant_tensor_array(&[3], &[1.0, 2.0, 2.0], 4)?;
        let rhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // dot=1, ||rhs||=1, expected=1/3
                1.0, 2.0, 2.0, // dot=9, ||rhs||=3, expected=1
                0.0, 0.0, 1.0, // dot=2, ||rhs||=1, expected=2/3
                2.0, 1.0, 2.0, // dot=8, ||rhs||=3, expected=8/9
            ],
        )?;
        assert_close(
            &evaluate_cosine_similarity(lhs, rhs)?,
            &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
        );
        Ok(())
    }

    #[test]
    fn constant_rhs_matches_plain_tensor() -> VortexResult<()> {
        // Mirror of `constant_lhs_matches_plain_tensor` with the constant on the right.
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, //
                1.0, 2.0, 2.0, //
                0.0, 0.0, 1.0, //
                2.0, 1.0, 2.0, //
            ],
        )?;
        let rhs = constant_tensor_array(&[3], &[1.0, 2.0, 2.0], 4)?;
        assert_close(
            &evaluate_cosine_similarity(lhs, rhs)?,
            &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
        );
        Ok(())
    }

    #[test]
    fn both_constant_tensors() -> VortexResult<()> {
        // `[1, 0, 0]` vs `[1, 1, 0]`. dot=1, ||lhs||=1, ||rhs||=sqrt(2), expected=1/sqrt(2).
        let lhs = constant_tensor_array(&[3], &[1.0, 0.0, 0.0], 3)?;
        let rhs = constant_tensor_array(&[3], &[1.0, 1.0, 0.0], 3)?;
        let expected = 1.0 / 2.0_f64.sqrt();
        assert_close(
            &evaluate_cosine_similarity(lhs, rhs)?,
            &[expected, expected, expected],
        );
        Ok(())
    }

    #[test]
    fn constant_zero_norm_query() -> VortexResult<()> {
        // A zero-norm constant query must produce `0.0` for every row.
        let lhs = constant_tensor_array(&[3], &[0.0, 0.0, 0.0], 3)?;
        let rhs = tensor_array(
            &[3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        )?;
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn constant_self_similarity_nonunit() -> VortexResult<()> {
        // A non-unit constant query compared to itself must produce `1.0`.
        let lhs = constant_tensor_array(&[3], &[3.0, 4.0, 0.0], 5)?;
        let rhs = constant_tensor_array(&[3], &[3.0, 4.0, 0.0], 5)?;
        assert_close(&evaluate_cosine_similarity(lhs, rhs)?, &[1.0; 5]);
        Ok(())
    }

    #[test]
    fn vector_constant_matches_plain() -> VortexResult<()> {
        // Exercise the `Vector` extension variant through the new pre-pass.
        let lhs = Vector::constant_array(&[1.0, 2.0, 2.0], 4)?;
        let rhs = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, //
                1.0, 2.0, 2.0, //
                0.0, 0.0, 1.0, //
                2.0, 1.0, 2.0, //
            ],
        )?;
        assert_close(
            &evaluate_cosine_similarity(lhs, rhs)?,
            &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
        );
        Ok(())
    }

    #[rstest]
    #[case::vector(cosine_vector_lhs(), cosine_vector_rhs())]
    #[case::fixed_shape_tensor(cosine_tensor_lhs(), cosine_tensor_rhs())]
    fn serde_round_trip(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) -> VortexResult<()> {
        let original = CosineSimilarity::try_new(lhs.clone(), rhs.clone())?.into_array();

        let plugin = ScalarFnArrayPlugin::new(CosineSimilarity);
        let serialization = plugin
            .serialize(&original, &SESSION)?
            .expect("CosineSimilarity serialize must produce metadata");

        let children = vec![lhs, rhs];
        let recovered = plugin.deserialize(
            ArrayDeserialization::new(
                plugin.id(),
                original.dtype(),
                original.len(),
                &serialization.metadata,
                &[],
                &children,
            ),
            &SESSION,
        )?;

        assert_eq!(recovered.dtype(), original.dtype());
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered.encoding_id(), original.encoding_id());
        Ok(())
    }

    fn cosine_vector_lhs() -> ArrayRef {
        vector_array(3, &[1.0, 0.0, 0.0, 3.0, 4.0, 0.0]).expect("valid vector array")
    }

    fn cosine_vector_rhs() -> ArrayRef {
        vector_array(3, &[0.0, 1.0, 0.0, 3.0, 4.0, 0.0]).expect("valid vector array")
    }

    fn cosine_tensor_lhs() -> ArrayRef {
        tensor_array(&[2], &[1.0, 0.0, 3.0, 4.0]).expect("valid tensor array")
    }

    fn cosine_tensor_rhs() -> ArrayRef {
        tensor_array(&[2], &[0.0, 1.0, 3.0, 4.0]).expect("valid tensor array")
    }
}
