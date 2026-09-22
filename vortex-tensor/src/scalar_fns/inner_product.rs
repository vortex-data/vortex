// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inner products for tensor-like columns.
//!
//! [`InnerProduct`] derives each result from the decoded input coordinates and preserves
//! left-to-right floating-point accumulation.

use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
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
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::BinaryTensorOpMetadata;

/// Inner product (dot product) between two columns.
///
/// Computes `sum(a_i * b_i)` over the flat backing buffer of each tensor or vector. For vectors
/// this is the standard dot product; for higher-rank ([`FixedShapeTensor`]) arrays this is the
/// Frobenius inner product.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone, Debug, Default)]
pub struct InnerProduct;

impl InnerProduct {
    /// Constructs a [`ScalarFnArray`] that lazily computes the inner product between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error unless both inputs are float tensors with the same dtype, ignoring
    /// top-level nullability.
    pub fn try_new(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(InnerProduct.bind(EmptyOptions), vec![lhs, rhs])
    }
}

impl RowFn for InnerProduct {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.inner_product");
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
                    unsafe { InitializedElement::write(output, inner_product_row(lhs, rhs)) }
                },
            )
        })
    }
}

impl ScalarFnArrayVTable for InnerProduct {
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

    use crate::scalar_fns::inner_product::InnerProduct;
    use crate::tests::SESSION;
    use crate::types::vector::Vector;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::literal_vector_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;
    use crate::utils::test_helpers::zero_width_vector_array;

    fn evaluate_inner_product<T: NativePType>(
        lhs: ArrayRef,
        rhs: ArrayRef,
    ) -> VortexResult<Vec<T>> {
        let result = InnerProduct::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let output: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        Ok(output.as_slice::<T>().to_vec())
    }

    #[test]
    fn test_zero_width_and_empty_inputs() -> VortexResult<()> {
        let lhs = zero_width_vector_array::<f64>(3)?;
        let rhs = zero_width_vector_array::<f64>(3)?;
        assert_close(&evaluate_inner_product(lhs, rhs)?, &[0.0, 0.0, 0.0]);

        let lhs = vector_array::<f64>(2, &[])?;
        let rhs = vector_array::<f64>(2, &[])?;
        assert!(evaluate_inner_product::<f64>(lhs, rhs)?.is_empty());

        Ok(())
    }

    #[test]
    fn test_encoded_constants_match_materialized_rows_bitwise() -> VortexResult<()> {
        let lhs_row = [1.0e10f32, 1.0, -1.0e10];
        let rhs_row = [1.0f32, 1.0, 1.0];
        let materialized_lhs = vector_array(3, &[lhs_row, lhs_row, lhs_row].concat())?;
        let materialized_rhs = vector_array(3, &[rhs_row, rhs_row, rhs_row].concat())?;
        let expected: Vec<_> =
            evaluate_inner_product::<f32>(materialized_lhs.clone(), materialized_rhs.clone())?
                .into_iter()
                .map(f32::to_bits)
                .collect();
        assert_eq!(expected, vec![0.0f32.to_bits(); 3]);

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
            let actual: Vec<_> = evaluate_inner_product::<f32>(lhs, rhs)?
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(actual, expected);
        }

        Ok(())
    }

    /// Single-row inner product for various vector pairs.
    #[rstest]
    // Orthogonal: [1, 0] . [0, 1] = 0.
    #[case::orthogonal(&[2], &[1.0, 0.0], &[0.0, 1.0], &[0.0])]
    // Parallel: [3, 4] . [3, 4] = 9 + 16 = 25.
    #[case::parallel(&[2], &[3.0, 4.0], &[3.0, 4.0], &[25.0])]
    // Antiparallel: [1, 2] . [-1, -2] = -1 + -4 = -5.
    #[case::antiparallel(&[2], &[1.0, 2.0], &[-1.0, -2.0], &[-5.0])]
    // Scaled: [2, 0] . [3, 0] = 6.
    #[case::scaled(&[2], &[2.0, 0.0], &[3.0, 0.0], &[6.0])]
    fn single_row(
        #[case] shape: &[usize],
        #[case] lhs_elems: &[f64],
        #[case] rhs_elems: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let lhs = tensor_array(shape, lhs_elems)?;
        let rhs = tensor_array(shape, rhs_elems)?;
        assert_close(&evaluate_inner_product(lhs, rhs)?, expected);
        Ok(())
    }

    #[test]
    fn multiple_rows() -> VortexResult<()> {
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // tensor 0
                3.0, 4.0, 0.0, // tensor 1
                1.0, 1.0, 1.0, // tensor 2
            ],
        )?;
        let rhs = tensor_array(
            &[3],
            &[
                0.0, 1.0, 0.0, // tensor 0: dot = 0
                3.0, 4.0, 0.0, // tensor 1: dot = 25
                2.0, 2.0, 2.0, // tensor 2: dot = 6
            ],
        )?;
        assert_close(&evaluate_inner_product(lhs, rhs)?, &[0.0, 25.0, 6.0]);
        Ok(())
    }

    #[test]
    fn vector_inner_product() -> VortexResult<()> {
        let lhs = vector_array(
            2,
            &[
                3.0, 4.0, // vector 0
                1.0, 0.0, // vector 1
            ],
        )?;
        let rhs = vector_array(
            2,
            &[
                3.0, 4.0, // vector 0: dot = 25
                0.0, 1.0, // vector 1: dot = 0
            ],
        )?;
        assert_close(&evaluate_inner_product(lhs, rhs)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 3 rows of dim-2 vectors. Row 1 of lhs is masked as null.
        let lhs = tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let rhs = tensor_array(&[2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
        let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

        let result = InnerProduct::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 1*7 + 2*8 = 23, row 1: null, row 2: 5*11 + 6*12 = 127.
        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert!(prim.is_valid(2, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[23.0]);
        assert_close(&[prim.as_slice::<f64>()[2]], &[127.0]);
        Ok(())
    }

    #[test]
    fn rejects_non_extension_dtype() {
        let lhs = PrimitiveArray::from_iter([1.0_f64, 2.0]).into_array();
        let rhs = PrimitiveArray::from_iter([3.0_f64, 4.0]).into_array();
        let result = InnerProduct::try_new(lhs, rhs);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_mismatched_dtypes() -> VortexResult<()> {
        let lhs = tensor_array(&[2], &[1.0_f64, 2.0])?;
        let rhs = vector_array(2, &[3.0_f64, 4.0])?;
        let result = InnerProduct::try_new(lhs, rhs);
        assert!(result.is_err());
        Ok(())
    }

    #[rstest]
    #[case::vector(inner_product_vector_lhs(), inner_product_vector_rhs())]
    #[case::fixed_shape_tensor(inner_product_tensor_lhs(), inner_product_tensor_rhs())]
    fn serde_round_trip(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) -> VortexResult<()> {
        let original = InnerProduct::try_new(lhs.clone(), rhs.clone())?.into_array();

        let plugin = ScalarFnArrayPlugin::new(InnerProduct);
        let serialization = plugin
            .serialize(&original, &SESSION)?
            .expect("InnerProduct serialize must produce metadata");

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

    fn inner_product_vector_lhs() -> ArrayRef {
        vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid vector array")
    }

    fn inner_product_vector_rhs() -> ArrayRef {
        vector_array(3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("valid vector array")
    }

    fn inner_product_tensor_lhs() -> ArrayRef {
        tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0]).expect("valid tensor array")
    }

    fn inner_product_tensor_rhs() -> ArrayRef {
        tensor_array(&[2], &[5.0, 6.0, 7.0, 8.0]).expect("valid tensor array")
    }
}
