// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 norms for tensor-like columns.
//!
//! [`L2Norm`] computes only the magnitude of each input row. Use
//! [`L2Normalize`](super::l2_normalize::L2Normalize) when the normalized coordinates and the
//! magnitude are both required.

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFn as ScalarFnArrayEncoding;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::proto::dtype as pb;
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
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::scalar_fns::arithmetic::l2_norm_row;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;

/// L2 norm (Euclidean norm) of a tensor or vector column.
///
/// Computes `||v|| = sqrt(sum(v_i^2))` over the flat coordinates of each tensor-like value.
///
/// The input must be a tensor-like extension array with a float element type. The output is a float
/// column of the same float type. Use [`L2Normalize`] when both the normalized value and its norm
/// are required.
///
/// [`L2Normalize`]: crate::scalar_fns::l2_normalize::L2Normalize
#[derive(Clone, Debug, Default)]
pub struct L2Norm;

impl L2Norm {
    /// Constructs a [`ScalarFnArray`] that lazily computes the L2 norm over `child`.
    ///
    /// # Errors
    ///
    /// Returns an error if `child` is not a float [`Vector`] or [`FixedShapeTensor`].
    ///
    /// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
    /// [`Vector`]: crate::vector::Vector
    pub fn try_new(child: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Norm.bind(EmptyOptions), vec![child])
    }
}

impl RowFn for L2Norm {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.l2_norm");
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
            visitor.visit_into::<(TensorRow<T>,), UninitElementSink<T>, _>((), |(row,), output| {
                // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
                unsafe { InitializedElement::write(output, l2_norm_row(row)) }
            })
        })
    }
}

/// Metadata for a serialized [`L2Norm`] array: the single `input` child's [`DType`], which carries
/// the extension type (`FixedShapeTensor` vs `Vector`), dimension, and nullability that are not
/// recoverable from the parent's primitive-float output.
#[derive(Clone, prost::Message)]
struct L2NormMetadata {
    #[prost(message, optional, tag = "1")]
    input_dtype: Option<pb::DType>,
}

impl ScalarFnArrayVTable for L2Norm {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let scalar_fn_array = view.as_::<ScalarFnArrayEncoding>();
        let input_dtype = Some(scalar_fn_array.child_at(0).dtype().try_into()?);
        Ok(Some(L2NormMetadata { input_dtype }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let metadata = L2NormMetadata::decode(metadata)
            .map_err(|error| vortex_err!("failed to decode L2Norm metadata: {error}"))?;
        let input_pb = metadata
            .input_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2Norm metadata must contain input_dtype"))?;
        let input_dtype = DType::from_proto(input_pb, session)?;
        let child = children.get(0, &input_dtype, len)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![child],
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::l2_norm::L2Norm;
    use crate::tests::SESSION;
    use crate::types::vector::Vector;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::literal_vector_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;
    use crate::utils::test_helpers::zero_width_vector_array;

    fn evaluate_l2_norm<T: NativePType>(input: ArrayRef) -> VortexResult<Vec<T>> {
        let result = L2Norm::try_new(input)?;
        let mut ctx = SESSION.create_execution_ctx();
        let output: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        Ok(output.as_slice::<T>().to_vec())
    }

    #[test]
    fn test_zero_width_and_empty_inputs() -> VortexResult<()> {
        assert_close(
            &evaluate_l2_norm(zero_width_vector_array::<f64>(3)?)?,
            &[0.0, 0.0, 0.0],
        );
        assert!(evaluate_l2_norm::<f64>(vector_array::<f64>(2, &[])?)?.is_empty());
        assert_close(
            &evaluate_l2_norm(Vector::constant_array::<f64>(&[], 3)?)?,
            &[0.0, 0.0, 0.0],
        );

        Ok(())
    }

    #[rstest]
    #[case::three_four_five(&[2], &[3.0, 4.0], &[5.0])]
    #[case::zero_vector(&[3], &[0.0, 0.0, 0.0], &[0.0])]
    #[case::single_element(&[1], &[7.0], &[7.0])]
    #[case::negative_elements(&[2], &[-3.0, -4.0], &[5.0])]
    fn known_norms(
        #[case] shape: &[usize],
        #[case] elements: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let arr = tensor_array(shape, elements)?;
        assert_close(&evaluate_l2_norm(arr)?, expected);
        Ok(())
    }

    #[test]
    fn multiple_rows() -> VortexResult<()> {
        let arr = tensor_array(
            &[3],
            &[
                3.0, 4.0, 0.0, // norm = 5.0
                0.0, 0.0, 0.0, // norm = 0.0
                1.0, 1.0, 1.0, // norm = sqrt(3)
            ],
        )?;
        assert_close(&evaluate_l2_norm(arr)?, &[5.0, 0.0, 3.0_f64.sqrt()]);
        Ok(())
    }

    #[test]
    fn vector_multiple_rows() -> VortexResult<()> {
        let arr = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // norm = 1.0
                3.0, 4.0, 0.0, // norm = 5.0
            ],
        )?;
        assert_close(&evaluate_l2_norm(arr)?, &[1.0, 5.0]);
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 2 rows of dim-2 vectors. Row 1 is masked as null.
        let arr = tensor_array(&[2], &[3.0, 4.0, 0.0, 0.0])?;
        let arr = MaskedArray::try_new(arr, Validity::from_iter([true, false]))?.into_array();

        let result = L2Norm::try_new(arr)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: norm = 5.0, row 1: null.
        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[5.0]);
        Ok(())
    }

    /// A constant input whose scalar is a non-null tensor should short-circuit to a
    /// [`ConstantArray`] output whose scalar is the precomputed norm. Uses [`execute_until`] so
    /// execution stops at the [`Constant`] encoding instead of canonicalizing into a
    /// [`PrimitiveArray`].
    #[test]
    fn constant_non_null_input_yields_constant_output() -> VortexResult<()> {
        let input = literal_vector_array(&[3.0f64, 4.0], 4);

        let result = L2Norm::try_new(input)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let output = result.execute_until::<Constant>(&mut ctx)?;

        let constant = output
            .as_opt::<Constant>()
            .expect("L2Norm over a constant input must produce a constant output");
        assert_eq!(constant.len(), 4);
        let norm = constant
            .scalar()
            .as_primitive()
            .as_::<f64>()
            .expect("norm scalar must be a non-null primitive");
        assert_close(&[norm], &[5.0]);
        Ok(())
    }

    #[test]
    fn test_extension_backed_constant_yields_constant_output() -> VortexResult<()> {
        let input = Vector::constant_array(&[3.0f64, 4.0], 4)?;

        let result = L2Norm::try_new(input)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let output = result.execute_until::<Constant>(&mut ctx)?;

        let constant = output
            .as_opt::<Constant>()
            .expect("L2Norm over constant-backed extension storage must be constant");
        assert_eq!(constant.len(), 4);
        let norm = constant
            .scalar()
            .as_primitive()
            .as_::<f64>()
            .expect("norm scalar must be a non-null primitive");
        assert_eq!(norm.to_bits(), 5.0f64.to_bits());

        Ok(())
    }

    #[test]
    fn test_encoded_constants_match_materialized_rows_bitwise() -> VortexResult<()> {
        let row = [f32::MAX, 1.0, -1.0];
        let materialized = vector_array(3, &[row, row, row].concat())?;
        let storage_constant = Vector::constant_array(&row, 3)?;
        let literal_constant = literal_vector_array(&row, 3);

        let expected: Vec<_> = evaluate_l2_norm::<f32>(materialized)?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        for encoded in [storage_constant, literal_constant] {
            let actual: Vec<_> = evaluate_l2_norm::<f32>(encoded)?
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(actual, expected);
        }

        Ok(())
    }

    /// A constant input whose scalar is null should short-circuit to a null [`ConstantArray`] of
    /// the correct primitive dtype and length.
    #[test]
    fn constant_null_input_yields_null_constant_output() -> VortexResult<()> {
        let storage_dtype = DType::FixedSizeList(
            DType::Primitive(PType::F64, Nullability::NonNullable).into(),
            2,
            Nullability::Nullable,
        );
        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?.erased();
        let null_scalar = Scalar::null(DType::Extension(ext_dtype));
        let input = ConstantArray::new(null_scalar, 3).into_array();

        let result = L2Norm::try_new(input)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let output = result.execute_until::<Constant>(&mut ctx)?;

        let constant = output
            .as_opt::<Constant>()
            .expect("null constant input must produce a constant output");
        assert_eq!(constant.len(), 3);
        assert!(constant.scalar().is_null());
        assert_eq!(
            constant.dtype(),
            &DType::Primitive(PType::F64, Nullability::Nullable)
        );
        Ok(())
    }

    #[rstest]
    #[case::fixed_shape_tensor(l2_norm_tensor_child())]
    #[case::vector(l2_norm_vector_child())]
    fn serde_round_trip(#[case] child: ArrayRef) -> VortexResult<()> {
        let original = L2Norm::try_new(child.clone())?.into_array();

        let plugin = ScalarFnArrayPlugin::new(L2Norm);
        let serialization = plugin
            .serialize(&original, &SESSION)?
            .expect("L2Norm serialize must produce metadata");

        let children = vec![child];
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

    fn l2_norm_tensor_child() -> ArrayRef {
        tensor_array(&[3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid tensor array")
    }

    fn l2_norm_vector_child() -> ArrayRef {
        vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid vector array")
    }
}
