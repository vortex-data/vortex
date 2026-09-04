// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 normalization for tensor-like columns.
//!
//! [`L2Normalize`] computes normalized coordinates and their norm together. The implementation
//! writes both outputs in one pass so that the returned norm is exactly the value used for scaling.

use num_traits::Zero;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn as ScalarFnArrayEncoding;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::matcher::AnyTensor;
use crate::scalar_fns::arithmetic::l2_norm_row;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

const NORMALIZED_FIELD_NAME: &str = "normalized";
const NORM_FIELD_NAME: &str = "norm";

/// Splits each tensor or vector into its L2-normalized value and norm.
///
/// The result is a struct with `normalized` and `norm` fields. `norm` has the input element type,
/// and `normalized` has the non-nullable input tensor type. A null input produces a null struct
/// row. A valid zero-norm input produces a zero tensor and `0`; all other floating-point behavior
/// follows the input element type's arithmetic. Use [`L2Norm`] when only the norm is required.
///
/// [`L2Norm`]: crate::scalar_fns::l2_norm::L2Norm
#[derive(Clone, Debug, Default)]
pub struct L2Normalize;

impl L2Normalize {
    /// Constructs a [`ScalarFnArray`] that lazily computes the normalized value and norm of
    /// `child`.
    ///
    /// # Errors
    ///
    /// Returns an error if `child` is not a float [`Vector`] or [`FixedShapeTensor`], or if the
    /// scalar-function array cannot be constructed.
    ///
    /// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
    /// [`Vector`]: crate::vector::Vector
    pub fn try_new(child: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Normalize.bind(EmptyOptions), vec![child])
    }
}

impl ScalarFnVTable for L2Normalize {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.l2_normalize");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("L2Normalize must have exactly one child"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        l2_normalize_dtype(&arg_dtypes[0])
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let row_count = args.row_count();
        let ext_dtype = input.dtype().as_extension().clone();
        let tensor_match = ext_dtype
            .metadata_opt::<AnyTensor>()
            .vortex_expect("L2Normalize::return_dtype validated the input tensor metadata");
        let row_width = tensor_match.list_size() as usize;

        let input: ExtensionArray = input.execute(ctx)?;
        let validity = input.as_ref().validity()?;
        let validity_mask = validity
            .nullability()
            .is_nullable()
            .then(|| validity.execute_mask(row_count, ctx))
            .transpose()?;

        let flat = extract_flat_elements(input.storage_array(), row_width, ctx)?;

        match_each_float_ptype!(tensor_match.element_ptype(), |T| {
            let mut normalized_values = BufferMut::<T>::with_capacity(row_count * row_width);
            let mut norms = BufferMut::<T>::with_capacity(row_count);

            for row_index in 0..row_count {
                if validity_mask
                    .as_ref()
                    .is_some_and(|validity_mask| !validity_mask.value(row_index))
                {
                    normalized_values.push_n(T::zero(), row_width);
                    norms.push(T::zero());
                    continue;
                }

                let row = flat.row::<T>(row_index);
                let norm = l2_norm_row(row);
                norms.push(norm);

                if norm == T::zero() {
                    normalized_values.push_n(T::zero(), row_width);
                    continue;
                }

                for &value in row {
                    normalized_values.push(value / norm);
                }
            }

            let normalized_elements =
                PrimitiveArray::new(normalized_values.freeze(), Validity::NonNullable).into_array();
            let normalized_storage = FixedSizeListArray::try_new(
                normalized_elements,
                tensor_match.list_size(),
                Validity::NonNullable,
                row_count,
            )?
            .into_array();
            let normalized_dtype = ext_dtype.with_nullability(Nullability::NonNullable);
            let normalized =
                ExtensionArray::try_new(normalized_dtype, normalized_storage)?.into_array();
            let norms = PrimitiveArray::new(norms.freeze(), Validity::NonNullable).into_array();

            Ok(StructArray::try_new(
                [NORMALIZED_FIELD_NAME, NORM_FIELD_NAME].into(),
                [normalized, norms],
                row_count,
                validity,
            )?
            .into_array())
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[derive(Clone, prost::Message)]
struct L2NormalizeMetadata {
    /// The input dtype required to deserialize the child array.
    #[prost(message, optional, tag = "1")]
    input_dtype: Option<pb::DType>,
}

impl ScalarFnArrayVTable for L2Normalize {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let array = view.as_::<ScalarFnArrayEncoding>();
        let input_dtype = Some(array.child_at(0).dtype().try_into()?);

        Ok(Some(L2NormalizeMetadata { input_dtype }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let metadata = L2NormalizeMetadata::decode(metadata)
            .map_err(|error| vortex_err!("failed to decode L2Normalize metadata: {error}"))?;
        let input_dtype = metadata
            .input_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2Normalize metadata must contain input_dtype"))?;
        let input_dtype = DType::from_proto(input_dtype, session)?;
        l2_normalize_dtype(&input_dtype)?;
        let child = children.get(0, &input_dtype, len)?;

        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![child],
        })
    }
}

fn l2_normalize_dtype(input_dtype: &DType) -> VortexResult<DType> {
    let tensor_match = validate_tensor_float_input(input_dtype)?;
    let fields = StructFields::from_iter([
        (NORMALIZED_FIELD_NAME, input_dtype.as_nonnullable()),
        (
            NORM_FIELD_NAME,
            DType::Primitive(tensor_match.element_ptype(), Nullability::NonNullable),
        ),
    ]);

    Ok(DType::Struct(
        fields,
        Nullability::from(input_dtype.is_nullable()),
    ))
}

#[cfg(test)]
mod tests {
    use half::f16;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::l2_norm::L2Norm;
    use crate::scalar_fns::l2_normalize::L2Normalize;
    use crate::tests::SESSION;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    fn evaluate(input: ArrayRef) -> VortexResult<StructArray> {
        let mut ctx = SESSION.create_execution_ctx();
        L2Normalize::try_new(input)?.into_array().execute(&mut ctx)
    }

    fn evaluate_l2_norm<T: NativePType>(input: ArrayRef) -> VortexResult<Vec<T>> {
        let mut ctx = SESSION.create_execution_ctx();
        let output: PrimitiveArray = L2Norm::try_new(input)?.into_array().execute(&mut ctx)?;

        Ok(output.as_slice::<T>().to_vec())
    }

    fn normalized_values<T: NativePType>(output: &StructArray) -> VortexResult<Vec<T>> {
        let mut ctx = SESSION.create_execution_ctx();
        let normalized: ExtensionArray = output
            .unmasked_field_by_name(super::NORMALIZED_FIELD_NAME)?
            .clone()
            .execute(&mut ctx)?;
        let storage: FixedSizeListArray = normalized.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

        Ok(elements.as_slice::<T>().to_vec())
    }

    fn norms<T: NativePType>(output: &StructArray) -> VortexResult<Vec<T>> {
        let mut ctx = SESSION.create_execution_ctx();
        let norms: PrimitiveArray = output
            .unmasked_field_by_name(super::NORM_FIELD_NAME)?
            .clone()
            .execute(&mut ctx)?;

        Ok(norms.as_slice::<T>().to_vec())
    }

    #[test]
    fn test_normalizes_vector_rows() -> VortexResult<()> {
        let output = evaluate(vector_array(
            2,
            &[
                3.0f64, 4.0, // The first row.
                1.0, 0.0, // The second row.
            ],
        )?)?;

        assert_eq!(normalized_values::<f64>(&output)?, [0.6, 0.8, 1.0, 0.0]);
        assert_eq!(norms::<f64>(&output)?, [5.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_normalizes_fixed_shape_tensor_rows() -> VortexResult<()> {
        let output = evaluate(tensor_array(&[2, 1], &[3.0f32, 4.0])?)?;

        assert_eq!(normalized_values::<f32>(&output)?, [0.6, 0.8]);
        assert_eq!(norms::<f32>(&output)?, [5.0]);

        Ok(())
    }

    #[test]
    fn test_preserves_f16_output_type() -> VortexResult<()> {
        let output = evaluate(vector_array(2, &[f16::from_f32(3.0), f16::from_f32(4.0)])?)?;

        assert_eq!(norms::<f16>(&output)?, [f16::from_f32(5.0)]);
        assert_eq!(
            normalized_values::<f16>(&output)?,
            [
                f16::from_f32(3.0) / f16::from_f32(5.0),
                f16::from_f32(4.0) / f16::from_f32(5.0),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_zero_rows_produce_zero_fields() -> VortexResult<()> {
        let output = evaluate(vector_array(2, &[-0.0f64, 0.0])?)?;

        assert_eq!(normalized_values::<f64>(&output)?, [0.0, 0.0]);
        assert_eq!(norms::<f64>(&output)?, [0.0]);

        Ok(())
    }

    #[test]
    fn test_input_nulls_produce_null_struct_rows() -> VortexResult<()> {
        let input = vector_array(
            2,
            &[
                3.0f64, 4.0, // The valid row.
                1.0, 0.0, // The null row's placeholder values.
            ],
        )?;
        let input = MaskedArray::try_new(input, Validity::from_iter([true, false]))?.into_array();
        let output = evaluate(input)?;
        let mut ctx = SESSION.create_execution_ctx();

        assert!(output.is_valid(0, &mut ctx)?);
        assert!(!output.is_valid(1, &mut ctx)?);
        assert_eq!(normalized_values::<f64>(&output)?, [0.6, 0.8, 0.0, 0.0]);
        assert_eq!(norms::<f64>(&output)?, [5.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_follows_ieee_nonfinite_arithmetic() -> VortexResult<()> {
        let output = evaluate(vector_array(
            2,
            &[
                f64::INFINITY,
                1.0, // The infinite row.
                f64::NAN,
                1.0, // The NaN row.
            ],
        )?)?;
        let normalized = normalized_values::<f64>(&output)?;
        let norms = norms::<f64>(&output)?;

        assert!(normalized[0].is_nan());
        assert_eq!(normalized[1], 0.0);
        assert!(normalized[2].is_nan());
        assert!(normalized[3].is_nan());
        assert!(norms[0].is_infinite());
        assert!(norms[1].is_nan());

        Ok(())
    }

    #[test]
    fn test_follows_norm_overflow_and_underflow() -> VortexResult<()> {
        let output = evaluate(vector_array(
            2,
            &[
                f32::MAX,
                f32::MAX, // The overflowing row.
                f32::from_bits(1),
                0.0, // The underflowing row.
            ],
        )?)?;

        assert_eq!(normalized_values::<f32>(&output)?, [0.0; 4]);
        assert_eq!(norms::<f32>(&output)?, [f32::INFINITY, 0.0]);

        Ok(())
    }

    #[test]
    fn test_norm_field_matches_l2_norm_bit_for_bit() -> VortexResult<()> {
        let input = tensor_array(
            &[2, 2],
            &[
                3.0f32,
                4.0,
                0.0,
                0.0, // The finite row.
                f32::MAX,
                f32::MAX,
                f32::from_bits(1),
                0.0, // The overflowing row.
            ],
        )?;
        let expected = evaluate_l2_norm::<f32>(input.clone())?;
        let output = evaluate(input)?;
        let actual = norms::<f32>(&output)?;

        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    #[test]
    fn test_return_dtype_has_nonnullable_fields() -> VortexResult<()> {
        let input = vector_array(2, &[3.0f64, 4.0])?;
        let input = MaskedArray::try_new(input, Validity::AllValid)?.into_array();
        let output = L2Normalize::try_new(input)?.into_array();

        let DType::Struct(fields, Nullability::Nullable) = output.dtype() else {
            panic!("expected a nullable struct dtype, got {}", output.dtype());
        };
        assert_eq!(
            fields.names().iter().map(AsRef::as_ref).collect::<Vec<_>>(),
            ["normalized", "norm"]
        );
        assert!(
            !fields
                .field_by_index(0)
                .expect("L2Normalize return dtype must contain the normalized field")
                .is_nullable()
        );
        assert_eq!(
            fields.field_by_index(1),
            Some(DType::Primitive(PType::F64, Nullability::NonNullable))
        );

        Ok(())
    }

    #[test]
    fn test_serde_round_trip() -> VortexResult<()> {
        let child = vector_array(2, &[3.0f64, 4.0])?;
        let original = L2Normalize::try_new(child.clone())?.into_array();
        let plugin = ScalarFnArrayPlugin::new(L2Normalize);
        let serialization = plugin
            .serialize(&original, &SESSION)?
            .expect("L2Normalize serialize must produce metadata");

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
}
