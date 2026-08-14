// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splits float tensor rows into L2-normalized directions and norms.
//!
//! [`L2Normalize`] returns a struct with non-nullable `normalized` and `norms` fields. The struct
//! carries the input nullability, and null rows use zero-valued child payloads. An exact-zero row
//! produces a zero direction and zero norm.
//!
//! A [`Vector`](crate::vector::Vector) direction is refined to
//! [`UnitVector`]. A
//! [`FixedShapeTensor`](crate::fixed_shape_tensor::FixedShapeTensor) direction keeps its ordinary
//! tensor dtype because Vortex does not define a unit-tensor refinement. Normalization accumulates
//! squares in f64 and uses scaled accumulation when the ordinary sum overflows or underflows.

use num_traits::ToPrimitive;
use num_traits::Zero;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
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
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDType;
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
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::types::unit_vector::UnitVector;
use crate::types::vector::AnyVector;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// Splits each tensor-like row into its L2-normalized direction and physical norm.
///
/// Vector inputs produce a [`UnitVector`] direction. Fixed-shape tensor inputs retain their
/// ordinary tensor dtype because Vortex does not define a unit-tensor refinement. The two fields
/// are non-nullable, and input nullability is carried by the returned struct.
#[derive(Clone)]
pub struct L2Normalize;

impl L2Normalize {
    /// Creates an [`L2Normalize`] scalar function instance.
    pub fn new() -> TypedScalarFnInstance<L2Normalize> {
        TypedScalarFnInstance::new(L2Normalize, EmptyOptions)
    }

    /// Constructs a lazy [`ScalarFnArray`] that normalizes `child`.
    ///
    /// # Errors
    ///
    /// Returns an error if `child` is not a float tensor-like array or the scalar-function array
    /// cannot be constructed.
    pub fn try_new_array(child: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Normalize::new().erased(), vec![child])
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
        let input_dtype = &arg_dtypes[0];
        let tensor_match = validate_tensor_float_input(input_dtype)?;
        let normalized_dtype = normalized_output_dtype(input_dtype)?;
        let norms_dtype = DType::Primitive(tensor_match.element_ptype(), Nullability::NonNullable);

        Ok(DType::Struct(
            StructFields::new(
                FieldNames::from(["normalized", "norms"]),
                vec![normalized_dtype, norms_dtype],
            ),
            input_dtype.nullability(),
        ))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        normalize_array(input, ctx).map(|array| array.into_array())
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

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[derive(Clone, prost::Message)]
struct L2NormalizeMetadata {
    /// The child dtype required before deserializing the child array.
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
            .map_err(|error| vortex_err!("Failed to decode L2NormalizeMetadata: {error}"))?;
        let input_dtype = metadata
            .input_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2NormalizeMetadata missing input_dtype"))?;
        let input_dtype = DType::from_proto(input_dtype, session)?;
        let child = children.get(0, &input_dtype, len)?;

        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![child],
        })
    }
}

pub(crate) fn normalize_array(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    let row_count = input.len();
    let (normalized, norms, validity) = normalize_children(input, ctx)?;

    StructArray::try_new(
        FieldNames::from(["normalized", "norms"]),
        vec![normalized, norms],
        row_count,
        validity,
    )
}

pub(crate) fn normalize_children(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, ArrayRef, Validity)> {
    let row_count = input.len();
    let tensor_match = validate_tensor_float_input(input.dtype())?;
    let tensor_flat_size = tensor_match.list_size() as usize;
    let output_dtype = normalized_output_dtype(input.dtype())?;

    let input: ExtensionArray = input.execute(ctx)?;
    let validity = input.as_ref().validity()?;
    let valid_rows = validity
        .nullability()
        .is_nullable()
        .then(|| validity.execute_mask(row_count, ctx))
        .transpose()?;
    let flat = extract_flat_elements(input.storage_array(), tensor_flat_size, ctx)?;
    let (normalized, norms) = match_each_float_ptype!(flat.ptype(), |T| {
        let mut elements = BufferMut::<T>::with_capacity(row_count * tensor_flat_size);
        let mut norms = BufferMut::<T>::with_capacity(row_count);

        if let Some(valid_rows) = &valid_rows {
            for row_idx in 0..row_count {
                if !valid_rows.value(row_idx) {
                    // SAFETY: The buffers reserve one direction row and one norm per input row.
                    unsafe {
                        elements.push_n_unchecked(T::zero(), tensor_flat_size);
                        norms.push_unchecked(T::zero());
                    }
                } else {
                    // SAFETY: `elements` reserves `tensor_flat_size` values per input row.
                    let norm =
                        unsafe { normalize_row_into(flat.row::<T>(row_idx), &mut elements)? };
                    // SAFETY: `norms` reserves one value per input row.
                    unsafe { norms.push_unchecked(norm) };
                }
            }
        } else {
            for row_idx in 0..row_count {
                // SAFETY: `elements` reserves `tensor_flat_size` values for every input row.
                let norm = unsafe { normalize_row_into(flat.row::<T>(row_idx), &mut elements)? };
                // SAFETY: `norms` reserves one value for every input row.
                unsafe { norms.push_unchecked(norm) };
            }
        }

        let normalized =
            build_normalized_array(output_dtype, tensor_flat_size, row_count, elements.freeze())?;
        // SAFETY: The loop writes exactly one norm for each input row.
        let norms = unsafe { PrimitiveArray::new_unchecked(norms.freeze(), Validity::NonNullable) };

        Ok::<_, vortex_error::VortexError>((normalized, norms))
    })?;

    Ok((normalized, norms.into_array(), validity))
}

/// Writes the normalized row to `output` and returns its physical L2 norm.
///
/// # Safety
///
/// `output` must have spare capacity for every value in `row`.
// This runs once per row; inlining avoids returning a large `VortexResult` from the hot loop.
#[inline(always)]
pub(crate) unsafe fn normalize_row_into<T: NativePType>(
    row: &[T],
    output: &mut BufferMut<T>,
) -> VortexResult<T> {
    let sum_squares = row.iter().fold(0.0f64, |sum, value| {
        let value =
            ToPrimitive::to_f64(value).vortex_expect("float NativePType values convert to f64");
        sum + value * value
    });

    let (norm_f64, scaled_divisor) = if sum_squares.is_finite() && sum_squares != 0.0 {
        (sum_squares.sqrt(), None)
    } else if sum_squares == 0.0 && row.iter().all(Zero::is_zero) {
        (0.0, None)
    } else {
        let (scale, scaled_norm) = scaled_l2_norm(row);
        (scale * scaled_norm, Some((scale, scaled_norm)))
    };

    vortex_ensure!(
        norm_f64.is_finite(),
        "L2 norm must be finite, got {norm_f64}"
    );
    let norm = T::from_f64(norm_f64).ok_or_else(|| {
        vortex_err!(
            "L2 norm must be representable as {}, got {norm_f64}",
            T::PTYPE,
        )
    })?;
    vortex_ensure!(
        ToPrimitive::to_f64(&norm).is_some_and(f64::is_finite),
        "L2 norm must be representable as {}, got {norm_f64}",
        T::PTYPE,
    );
    if norm_f64 == 0.0 {
        // SAFETY: The caller reserves space for the entire row.
        unsafe { output.push_n_unchecked(T::zero(), row.len()) };
        return Ok(norm);
    }

    if let Some((scale, scaled_norm)) = scaled_divisor {
        for value in row {
            let value =
                ToPrimitive::to_f64(value).vortex_expect("float NativePType values convert to f64");
            let normalized = T::from_f64((value / scale) / scaled_norm)
                .vortex_expect("float NativePType values can represent an f64 direction");
            // SAFETY: The caller reserves space for the entire row.
            unsafe { output.push_unchecked(normalized) };
        }
    } else {
        for value in row {
            let value =
                ToPrimitive::to_f64(value).vortex_expect("float NativePType values convert to f64");
            let normalized = T::from_f64(value / norm_f64)
                .vortex_expect("float NativePType values can represent an f64 direction");
            // SAFETY: The caller reserves space for the entire row.
            unsafe { output.push_unchecked(normalized) };
        }
    }

    Ok(norm)
}

fn scaled_l2_norm<T: NativePType>(row: &[T]) -> (f64, f64) {
    let mut scale = 0.0f64;
    let mut sum_squares = 1.0f64;

    for value in row {
        let absolute = ToPrimitive::to_f64(value)
            .vortex_expect("float NativePType values convert to f64")
            .abs();
        if absolute.is_nan() {
            scale = f64::NAN;
            break;
        }
        if absolute.is_infinite() {
            scale = f64::INFINITY;
            break;
        }
        if absolute == 0.0 {
            continue;
        }

        if scale < absolute {
            let ratio = scale / absolute;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = absolute;
        } else {
            let ratio = absolute / scale;
            sum_squares += ratio * ratio;
        }
    }

    (scale, sum_squares.sqrt())
}

pub(crate) fn normalized_output_dtype(input_dtype: &DType) -> VortexResult<DType> {
    let ext_dtype = input_dtype.as_extension();
    if ext_dtype.is::<AnyVector>() {
        let unit_dtype = ExtDType::<UnitVector>::try_new(
            EmptyMetadata,
            ext_dtype.storage_dtype().as_nonnullable(),
        )?;
        return Ok(DType::Extension(unit_dtype.erased()));
    }

    Ok(input_dtype.as_nonnullable())
}

fn build_normalized_array<T: NativePType>(
    dtype: DType,
    tensor_flat_size: usize,
    row_count: usize,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    let list_size =
        u32::try_from(tensor_flat_size).vortex_expect("tensor flat size must fit into `u32`");
    // SAFETY: Tensor elements are always non-nullable, so the validity carries no length.
    let elements = unsafe { PrimitiveArray::new_unchecked(elements, Validity::NonNullable) };
    let storage = FixedSizeListArray::try_new(
        elements.into_array(),
        list_size,
        Validity::NonNullable,
        row_count,
    )?;

    if dtype.as_extension().is::<UnitVector>() {
        // SAFETY: `normalize_row_into` produced every valid row, and null rows contain zeros.
        return unsafe { UnitVector::new_unchecked(storage.into_array()) };
    }

    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage.into_array()).into_array())
}

#[cfg(test)]
mod tests {
    use half::f16;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_error::VortexResult;

    use crate::encodings::normalized::validate_normalized_rows;
    use crate::scalar_fns::l2_normalize::L2Normalize;
    use crate::tests::SESSION;
    use crate::unit_vector::AnyUnitVector;
    use crate::unit_vector::UnitVector;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    fn evaluate(input: ArrayRef) -> VortexResult<StructArray> {
        let mut ctx = SESSION.create_execution_ctx();
        L2Normalize::try_new_array(input)?
            .into_array()
            .execute(&mut ctx)
    }

    #[test]
    fn vector_returns_unit_vector_and_norms() -> VortexResult<()> {
        let result = evaluate(vector_array(2, &[3.0f64, 4.0, 0.0, 0.0])?)?;
        let normalized = result.unmasked_field_by_name("normalized")?;
        let norms: PrimitiveArray = result
            .unmasked_field_by_name("norms")?
            .clone()
            .execute(&mut SESSION.create_execution_ctx())?;

        assert!(normalized.dtype().as_extension().is::<AnyUnitVector>());
        assert_eq!(norms.as_slice::<f64>(), &[5.0, 0.0]);
        Ok(())
    }

    #[test]
    fn fixed_shape_tensor_keeps_its_dtype() -> VortexResult<()> {
        let input = tensor_array(&[2], &[3.0f64, 4.0])?;
        let expected = input.dtype().as_nonnullable();
        let result = evaluate(input)?;

        assert_eq!(
            result.unmasked_field_by_name("normalized")?.dtype(),
            &expected,
        );
        Ok(())
    }

    #[test]
    fn unit_vector_input_reports_physical_norm() -> VortexResult<()> {
        let vector = vector_array(2, &[0.6000005f32, 0.8])?;
        let mut ctx = SESSION.create_execution_ctx();
        let vector: ExtensionArray = vector.execute(&mut ctx)?;
        let unit = UnitVector::try_new_unit_vector_array(vector.storage_array().clone(), &mut ctx)?;
        let result = evaluate(unit)?;
        let norms: PrimitiveArray = result
            .unmasked_field_by_name("norms")?
            .clone()
            .execute(&mut ctx)?;

        assert_ne!(norms.as_slice::<f32>()[0], 1.0);
        assert!(
            result
                .unmasked_field_by_name("normalized")?
                .dtype()
                .as_extension()
                .is::<AnyUnitVector>()
        );
        Ok(())
    }

    #[test]
    fn serde_round_trip() -> VortexResult<()> {
        let child = vector_array(2, &[3.0f64, 4.0])?;
        let original = L2Normalize::try_new_array(child.clone())?.into_array();
        let plugin = ScalarFnArrayPlugin::new(L2Normalize);
        let metadata = plugin
            .serialize(&original, &SESSION)?
            .expect("L2Normalize must serialize metadata");
        let recovered = plugin.deserialize(
            original.dtype(),
            original.len(),
            &metadata,
            &[],
            &[child],
            &SESSION,
        )?;

        assert_eq!(recovered.dtype(), original.dtype());
        assert_eq!(recovered.encoding_id(), original.encoding_id());
        Ok(())
    }

    #[test]
    fn reports_value_dependent_errors() {
        assert!(L2Normalize.is_fallible(&EmptyOptions));
    }

    #[test]
    fn f16_output_satisfies_the_capped_unit_tolerance() -> VortexResult<()> {
        for dimensions in [128_u32, 768, 4096] {
            let length = usize::try_from(dimensions)?;
            let values = vec![f16::ONE; length];
            let result = evaluate(vector_array(dimensions, &values)?)?;
            let normalized = result.unmasked_field_by_name("normalized")?;

            assert!(normalized.dtype().as_extension().is::<AnyUnitVector>());
            validate_normalized_rows(normalized, None, &mut SESSION.create_execution_ctx())?;
        }
        Ok(())
    }

    #[test]
    fn scaled_fallback_handles_overflowing_sum_of_squares() -> VortexResult<()> {
        let value = f64::MAX / 2.0;
        let result = evaluate(vector_array(2, &[value, value])?)?;
        let normalized = result.unmasked_field_by_name("normalized")?;
        let norms: PrimitiveArray = result
            .unmasked_field_by_name("norms")?
            .clone()
            .execute(&mut SESSION.create_execution_ctx())?;

        validate_normalized_rows(normalized, None, &mut SESSION.create_execution_ctx())?;
        let norm = norms.as_slice::<f64>()[0];
        assert!(norm.is_finite());
        assert!(norm > value);
        Ok(())
    }

    #[test]
    fn scaled_fallback_handles_underflowing_sum_of_squares() -> VortexResult<()> {
        let value = f64::MIN_POSITIVE;
        let result = evaluate(vector_array(2, &[value, value])?)?;
        let normalized = result.unmasked_field_by_name("normalized")?;
        let norms: PrimitiveArray = result
            .unmasked_field_by_name("norms")?
            .clone()
            .execute(&mut SESSION.create_execution_ctx())?;

        validate_normalized_rows(normalized, None, &mut SESSION.create_execution_ctx())?;
        assert!(norms.as_slice::<f64>()[0] > 0.0);
        Ok(())
    }

    #[test]
    fn rejects_non_finite_norms() -> VortexResult<()> {
        let input = vector_array(2, &[f64::NAN, 0.0])?;

        assert!(evaluate(input).is_err());
        Ok(())
    }

    #[test]
    fn rejects_norms_that_do_not_fit_the_input_ptype() -> VortexResult<()> {
        let input = vector_array(2, &[f16::MAX, f16::MAX])?;

        assert!(evaluate(input).is_err());
        Ok(())
    }
}
