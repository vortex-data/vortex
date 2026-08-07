// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::CheckedMul;
use num_traits::NumCast;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::ReinterpretSink;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn;
use crate::array::ArrayView;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::ToI256;
use crate::dtype::i256;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

impl CastReduce for Primitive {
    fn cast(array: ArrayView<'_, Primitive>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only the same ptype is reducible without execution; type changes need the kernel
        // to verify values fit in the target range.
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };
        if *new_ptype != array.ptype() {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(*new_nullability, array.len())?
        else {
            return Ok(None);
        };

        // SAFETY: validity and data buffer still have same length.
        Ok(Some(unsafe {
            PrimitiveArray::new_unchecked_from_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                new_validity,
            )
            .into_array()
        }))
    }
}

impl CastKernel for Primitive {
    fn cast(
        array: ArrayView<'_, Primitive>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let DType::Decimal(decimal_dtype, nullability) = dtype {
            return cast_to_decimal(array, *decimal_dtype, *nullability, ctx).map(Some);
        }
        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };
        let (new_ptype, new_nullability) = (*new_ptype, *new_nullability);
        let src_ptype = array.ptype();

        let new_validity = array
            .validity()?
            .cast_nullability(new_nullability, array.len(), ctx)?;

        // Same bit representation: either the same ptype (only the nullability changed) or two
        // same-width integers (identical layout under 2's complement). The only non-trivial case
        // is the sign change between same-width ints, which still needs a value-range check.
        let same_rep = src_ptype == new_ptype
            || (src_ptype.is_int()
                && new_ptype.is_int()
                && src_ptype.byte_width() == new_ptype.byte_width());
        if same_rep {
            if !values_fit_in(array, new_ptype, ctx, true) {
                vortex_bail!(
                    Compute: "Cannot cast {} to {} — values exceed target range",
                    src_ptype, new_ptype,
                );
            }
            return Ok(Some(reinterpret(array, new_ptype, new_validity)));
        }

        // Different bit rep: cast each element. `cast_values` picks a pure or checked loop based
        // on whether the conversion is statically infallible.
        Ok(Some(match_each_native_ptype!(new_ptype, |T| {
            match_each_native_ptype!(src_ptype, |F| {
                cast_values::<F, T>(array, new_validity, ctx)?
            })
        })))
    }
}

fn cast_to_decimal(
    array: ArrayView<'_, Primitive>,
    decimal_dtype: DecimalDType,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if !array.ptype().is_int() {
        vortex_bail!(
            Compute: "Cannot cast floating primitive {} to decimal {}",
            array.ptype(), decimal_dtype
        );
    }

    let source_validity = array.validity()?;
    let validity = source_validity
        .clone()
        .cast_nullability(nullability, array.len(), ctx)?;
    let valid_values = source_validity.execute_mask(array.len(), ctx)?;
    let values_type = DecimalType::smallest_decimal_value_type(&decimal_dtype);

    match_each_integer_ptype!(array.ptype(), |S| {
        match_each_decimal_value_type!(values_type, |T| {
            cast_integer_values_to_decimal::<S, T>(array, decimal_dtype, validity, &valid_values)
        })
    })
}

fn cast_integer_values_to_decimal<S, T>(
    array: ArrayView<'_, Primitive>,
    decimal_dtype: DecimalDType,
    validity: Validity,
    valid_values: &Mask,
) -> VortexResult<ArrayRef>
where
    S: IntegerPType + ToI256,
    T: NativeDecimalType + CheckedMul,
{
    let scale = decimal_dtype.scale();
    let buffer = if scale == 0 {
        cast_unscaled_integer_values_to_decimal::<S, T>(array, decimal_dtype, valid_values)?
    } else if scale > 0 {
        cast_scaled_up_integer_values_to_decimal::<S, T>(array, decimal_dtype, valid_values)?
    } else {
        cast_scaled_down_integer_values_to_decimal::<S, T>(array, decimal_dtype, valid_values)?
    };

    Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
}

fn cast_unscaled_integer_values_to_decimal<S, T>(
    array: ArrayView<'_, Primitive>,
    decimal_dtype: DecimalDType,
    valid_values: &Mask,
) -> VortexResult<Buffer<T>>
where
    S: IntegerPType + ToI256,
    T: NativeDecimalType,
{
    let values = array.as_slice::<S>();
    cast_integer_values_to_decimal_buffer(values, decimal_dtype, valid_values, |value| {
        let value = <T as BigCast>::from(value)?;
        decimal_value_fits_precision(value, decimal_dtype).then_some(value)
    })
}

fn cast_scaled_up_integer_values_to_decimal<S, T>(
    array: ArrayView<'_, Primitive>,
    decimal_dtype: DecimalDType,
    valid_values: &Mask,
) -> VortexResult<Buffer<T>>
where
    S: IntegerPType + ToI256,
    T: NativeDecimalType + CheckedMul,
{
    let values = array.as_slice::<S>();
    let scale_factor = decimal_scale_factor::<T>(decimal_dtype.scale())?;
    cast_integer_values_to_decimal_buffer(values, decimal_dtype, valid_values, |value| {
        let value = <T as BigCast>::from(value)?;
        let value = value.checked_mul(&scale_factor)?;
        decimal_value_fits_precision(value, decimal_dtype).then_some(value)
    })
}

fn cast_scaled_down_integer_values_to_decimal<S, T>(
    array: ArrayView<'_, Primitive>,
    decimal_dtype: DecimalDType,
    valid_values: &Mask,
) -> VortexResult<Buffer<T>>
where
    S: IntegerPType + ToI256,
    T: NativeDecimalType,
{
    let values = array.as_slice::<S>();
    if decimal_dtype.scale().unsigned_abs() >= primitive_max_decimal_digits(array.ptype()) {
        // The scale factor exceeds every source value, so only zero can be exactly rescaled.
        return cast_integer_values_to_decimal_buffer(
            values,
            decimal_dtype,
            valid_values,
            |value| (value == S::default()).then_some(T::default()),
        );
    }

    // Scaling down can shrink a value into a narrower target type, so first select the smallest
    // signed carrier that can represent both the source and target.
    let carrier_type = primitive_decimal_carrier_type(array.ptype()).max(T::DECIMAL_TYPE);
    match_each_decimal_value_type!(carrier_type, |W| {
        let scale_factor = decimal_scale_factor::<W>(decimal_dtype.scale())?;
        cast_integer_values_to_decimal_buffer(values, decimal_dtype, valid_values, |value| {
            let value = <W as BigCast>::from(value)?;
            let value = (value % scale_factor == W::default()).then_some(value / scale_factor)?;
            let value = <T as BigCast>::from(value)?;
            decimal_value_fits_precision(value, decimal_dtype).then_some(value)
        })
    })
}

fn cast_integer_values_to_decimal_buffer<S, T>(
    values: &[S],
    decimal_dtype: DecimalDType,
    valid_values: &Mask,
    cast: impl FnMut(S) -> Option<T>,
) -> VortexResult<Buffer<T>>
where
    S: IntegerPType + ToI256,
    T: NativeDecimalType,
{
    cast_primitive_to_decimal_buffer(values, valid_values, cast)
        .map_err(|idx| primitive_to_decimal_cast_error(values[idx], decimal_dtype))
}

fn primitive_decimal_carrier_type(ptype: PType) -> DecimalType {
    match ptype {
        PType::I8 => DecimalType::I8,
        PType::U8 | PType::I16 => DecimalType::I16,
        PType::U16 | PType::I32 => DecimalType::I32,
        PType::U32 | PType::I64 => DecimalType::I64,
        PType::U64 => DecimalType::I128,
        PType::F16 | PType::F32 | PType::F64 => {
            unreachable!("floating primitives are rejected before selecting a decimal carrier")
        }
    }
}

fn primitive_max_decimal_digits(ptype: PType) -> u8 {
    match ptype {
        PType::U8 | PType::I8 => 3,
        PType::U16 | PType::I16 => 5,
        PType::U32 | PType::I32 => 10,
        PType::U64 => 20,
        PType::I64 => 19,
        PType::F16 | PType::F32 | PType::F64 => {
            unreachable!("floating primitives are rejected before inspecting decimal digits")
        }
    }
}

fn decimal_scale_factor<T>(scale: i8) -> VortexResult<T>
where
    T: NativeDecimalType + CheckedMul,
{
    let exponent = if scale > 0 {
        scale as u32
    } else {
        (-(scale as i16)) as u32
    };
    let ten = <T as BigCast>::from(10i8).ok_or_else(
        || vortex_err!(Compute: "Cannot create decimal scale factor for scale {scale}"),
    )?;
    let mut factor = <T as BigCast>::from(1i8).ok_or_else(
        || vortex_err!(Compute: "Cannot create decimal scale factor for scale {scale}"),
    )?;
    for _ in 0..exponent {
        factor = factor.checked_mul(&ten).ok_or_else(
            || vortex_err!(Compute: "Cannot create decimal scale factor for scale {scale}"),
        )?;
    }
    Ok(factor)
}

fn decimal_value_fits_precision<T: NativeDecimalType>(
    value: T,
    decimal_dtype: DecimalDType,
) -> bool {
    let precision = decimal_dtype.precision() as usize;
    value >= T::MIN_BY_PRECISION[precision] && value <= T::MAX_BY_PRECISION[precision]
}

fn cast_primitive_to_decimal_buffer<S, T>(
    values: &[S],
    valid_values: &Mask,
    mut cast: impl FnMut(S) -> Option<T>,
) -> Result<Buffer<T>, usize>
where
    S: NativePType,
    T: NativeDecimalType,
{
    if matches!(valid_values, Mask::AllFalse(_)) {
        return Ok(BufferMut::<T>::zeroed(values.len()).freeze());
    }

    let mut buffer = BufferMut::<T>::with_capacity(values.len());
    match valid_values {
        Mask::AllTrue(_) => {
            values.try_map_into(&mut buffer.spare_capacity_mut()[..values.len()], &mut cast)?;
        }
        Mask::Values(mask) => {
            values.try_map_masked_into(
                mask.bit_buffer(),
                &mut buffer.spare_capacity_mut()[..values.len()],
                &mut cast,
            )?;
        }
        Mask::AllFalse(_) => unreachable!("all-null values are handled before allocating"),
    }
    // SAFETY: the selected map kernel initialized every lane before returning Ok.
    unsafe { buffer.set_len(values.len()) };
    Ok(buffer.freeze())
}

fn primitive_to_decimal_cast_error<S>(value: S, decimal_dtype: DecimalDType) -> VortexError
where
    S: IntegerPType + ToI256,
{
    let Some(value) = <i256 as BigCast>::from(value) else {
        return vortex_err!(
            Compute: "primitive value cannot be represented while casting to {}",
            decimal_dtype
        );
    };

    match DecimalValue::rescale_i256(value, 0, decimal_dtype.scale())
        .and_then(|value| DecimalValue::try_from_i256(value, decimal_dtype))
    {
        Err(error) => error,
        Ok(_) => {
            debug_assert!(
                false,
                "primitive-to-decimal fast path rejected a value that the scalar cast accepts"
            );
            vortex_err!(
                Compute: "primitive value cannot be represented while casting to {}",
                decimal_dtype
            )
        }
    }
}

/// Cast Primitive values from `F` to `T`.
fn cast_values<F, T>(
    array: ArrayView<'_, Primitive>,
    new_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    let overflow = || {
        vortex_err!(
            Compute: "Cannot cast {} to {} — value exceeds target range",
            F::PTYPE, T::PTYPE,
        )
    };

    // Returns `true` if every value of `from` is representable in `to` without loss.
    fn casts_losslessly_to(from: PType, to: PType) -> bool {
        from.least_supertype(to) == Some(to)
    }

    // Skip the fallible kernel when type widening or (cached) min/max prove every value fits.
    let target_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
    let infallible = casts_losslessly_to(F::PTYPE, T::PTYPE)
        || cached_values_fit_in(array, &target_dtype).unwrap_or(false);

    let len = array.len();

    // If F and T have the same byte width, try to take unique ownership of the buffer.
    let same_bit_width = F::PTYPE.byte_width() == T::PTYPE.byte_width();
    let owned: Option<BufferMut<F>> = same_bit_width
        .then(|| array.into_owned().try_into_buffer_mut::<F>().ok())
        .flatten();
    let values: &[F] = array.as_slice::<F>();

    if infallible {
        return match owned {
            Some(mut buf) => {
                ReinterpretSink::<F, T>::new(buf.as_mut_slice()).map_into_in_place(|v: F| v.as_());
                // SAFETY: same size + alignment for NativePType
                let result: BufferMut<T> = unsafe { buf.transmute::<T>() };
                Ok(PrimitiveArray::new(result.freeze(), new_validity).into_array())
            }
            None => {
                let mut buffer = BufferMut::<T>::with_capacity(len);
                values.map_into(&mut buffer.spare_capacity_mut()[..len], |v| v.as_());
                // SAFETY: map_into initializes every lane.
                unsafe { buffer.set_len(len) };
                Ok(PrimitiveArray::new(buffer.freeze(), new_validity).into_array())
            }
        };
    }

    let mask = array.validity()?.execute_mask(len, ctx)?;

    let buffer: Buffer<T> = match (&mask, owned) {
        (Mask::AllTrue(_), Some(mut buf)) => {
            ReinterpretSink::<F, T>::new(buf.as_mut_slice())
                .try_map_in_place(|v: F| <T as NumCast>::from(v))
                .map_err(|_| overflow())?;
            // SAFETY: same size + alignment for NativePType
            let result: BufferMut<T> = unsafe { buf.transmute::<T>() };
            result.freeze()
        }
        (Mask::AllTrue(_), None) => {
            let mut buffer = BufferMut::<T>::with_capacity(len);
            values
                .try_map_into(&mut buffer.spare_capacity_mut()[..len], |v| {
                    <T as NumCast>::from(v)
                })
                .map_err(|_| overflow())?;
            // SAFETY: initialized every lane.
            unsafe { buffer.set_len(len) };
            buffer.freeze()
        }
        (Mask::AllFalse(_), _) => BufferMut::<T>::zeroed(len).freeze(),
        (Mask::Values(m), Some(mut buf)) => {
            ReinterpretSink::<F, T>::new(buf.as_mut_slice())
                .try_map_masked_in_place(m.bit_buffer(), |v: F| <T as NumCast>::from(v))
                .map_err(|_| overflow())?;
            // SAFETY: same size + alignment for NativePType
            let result: BufferMut<T> = unsafe { buf.transmute::<T>() };
            result.freeze()
        }
        (Mask::Values(m), None) => {
            let mut buffer = BufferMut::<T>::with_capacity(len);
            values
                .try_map_masked_into(
                    m.bit_buffer(),
                    &mut buffer.spare_capacity_mut()[..len],
                    |v| <T as NumCast>::from(v),
                )
                .map_err(|_| overflow())?;
            // SAFETY: initialized every lane.
            unsafe { buffer.set_len(len) };
            buffer.freeze()
        }
    };

    Ok(PrimitiveArray::new(buffer, new_validity).into_array())
}

fn reinterpret(
    array: ArrayView<'_, Primitive>,
    new_ptype: PType,
    new_validity: Validity,
) -> ArrayRef {
    // SAFETY: caller has verified the bit representation is compatible and that validity length
    // still matches the buffer length.
    unsafe {
        PrimitiveArray::new_unchecked_from_handle(
            array.buffer_handle().clone(),
            new_ptype,
            new_validity,
        )
    }
    .into_array()
}

/// Returns `true` if all valid values in `array` are representable as `target_ptype`.
///
/// Cached min/max statistics are consulted first. If either bound is missing, the function either
/// computes them with a single pass (when `compute` is `true`) or returns `false` so the caller
/// can fall back to a slower path (when `compute` is `false`).
fn values_fit_in(
    array: ArrayView<'_, Primitive>,
    target_ptype: PType,
    ctx: &mut ExecutionCtx,
    compute: bool,
) -> bool {
    let target_dtype = DType::Primitive(target_ptype, Nullability::NonNullable);
    if let Some(fits) = cached_values_fit_in(array, &target_dtype) {
        return fits;
    }
    if !compute {
        return false;
    }
    aggregate_fn::fns::min_max::min_max(
        array.array(),
        ctx,
        aggregate_fn::NumericalAggregateOpts::default(),
    )
    .ok()
    .flatten()
    .is_none_or(|mm| mm.min.cast(&target_dtype).is_ok() && mm.max.cast(&target_dtype).is_ok())
}

/// Cached-only check: returns `Some(fits)` if both `Min` and `Max` are present as `Exact` in the
/// stats cache, otherwise `None`.
fn cached_values_fit_in(array: ArrayView<'_, Primitive>, target_dtype: &DType) -> Option<bool> {
    let stats = array.array().statistics();
    let min = stats.get(Stat::Min).as_exact()?;
    let max = stats.get(Stat::Max).as_exact()?;
    Some(min.cast(target_dtype).is_ok() && max.cast(target_dtype).is_ok())
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::validity::Validity;

    #[test]
    fn cast_u32_u8() {
        let mut ctx = array_session().create_execution_ctx();
        let arr = buffer![0u32, 10, 200].into_array();

        // cast from u32 to u8
        let p = arr
            .cast(PType::U8.into())
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]), &mut ctx);
        assert!(matches!(p.validity(), Ok(Validity::NonNullable)));

        // to nullable
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u8, 10, 200], Validity::AllValid),
            &mut ctx
        );
        assert!(matches!(p.validity(), Ok(Validity::AllValid)));

        // back to non-nullable
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]), &mut ctx);
        assert!(matches!(p.validity(), Ok(Validity::NonNullable)));

        // to nullable u32
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u32, 10, 200], Validity::AllValid),
            &mut ctx
        );
        assert!(matches!(p.validity(), Ok(Validity::AllValid)));

        // to non-nullable u8
        let p = p
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]), &mut ctx);
        assert!(matches!(p.validity(), Ok(Validity::NonNullable)));
    }

    #[test]
    fn cast_u32_f32() {
        let mut ctx = array_session().create_execution_ctx();
        let arr = buffer![0u32, 10, 200].into_array();
        let u8arr = arr
            .cast(PType::F32.into())
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            u8arr,
            PrimitiveArray::from_iter([0.0f32, 10., 200.]),
            &mut ctx
        );
    }

    #[test]
    fn cast_integer_to_decimal_rescales() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(5, 2);
        let casted = PrimitiveArray::from_iter([42i32, -7])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.values_type(), DecimalType::I32);
        assert_eq!(casted.buffer::<i32>().as_ref(), &[4_200, -700]);
        Ok(())
    }

    #[test]
    fn cast_u64_to_decimal() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(20, 0);
        let casted = PrimitiveArray::from_iter([u64::MAX])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.values_type(), DecimalType::I128);
        assert_eq!(casted.buffer::<i128>().as_ref(), &[i128::from(u64::MAX)]);
        Ok(())
    }

    #[test]
    fn cast_u8_to_negative_scale_decimal() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(2, -2);
        let casted = PrimitiveArray::from_iter([200u8])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.values_type(), DecimalType::I8);
        assert_eq!(casted.buffer::<i8>().as_ref(), &[2]);
        Ok(())
    }

    #[test]
    fn cast_u64_to_negative_scale_decimal() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(1, -19);
        let casted = PrimitiveArray::from_iter([10_000_000_000_000_000_000u64])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.values_type(), DecimalType::I8);
        assert_eq!(casted.buffer::<i8>().as_ref(), &[1]);
        Ok(())
    }

    #[test]
    fn cast_integer_to_i256_decimal() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(39, 2);
        let casted = PrimitiveArray::from_iter([42i64])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.values_type(), DecimalType::I256);
        assert_eq!(casted.buffer::<i256>().as_ref(), &[i256::from_i128(4_200)]);
        Ok(())
    }

    #[test]
    fn cast_all_null_integer_to_decimal() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(39, 2);
        let casted = PrimitiveArray::new(buffer![i64::MAX, i64::MIN], Validity::AllInvalid)
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert!(matches!(casted.validity(), Ok(Validity::AllInvalid)));
        assert_eq!(casted.buffer::<i256>().as_ref(), &[i256::ZERO, i256::ZERO]);
        Ok(())
    }

    #[test]
    fn cast_integer_to_negative_scale_decimal_requires_exact_rescale()
    -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, -2);
        let casted = PrimitiveArray::from_iter([1_200i32, -500])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.buffer::<i16>().as_ref(), &[12, -5]);

        let error = PrimitiveArray::from_iter([42i32])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)
            .unwrap_err();
        assert!(error.to_string().contains("would lose precision"));
        Ok(())
    }

    #[test]
    fn cast_zero_to_large_negative_scale_decimal() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, -128);
        let casted = PrimitiveArray::from_iter([0i32])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.buffer::<i16>().as_ref(), &[0]);
        Ok(())
    }

    #[test]
    fn cast_integer_to_decimal_ignores_out_of_range_null_lanes() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, 1);
        let casted = PrimitiveArray::new(buffer![999i32, 42], Validity::from_iter([false, true]))
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.buffer::<i16>().as_ref()[1], 420);
        assert_eq!(
            casted.validity()?.execute_mask(casted.len(), &mut ctx)?,
            Mask::from(BitBuffer::from(vec![false, true]))
        );
        Ok(())
    }

    #[test]
    fn cast_integer_to_decimal_checks_precision() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let casted = PrimitiveArray::from_iter([100i32])
            .into_array()
            .cast(DType::Decimal(
                DecimalDType::new(2, 1),
                Nullability::NonNullable,
            ))?;

        let error = casted.execute::<DecimalArray>(&mut ctx).unwrap_err();
        assert!(error.to_string().contains("does not fit in precision"));
        Ok(())
    }

    #[test]
    fn cast_floating_primitive_to_decimal_fails() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let casted = PrimitiveArray::from_iter([1.0f64])
            .into_array()
            .cast(DType::Decimal(
                DecimalDType::new(3, 1),
                Nullability::NonNullable,
            ))?;

        let error = casted.execute::<DecimalArray>(&mut ctx).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Cannot cast floating primitive f64 to decimal decimal(3,1)")
        );
        Ok(())
    }

    #[test]
    fn cast_i32_u32() {
        let arr = buffer![-1i32].into_array();
        #[expect(deprecated)]
        let error = arr
            .cast(PType::U32.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(error, VortexError::Compute(..)));
        assert!(error.to_string().contains("values exceed target range"));
    }

    #[test]
    fn cast_array_with_nulls_to_nonnullable() {
        let arr = PrimitiveArray::from_option_iter([Some(-1i32), None, Some(10)]);
        #[expect(deprecated)]
        let err = arr
            .into_array()
            .cast(PType::I32.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();

        assert!(matches!(err, VortexError::InvalidArgument(..)));
        assert!(
            err.to_string()
                .contains("Cannot cast array with invalid values to non-nullable type.")
        );
    }

    #[test]
    fn cast_with_invalid_nulls() {
        let mut ctx = array_session().create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![-1i32, 0, 10],
            Validity::from_iter([false, true, true]),
        );
        let p = arr
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            p,
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(10)]),
            &mut ctx
        );
        assert_eq!(
            p.as_ref()
                .validity()
                .unwrap()
                .execute_mask(
                    p.as_ref().len(),
                    &mut array_session().create_execution_ctx()
                )
                .unwrap(),
            Mask::from(BitBuffer::from(vec![false, true, true]))
        );
    }

    /// Same-width integer cast where all values fit: should reinterpret the
    /// buffer without allocation (pointer identity).
    #[test]
    fn cast_same_width_int_reinterprets_buffer() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let src = PrimitiveArray::from_iter([0u32, 10, 100]);
        let src_ptr = src.as_slice::<u32>().as_ptr();

        let dst = src
            .into_array()
            .cast(PType::I32.into())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let dst_ptr = dst.as_slice::<i32>().as_ptr();

        // Zero-copy: the data pointer should be identical.
        assert_eq!(src_ptr as usize, dst_ptr as usize);
        assert_arrays_eq!(dst, PrimitiveArray::from_iter([0i32, 10, 100]), &mut ctx);
        Ok(())
    }

    /// Same-width integer cast where values don't fit: should fall through
    /// to the allocating path and produce an error.
    #[test]
    fn cast_same_width_int_out_of_range_errors() {
        let arr = buffer![u32::MAX].into_array();
        #[expect(deprecated)]
        let err = arr
            .cast(PType::I32.into())
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap_err();
        assert!(matches!(err, VortexError::Compute(..)));
    }

    /// All-null array cast between same-width types should succeed without
    /// touching the buffer contents.
    #[test]
    fn cast_same_width_all_null() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = PrimitiveArray::new(buffer![0xFFu8, 0xFF], Validity::AllInvalid);
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I8, Nullability::Nullable))?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(casted.len(), 2);
        assert!(matches!(casted.validity(), Ok(Validity::AllInvalid)));
        Ok(())
    }

    /// Same-width integer cast with nullable values: out-of-range nulls should
    /// not prevent the cast from succeeding.
    #[test]
    fn cast_same_width_int_nullable_with_out_of_range_nulls() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // The null position holds u32::MAX which doesn't fit in i32, but it's
        // masked as invalid so the cast should still succeed via reinterpret.
        let arr = PrimitiveArray::new(
            buffer![u32::MAX, 0u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(0i32), Some(42)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_u32_to_u8_with_out_of_range_nulls() -> vortex_error::VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![1000u32, 10u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(10u8), Some(42)]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case(buffer![0u8, 1, 2, 3, 255].into_array())]
    #[case(buffer![0u16, 100, 1000, 65535].into_array())]
    #[case(buffer![0u32, 100, 1000, 1000000].into_array())]
    #[case(buffer![0u64, 100, 1000, 1000000000].into_array())]
    #[case(buffer![-128i8, -1, 0, 1, 127].into_array())]
    #[case(buffer![-1000i16, -1, 0, 1, 1000].into_array())]
    #[case(buffer![-1000000i32, -1, 0, 1, 1000000].into_array())]
    #[case(buffer![-1000000000i64, -1, 0, 1, 1000000000].into_array())]
    #[case(buffer![0.0f32, 1.5, -2.5, 100.0, 1e6].into_array())]
    #[case(buffer![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0f32].into_array())]
    #[case(buffer![0.0f64, 1.5, -2.5, 100.0, 1e12].into_array())]
    #[case(buffer![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0f64].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1u8), None, Some(255), Some(0), None]).into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(-100), Some(0), None]).into_array())]
    #[case(buffer![42u32].into_array())]
    fn test_cast_primitive_conformance(#[case] array: ArrayRef) {
        test_cast_conformance(&array, &mut array_session().create_execution_ctx());
    }
}
