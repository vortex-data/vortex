// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::slice;

use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::primitive::compute::integer_cast;
use crate::arrays::primitive::compute::integer_cast::IntegerCast;
use crate::arrays::primitive::compute::integer_cast::IntegerType;
use crate::arrays::primitive::compute::integer_cast::Remainder;
use crate::arrays::primitive::compute::integer_cast::all_valid_lanes;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::dtype::i256;
use crate::match_each_float_ptype;
use crate::match_each_integer_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::cast::CastKernel;
use crate::scalar_fn::builtin::cast::CastReduce;
use crate::stats::Precision;
use crate::stats::Stat;
use crate::stats::StatsProvider;
use crate::stats::StatsProviderExt;
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
        // same-width integers (identical layout under 2's complement).
        let reinterprets = src_ptype == new_ptype
            || (src_ptype.is_int()
                && new_ptype.is_int()
                && src_ptype.byte_width() == new_ptype.byte_width());
        let all_null = || {
            ConstantArray::new(
                Scalar::null(DType::Primitive(new_ptype, Nullability::Nullable)),
                array.len(),
            )
            .into_array()
        };

        // Only integer targets can reject a value: every float target accepts every value, with
        // finite values rounding or saturating to infinity. The accepted values of an integer
        // target form a single range, so checking the min and max checks every value. Once the
        // values are known to fit, the conversion below is the infallible `as` cast.
        let target_dtype = DType::Primitive(new_ptype, Nullability::NonNullable);
        if new_ptype.is_int()
            && !src_ptype.casts_losslessly_to(new_ptype)
            && !cached_values_fit_in(array, &target_dtype).unwrap_or(false)
        {
            let mask = array.validity()?.execute_mask(array.len(), ctx)?;
            if !reinterprets && !array.is_empty() && matches!(mask, Mask::AllFalse(_)) {
                return Ok(Some(all_null()));
            }
            if !values_fit_in(array, new_ptype, &mask) {
                vortex_bail!(
                    "Cannot cast {} to {} — values exceed target range",
                    src_ptype,
                    new_ptype,
                );
            }
        }

        if reinterprets {
            return Ok(Some(reinterpret(array, new_ptype, new_validity)));
        }
        // Converting null values would only produce more nulls.
        if !array.is_empty() && matches!(array.validity()?, Validity::AllInvalid) {
            return Ok(Some(all_null()));
        }

        let values = convert_values(array, new_ptype);
        Ok(Some(
            PrimitiveArray::from_byte_buffer(values, new_ptype, new_validity).into_array(),
        ))
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
            "Cannot cast floating primitive {} to decimal {}",
            array.ptype(),
            decimal_dtype
        );
    }

    let source_validity = array.validity()?;
    let validity = source_validity
        .clone()
        .cast_nullability(nullability, array.len(), ctx)?;
    let values_type = decimal_dtype.canonical_values_type();
    let cast = IntegerCast::to_decimal(IntegerType::of_ptype(array.ptype()), 0, decimal_dtype);
    let values = array.buffer_handle().as_host();

    let error = |index| {
        primitive_to_decimal_cast_error(cast.source_type().value_at(values, index), decimal_dtype)
    };

    // An unscaled cast from a signed integer to decimals of the same width keeps the values buffer.
    if decimal_dtype.scale() == 0
        && signed_primitive_decimal_type(array.ptype()) == Some(values_type)
    {
        let target_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        if !cached_values_fit_in(array, &target_dtype).unwrap_or(false) {
            let valid_values = source_validity.execute_mask(array.len(), ctx)?;
            if let Some(index) = cast.first_misfit(values, &valid_values) {
                return Err(error(index));
            }
        }
        // SAFETY: the decimal values type has the same physical representation as the source
        // ptype, and either exact min/max statistics or the validation above prove every valid
        // value fits.
        return Ok(unsafe {
            DecimalArray::new_unchecked_handle(
                array.buffer_handle().clone(),
                values_type,
                decimal_dtype,
                validity,
            )
            .into_array()
        });
    }

    let valid_values = source_validity.execute_mask(array.len(), ctx)?;
    if !array.is_empty() && matches!(valid_values, Mask::AllFalse(_)) {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Decimal(decimal_dtype, nullability)),
            array.len(),
        )
        .into_array());
    }
    let casted = cast.cast(values, &valid_values).map_err(error)?;
    // Every valid value has been checked against the precision, so skip the constructor's scan.
    Ok(DecimalArray::try_new_handle_from_trusted_values(
        BufferHandle::new_host(casted),
        values_type,
        decimal_dtype,
        validity,
    )?
    .into_array())
}

/// Decimal physical type with the same representation, when `ptype` is signed.
fn signed_primitive_decimal_type(ptype: PType) -> Option<DecimalType> {
    match ptype {
        PType::I8 => Some(DecimalType::I8),
        PType::I16 => Some(DecimalType::I16),
        PType::I32 => Some(DecimalType::I32),
        PType::I64 => Some(DecimalType::I64),
        PType::U8 | PType::U16 | PType::U32 | PType::U64 => None,
        PType::F16 | PType::F32 | PType::F64 => {
            unreachable!("floating primitives are rejected before selecting a decimal type")
        }
    }
}

#[cold]
#[inline(never)]
fn primitive_to_decimal_cast_error(value: i256, decimal_dtype: DecimalDType) -> VortexError {
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
                "primitive value cannot be represented while casting to {}",
                decimal_dtype
            )
        }
    }
}

/// Converts every value of `array` to `to` with `as` semantics and returns the raw values.
///
/// The caller has already checked that every valid value is representable in `to`. Null lanes
/// may hold any value; `as` is defined for every input, so they convert to unspecified values.
///
/// Integer-to-integer conversions, the bulk of the (source, target) pairs, share the kernels of
/// [`integer_cast::resize`], which only depend on the widths and the source's signedness.
///
/// Conversions to or from floats use one kernel per pair. Composing them through an integer
/// intermediate would share kernels too, but measured 50-60% slower than converting directly.
fn convert_values(array: ArrayView<'_, Primitive>, to: PType) -> ByteBuffer {
    let from = array.ptype();
    match (from.is_float(), to.is_float()) {
        (false, false) => integer_cast::resize(
            array.buffer_handle().as_host(),
            IntegerType::of_ptype(from),
            IntegerType::of_ptype(to),
        ),
        (false, true) => int_to_float(array, to),
        (true, false) => float_to_int(array, to),
        (true, true) => match (from, to) {
            (PType::F16, PType::F32) => convert_buffer::<f16, f32>(values_as(array)),
            (PType::F16, PType::F64) => convert_buffer::<f16, f64>(values_as(array)),
            (PType::F32, PType::F16) => convert_buffer::<f32, f16>(values_as(array)),
            (PType::F32, PType::F64) => convert_buffer::<f32, f64>(values_as(array)),
            (PType::F64, PType::F16) => convert_buffer::<f64, f16>(values_as(array)),
            (PType::F64, PType::F32) => convert_buffer::<f64, f32>(values_as(array)),
            _ => unreachable!("float conversion from {from} to {to}"),
        },
    }
}

fn int_to_float(array: ArrayView<'_, Primitive>, to: PType) -> ByteBuffer {
    match_each_float_ptype!(to, |T| {
        match_each_integer_ptype!(array.ptype(), |F| {
            convert_buffer::<F, T>(values_as(array))
        })
    })
}

fn float_to_int(array: ArrayView<'_, Primitive>, to: PType) -> ByteBuffer {
    match_each_float_ptype!(array.ptype(), |F| {
        match_each_integer_ptype!(to, |T| { convert_buffer::<F, T>(values_as(array)) })
    })
}

/// Converts `src` to `B`.
fn convert_buffer<A, B>(src: &[A]) -> ByteBuffer
where
    A: AsPrimitive<B>,
    B: Copy + 'static,
{
    let len = src.len();
    let mut out = BufferMut::<B>::with_capacity(len);
    convert::<A, B>(src, &mut out.spare_capacity_mut()[..len]);
    // SAFETY: `convert` initialized every lane.
    unsafe { out.set_len(len) };
    out.freeze().into_byte_buffer()
}

/// The conversion kernel, kept out of line so that each (`A`, `B`) pair is compiled once however
/// many target types share it.
#[inline(never)]
fn convert<A, B>(src: &[A], dst: &mut [MaybeUninit<B>])
where
    A: AsPrimitive<B>,
    B: Copy + 'static,
{
    src.map_into(dst, |v| v.as_());
}

/// The values of `array` as `T`, which has the same width as the array's ptype.
fn values_as<T: NativePType>(array: ArrayView<'_, Primitive>) -> &[T] {
    assert_eq!(
        T::PTYPE.byte_width(),
        array.ptype().byte_width(),
        "can only view values as a type of the same width"
    );
    let bytes = array.buffer_handle().as_host();
    // SAFETY: the buffer is aligned to the array's ptype, whose width and alignment `T` shares,
    // and every bit pattern is a valid `T`.
    unsafe { slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>()) }
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

/// Returns `true` if every valid value in `array` is representable as the integer `target_ptype`,
/// using the same rules as [`num_traits::NumCast`]: floats truncate toward zero, and NaN and
/// infinities never fit. Only the values that `mask` marks valid are checked.
///
/// The check is compiled once per source type, with the target's range passed at runtime.
fn values_fit_in(array: ArrayView<'_, Primitive>, target_ptype: PType, mask: &Mask) -> bool {
    if array.ptype().is_float() {
        // Floats truncate toward zero, so the values that fit lie strictly between
        // `target_min - 1` and `target_max + 1`.
        let (lower, upper) = float_exclusive_bounds(target_ptype);
        return match array.ptype() {
            PType::F16 => floats_fit_f32(array.as_slice::<f16>(), mask, lower, upper),
            PType::F32 => floats_fit_f32(array.as_slice::<f32>(), mask, lower, upper),
            _ => all_valid_lanes(array.as_slice::<f64>(), mask, |v| (lower < v) & (v < upper)),
        };
    }
    let cast = IntegerCast::new(
        IntegerType::of_ptype(array.ptype()),
        IntegerType::of_ptype(target_ptype),
        0,
        Remainder::Reject,
    );
    cast.first_misfit(array.buffer_handle().as_host(), mask)
        .is_none()
}

/// Checks f16 or f32 values against the exclusive f64 bounds in f32, which vectorizes better
/// than widening every value to f64.
///
/// Rounding each bound outward to f32 keeps the check exact: no f32 lies between a bound and its
/// rounding, so a value is above the rounded lower bound exactly when it is above the original.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bounds are rounded to f32 on purpose, and corrected outward"
)]
fn floats_fit_f32<F: AsPrimitive<f32>>(values: &[F], mask: &Mask, lower: f64, upper: f64) -> bool {
    let mut lower32 = lower as f32;
    if <f64 as From<f32>>::from(lower32) > lower {
        lower32 = lower32.next_down();
    }
    let mut upper32 = upper as f32;
    if <f64 as From<f32>>::from(upper32) < upper {
        upper32 = upper32.next_up();
    }
    all_valid_lanes(values, mask, |v| {
        let v: f32 = v.as_();
        (lower32 < v) & (v < upper32)
    })
}

/// The inclusive range of the integer `ptype`.
fn integer_range(ptype: PType) -> (i128, i128) {
    match ptype {
        PType::U8 => (0, u8::MAX.into()),
        PType::U16 => (0, u16::MAX.into()),
        PType::U32 => (0, u32::MAX.into()),
        PType::U64 => (0, u64::MAX.into()),
        PType::I8 => (i8::MIN.into(), i8::MAX.into()),
        PType::I16 => (i16::MIN.into(), i16::MAX.into()),
        PType::I32 => (i32::MIN.into(), i32::MAX.into()),
        PType::I64 => (i64::MIN.into(), i64::MAX.into()),
        _ => unreachable!("{ptype} is not an integer type"),
    }
}

/// The f64 bounds that a float must lie strictly between to truncate into the integer `ptype`.
///
/// The bounds are `min - 1` and `max + 1`, which f64 holds exactly for every integer type but
/// `i64`. There, `min - 1` is replaced by the next f64 below `i64::MIN`, which excludes the same
/// floats.
#[expect(
    clippy::cast_precision_loss,
    reason = "f64 holds integers below 64 bits exactly, and u64::MAX rounds up to exactly max + 1"
)]
fn float_exclusive_bounds(ptype: PType) -> (f64, f64) {
    match ptype {
        PType::I64 => (i64::MIN as f64 - 2048.0, -(i64::MIN as f64)),
        PType::U64 => (-1.0, u64::MAX as f64),
        _ => {
            let (min, max) = integer_range(ptype);
            (min as f64 - 1.0, max as f64 + 1.0)
        }
    }
}

/// Returns whether every valid value fits `target_dtype` if exact statistics can answer it.
///
/// Cached min and max skip NaN, so for floats they only answer once the NaN count is known.
fn cached_values_fit_in(array: ArrayView<'_, Primitive>, target_dtype: &DType) -> Option<bool> {
    let stats = array.array().statistics();
    if array.ptype().is_float() {
        match stats.get_as::<u64>(Stat::NaNCount) {
            Precision::Exact(0) => {}
            Precision::Exact(_) => return Some(false),
            _ => return None,
        }
    }
    let min = stats.get(Stat::Min).as_exact()?;
    let max = stats.get(Stat::Max).as_exact()?;
    Some(min.cast(target_dtype).is_ok() && max.cast(target_dtype).is_ok())
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Alignment;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::Constant;
    use crate::arrays::DecimalArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::match_each_decimal_value_type;
    use crate::match_each_native_ptype;
    use crate::scalar_fn::builtin::cast::CastKernel;
    use crate::stats::Stat;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn cast_u32_u8() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let mut ctx = SESSION.create_execution_ctx();
        let arr = buffer![0u32, 10, 200].into_array();
        let u8arr = arr.cast(PType::F32.into()).unwrap();
        assert_arrays_eq!(
            u8arr,
            PrimitiveArray::from_iter([0.0f32, 10., 200.]),
            &mut ctx
        );
    }

    #[test]
    fn cast_integer_to_decimal_rescales() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
    fn cast_same_width_signed_integer_to_decimal_reuses_buffer() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_iter([42i32, -7]);
        let source_ptr = source.as_slice::<i32>().as_ptr();
        let casted = source
            .into_array()
            .cast(DType::Decimal(
                DecimalDType::new(9, 0),
                Nullability::NonNullable,
            ))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.buffer::<i32>().as_ptr(), source_ptr);
        assert_eq!(casted.buffer::<i32>().as_ref(), &[42, -7]);
        assert_arrays_eq!(
            &casted,
            DecimalArray::from_iter([42i32, -7], DecimalDType::new(9, 0)),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_same_width_signed_integer_to_decimal_reuses_buffer_with_cached_bounds()
    -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_iter([42i32, -7]);
        let source_ptr = source.as_slice::<i32>().as_ptr();
        let source = source.into_array();
        source
            .statistics()
            .compute_all(&[Stat::Min, Stat::Max], &mut ctx)?;
        let casted = source
            .cast(DType::Decimal(
                DecimalDType::new(9, 0),
                Nullability::NonNullable,
            ))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.buffer::<i32>().as_ptr(), source_ptr);
        assert_arrays_eq!(
            &casted,
            DecimalArray::from_iter([42i32, -7], DecimalDType::new(9, 0)),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_same_width_signed_integer_to_decimal_ignores_out_of_range_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::new(buffer![i32::MAX, 42], Validity::from_iter([false, true]));
        let source_ptr = source.as_slice::<i32>().as_ptr();
        let casted = source
            .into_array()
            .cast(DType::Decimal(
                DecimalDType::new(9, 0),
                Nullability::Nullable,
            ))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.buffer::<i32>().as_ptr(), source_ptr);
        assert_eq!(
            casted.validity()?.execute_mask(casted.len(), &mut ctx)?,
            Mask::from(BitBuffer::from(vec![false, true]))
        );
        assert_arrays_eq!(
            &casted,
            DecimalArray::from_option_iter([None, Some(42i32)], DecimalDType::new(9, 0)),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_same_width_signed_integer_to_decimal_checks_precision() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let casted = PrimitiveArray::from_iter([i32::MAX])
            .into_array()
            .cast(DType::Decimal(
                DecimalDType::new(9, 0),
                Nullability::NonNullable,
            ))?;

        let error = casted.execute::<DecimalArray>(&mut ctx).unwrap_err();
        assert!(error.to_string().contains("does not fit in precision"));
        Ok(())
    }

    #[test]
    fn cast_u64_to_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(20, 0);
        let casted = PrimitiveArray::from_iter([u64::MAX])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.values_type(), DecimalType::I128);
        assert_eq!(casted.buffer::<i128>().as_ref(), &[i128::from(u64::MAX)]);
        Ok(())
    }

    #[test]
    fn cast_integer_to_decimal_reports_scale_up_overflow() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Decimal(DecimalDType::new(38, 20), Nullability::NonNullable);
        let casted = PrimitiveArray::from_iter([u64::MAX])
            .into_array()
            .cast(dtype)?;
        let actual = casted.execute::<DecimalArray>(&mut ctx).unwrap_err();

        assert!(
            actual
                .to_string()
                .contains("does not fit in precision of decimal(38,20)")
        );
        Ok(())
    }

    #[test]
    fn cast_u8_to_negative_scale_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(2, -2);
        let casted = PrimitiveArray::from_iter([200u8])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.values_type(), DecimalType::I8);
        assert_eq!(casted.buffer::<i8>().as_ref(), &[2]);
        Ok(())
    }

    #[test]
    fn cast_u64_to_negative_scale_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(1, -19);
        let casted = PrimitiveArray::from_iter([10_000_000_000_000_000_000u64])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.values_type(), DecimalType::I8);
        assert_eq!(casted.buffer::<i8>().as_ref(), &[1]);
        Ok(())
    }

    #[test]
    fn cast_integer_to_i256_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(39, 2);
        let casted = PrimitiveArray::from_iter([42i64])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.values_type(), DecimalType::I256);
        assert_eq!(casted.buffer::<i256>().as_ref(), &[i256::from_i128(4_200)]);
        Ok(())
    }

    #[test]
    fn cast_all_null_integer_to_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(39, 2);
        let casted = PrimitiveArray::new(buffer![i64::MAX, i64::MIN], Validity::AllInvalid)
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert!(matches!(casted.validity(), Ok(Validity::AllInvalid)));
        assert_arrays_eq!(
            &casted,
            DecimalArray::from_option_iter([None::<i256>; 2], decimal_dtype),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_integer_to_negative_scale_decimal_requires_exact_rescale() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, -2);
        let casted = PrimitiveArray::from_iter([1_200i32, -500])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.buffer::<i16>().as_ref(), &[12, -5]);

        let error = PrimitiveArray::from_iter([42i32])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)
            .unwrap_err();
        assert!(error.to_string().contains("would lose precision"));
        Ok(())
    }

    #[rstest]
    fn cast_integer_to_negative_scale_decimal_every_width(
        #[values(
            PType::U8, PType::U16, PType::U32, PType::U64, PType::I8, PType::I16, PType::I32,
            PType::I64
        )]
        ptype: PType,
        #[values(2, 4, 9, 18, 38, 76)] precision: u8,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(precision, -1);
        let casted = PrimitiveArray::from_iter([100u8, 30])
            .into_array()
            .cast(DType::Primitive(ptype, Nullability::NonNullable))?
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(casted.values_type(), decimal_dtype.canonical_values_type());
        match_each_decimal_value_type!(casted.values_type(), |T| {
            let expected = [10i8, 3].map(<T as crate::dtype::BigCast>::from);
            let actual = casted
                .buffer::<T>()
                .iter()
                .map(|&value| Some(value))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        });
        Ok(())
    }

    #[test]
    fn cast_zero_to_large_negative_scale_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, -128);
        let casted = PrimitiveArray::from_iter([0i32])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );
        assert_eq!(casted.buffer::<i16>().as_ref(), &[0]);
        Ok(())
    }

    #[test]
    fn cast_integer_to_decimal_ignores_out_of_range_null_lanes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, 1);
        let casted = PrimitiveArray::new(buffer![999i32, 42], Validity::from_iter([false, true]))
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))?
            .execute::<DecimalArray>(&mut ctx)?;

        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::Nullable)
        );
        assert_eq!(casted.buffer::<i16>().as_ref()[1], 420);
        assert_eq!(
            casted.validity()?.execute_mask(casted.len(), &mut ctx)?,
            Mask::from(BitBuffer::from(vec![false, true]))
        );
        Ok(())
    }

    /// Scaling up into storage of the source's width multiplies and range-checks in one pass.
    #[test]
    fn cast_same_width_integer_to_scaled_decimal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(9, 2);

        let casted = PrimitiveArray::new(
            buffer![i32::MAX, 42, -9_999_999],
            Validity::from_iter([false, true, true]),
        )
        .into_array()
        .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))?;
        assert_arrays_eq!(
            casted,
            DecimalArray::from_option_iter(
                [None, Some(4_200i32), Some(-999_999_900)],
                decimal_dtype
            ),
            &mut ctx
        );

        let casted = PrimitiveArray::from_iter([10_000_000u32, 3])
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?;
        let error = casted.execute::<DecimalArray>(&mut ctx).unwrap_err();
        assert!(error.to_string().contains("does not fit in precision"));
        Ok(())
    }

    /// A buffer that only records byte alignment still casts when its pointer is aligned.
    #[rstest]
    #[case::unscaled(DecimalDType::new(9, 0), [1, 2])]
    #[case::scaled(DecimalDType::new(9, 2), [100, 200])]
    fn cast_integer_with_unrecorded_alignment_to_same_width_decimal(
        #[case] decimal_dtype: DecimalDType,
        #[case] expected: [i32; 2],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let bytes = buffer![1u32, 2]
            .into_byte_buffer()
            .aligned(Alignment::none());
        let source = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(bytes),
            PType::U32,
            Validity::NonNullable,
        );

        let casted = source
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))?;
        assert_arrays_eq!(
            casted,
            DecimalArray::from_iter(expected, decimal_dtype),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_integer_to_decimal_checks_precision() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
    fn cast_floating_primitive_to_decimal_fails() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
        let error = arr
            .cast(PType::U32.into())
            .and_then(|a| {
                a.execute::<Canonical>(&mut SESSION.create_execution_ctx())
                    .map(|c| c.into_array())
            })
            .unwrap_err();
        assert!(error.to_string().contains("values exceed target range"));
    }

    #[test]
    fn cast_array_with_nulls_to_nonnullable() {
        let arr = PrimitiveArray::from_option_iter([Some(-1i32), None, Some(10)]);
        let err = arr
            .into_array()
            .cast(PType::I32.into())
            .and_then(|a| {
                a.execute::<Canonical>(&mut SESSION.create_execution_ctx())
                    .map(|c| c.into_array())
            })
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("Cannot cast array with invalid values to non-nullable type.")
        );
    }

    #[test]
    fn cast_with_invalid_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![-1i32, 0, 10],
            Validity::from_iter([false, true, true]),
        );
        let p = arr
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_arrays_eq!(
            p,
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(10)]),
            &mut ctx
        );
        assert_eq!(
            p.validity()
                .unwrap()
                .execute_mask(p.len(), &mut SESSION.create_execution_ctx())
                .unwrap(),
            Mask::from(BitBuffer::from(vec![false, true, true]))
        );
    }

    /// Same-width integer cast where all values fit: should reinterpret the
    /// buffer without allocation (pointer identity).
    #[test]
    fn cast_same_width_int_reinterprets_buffer() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
        let err = arr
            .cast(PType::I32.into())
            .and_then(|a| {
                a.execute::<Canonical>(&mut SESSION.create_execution_ctx())
                    .map(|c| c.into_array())
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("values exceed target range"),
            "{err}"
        );
    }

    /// All-null array cast between same-width types should succeed without
    /// touching the buffer contents.
    #[test]
    fn cast_same_width_all_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(buffer![0xFFu8, 0xFF], Validity::AllInvalid);
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I8, Nullability::Nullable))?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            &casted,
            PrimitiveArray::from_option_iter([None::<i8>; 2]),
            &mut ctx
        );
        assert!(matches!(casted.validity(), Ok(Validity::AllInvalid)));
        Ok(())
    }

    /// Same-width integer cast with nullable values: out-of-range nulls should
    /// not prevent the cast from succeeding.
    #[test]
    fn cast_same_width_int_nullable_with_out_of_range_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // The null position holds u32::MAX which doesn't fit in i32, but it's
        // masked as invalid so the cast should still succeed via reinterpret.
        let arr = PrimitiveArray::new(
            buffer![u32::MAX, 0u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I32, Nullability::Nullable))?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(0i32), Some(42)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_u32_to_u8_with_out_of_range_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![1000u32, 10u32, 42u32],
            Validity::from_iter([false, true, true]),
        );
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))?;
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
        test_cast_conformance(&array, &mut SESSION.create_execution_ctx());
    }

    /// Floats fit an integer when they truncate into its range, matching `NumCast`.
    #[rstest]
    #[case::u8_truncates_up_to_zero(-0.9, PType::U8, true)]
    #[case::u8_below_range(-1.0, PType::U8, false)]
    #[case::u8_truncates_down_to_max(255.9, PType::U8, true)]
    #[case::u8_above_range(256.0, PType::U8, false)]
    #[case::i8_truncates_up_to_min(-128.9, PType::I8, true)]
    #[case::i8_below_range(-129.0, PType::I8, false)]
    #[case::i32_max(2_147_483_647.0, PType::I32, true)]
    #[case::i32_above_range(2_147_483_648.0, PType::I32, false)]
    #[case::i64_min(-9_223_372_036_854_775_808.0, PType::I64, true)]
    #[case::i64_below_range(-9_223_372_036_854_777_856.0, PType::I64, false)]
    #[case::i64_above_range(9_223_372_036_854_775_808.0, PType::I64, false)]
    #[case::u64_above_range(18_446_744_073_709_551_616.0, PType::U64, false)]
    #[case::nan(f64::NAN, PType::I32, false)]
    #[case::infinity(f64::INFINITY, PType::I64, false)]
    fn cast_float_to_int_range(
        #[case] value: f64,
        #[case] ptype: PType,
        #[case] fits: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = buffer![0.0f64, value]
            .into_array()
            .cast(ptype.into())?
            .execute::<PrimitiveArray>(&mut ctx);
        assert_eq!(result.is_ok(), fits, "{value} as {ptype}");
        Ok(())
    }

    /// Integer-to-integer casts check the source value against the target range.
    #[rstest]
    #[case::i64_to_u8_max(255, PType::U8, true)]
    #[case::i64_to_u8_above(256, PType::U8, false)]
    #[case::i64_to_u8_negative(-1, PType::U8, false)]
    #[case::i64_to_i16_min(-32_768, PType::I16, true)]
    #[case::i64_to_i16_below(-32_769, PType::I16, false)]
    #[case::i64_to_u32_max(4_294_967_295, PType::U32, true)]
    #[case::i64_to_u64_negative(-1, PType::U64, false)]
    fn cast_int_to_int_range(
        #[case] value: i64,
        #[case] ptype: PType,
        #[case] fits: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = buffer![0i64, value]
            .into_array()
            .cast(ptype.into())?
            .execute::<PrimitiveArray>(&mut ctx);
        assert_eq!(result.is_ok(), fits, "{value} as {ptype}");
        Ok(())
    }

    /// Cached min/max skip NaN, so they must not prove that a float array fits an integer type.
    #[test]
    fn cast_float_with_nan_and_cached_min_max_to_int_errors() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = buffer![1.0f64, f64::NAN, 3.0].into_array();
        arr.statistics()
            .compute_all(&[Stat::Min, Stat::Max], &mut ctx)?;
        let result = arr
            .cast(PType::I32.into())?
            .execute::<PrimitiveArray>(&mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    /// Null lanes may hold NaN or out-of-range floats without failing the cast.
    #[test]
    fn cast_float_to_int_ignores_null_lanes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![f32::NAN, 1.5, 1e10, -2.5],
            Validity::from_iter([false, true, false, true]),
        );
        let casted = arr
            .into_array()
            .cast(DType::Primitive(PType::I8, Nullability::Nullable))?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(1i8), None, Some(-2)]),
            &mut ctx
        );
        Ok(())
    }

    /// Every pair of primitive types converts each value like `as`.
    #[test]
    fn cast_every_ptype_pair() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ptypes = [
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::F16,
            PType::F32,
            PType::F64,
        ];
        // Values that every type holds exactly, including a negative value for signed types.
        let values = [0i8, 1, 7, 100, 127];
        for from in ptypes {
            for to in ptypes {
                let signed = |p: PType| p.is_signed_int() || p.is_float();
                let source: Vec<i8> = if signed(from) && signed(to) {
                    values.iter().map(|v| -v).chain(values).collect()
                } else {
                    values.to_vec()
                };
                let make = |ptype: PType| {
                    match_each_native_ptype!(ptype, |T| {
                        PrimitiveArray::from_iter(source.iter().map(|&v| {
                            <T as num_traits::NumCast>::from(v)
                                .vortex_expect("every type holds the value")
                        }))
                        .into_array()
                    })
                };
                assert_arrays_eq!(make(from).cast(to.into())?, make(to), &mut ctx);
            }
        }
        Ok(())
    }

    /// Casts that convert values return a constant null for all-null input instead of
    /// converting values nobody can see; casts that reinterpret the buffer keep it.
    #[rstest]
    #[case::checked_all_false_mask(
        PrimitiveArray::new(
            buffer![1000i32, -1],
            Validity::Array(BoolArray::from_iter([false, false]).into_array()),
        ),
        PType::U8,
        true
    )]
    #[case::checked_all_invalid(
        PrimitiveArray::new(buffer![1000i32, -1], Validity::AllInvalid),
        PType::U8,
        true
    )]
    #[case::unchecked_all_invalid(
        PrimitiveArray::new(buffer![1i32, 2], Validity::AllInvalid),
        PType::F64,
        true
    )]
    #[case::reinterpret_all_invalid(
        PrimitiveArray::new(buffer![1i32, 2], Validity::AllInvalid),
        PType::U32,
        false
    )]
    fn cast_all_null(
        #[case] array: PrimitiveArray,
        #[case] ptype: PType,
        #[case] constant: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let target = DType::Primitive(ptype, Nullability::Nullable);
        let casted = <Primitive as CastKernel>::cast(array.as_view(), &target, &mut ctx)?
            .vortex_expect("primitive casts to primitives are handled");
        assert_eq!(casted.is::<Constant>(), constant);
        assert_eq!(casted.dtype(), &target);
        assert_eq!(casted.invalid_count(&mut ctx)?, array.len());
        Ok(())
    }
}
