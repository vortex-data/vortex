// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Materialization and checked casts for signed integer array children.
//!
//! Narrow wrappers are removed before materializing storage, so decimal kernels that need a buffer
//! decode at the stored width. Explicit integer casts are the boundary that changes buffer width.

use itertools::Itertools;
use itertools::MinMaxResult;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::Narrow;
use crate::arrays::NarrowArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::WideIntegerArray;
use crate::arrays::WideIntegerEncoding;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::narrow::NarrowArraySlotsExt;
use crate::arrays::narrow::NarrowSlots;
use crate::arrays::wide_integer::WideIntegerArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::dtype::integer::signed_integer_type;
use crate::extension::integer::IntegerWidth;
use crate::extension::integer::WideInteger;
use crate::match_each_decimal_value_type;
use crate::matcher::Matcher;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

/// Stops lazy execution before a Narrow result is expanded to its logical width.
struct IntegerStorage;

impl Matcher for IntegerStorage {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Primitive>()
            || array.is::<WideIntegerEncoding>()
            || array.is::<Narrow>()
            || array.is::<Extension>())
        .then_some(array)
    }
}

/// A materialized signed integer buffer at its stored width.
pub(crate) struct IntegerBuffer {
    /// Aligned native storage, with arbitrary payloads permitted in null rows.
    pub values: BufferHandle,
    /// Signed width of each native element in `values`.
    pub values_type: DecimalType,
    /// Validity with one entry per native integer.
    pub validity: Validity,
}

/// Returns the innermost stored child without materializing or widening it.
pub(crate) fn storage_child(mut array: &ArrayRef) -> &ArrayRef {
    while let Some(narrow) = array.as_opt::<Narrow>() {
        array = narrow.slots()[NarrowSlots::VALUES]
            .as_ref()
            .vortex_expect("Narrow validation requires a values child");
    }

    array
}

/// Borrows the native buffer when the integer storage has already been materialized.
pub(crate) fn buffer_handle(array: &ArrayRef) -> VortexResult<&BufferHandle> {
    let array = storage_child(array);
    if let Some(primitive) = array.as_opt::<Primitive>() {
        return Ok(primitive.data().buffer_handle());
    }
    if let Some(wide) = array.as_opt::<WideIntegerEncoding>() {
        return Ok(wide.data().buffer_handle());
    }

    vortex_bail!(
        "Integer values must be materialized before borrowing a buffer, got {}",
        array.encoding_id()
    )
}

/// Decodes signed integer storage without widening a [`Narrow`] wrapper.
pub(crate) fn materialize(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<IntegerBuffer> {
    let mut executed = storage_child(array).clone();
    loop {
        executed = executed.execute_until::<IntegerStorage>(ctx)?;
        if let Some(narrow) = executed.as_opt::<Narrow>() {
            executed = narrow.values().clone();
        } else {
            break;
        }
    }
    let array = &executed;
    let values_type = signed_integer_type(array.dtype())
        .ok_or_else(|| vortex_err!("Expected a signed integer child, got {}", array.dtype()))?;
    if let Some(wide) = array.as_opt::<WideIntegerEncoding>() {
        return Ok(IntegerBuffer {
            values: wide.buffer_handle().clone(),
            values_type,
            validity: wide.integer_validity(),
        });
    }
    if values_type <= DecimalType::I64 {
        let parts = array
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_data_parts();
        return Ok(IntegerBuffer {
            values: parts.buffer,
            values_type,
            validity: parts.validity,
        });
    }

    let extension = array.clone().execute::<ExtensionArray>(ctx)?;
    let storage = extension
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(ctx)?;
    let bytes = storage.elements().clone().execute::<PrimitiveArray>(ctx)?;
    let alignment = match_each_decimal_value_type!(values_type, |T| { Alignment::of::<T>() });
    let values = BufferHandle::new_host(bytes.to_buffer::<u8>().aligned(alignment));

    Ok(IntegerBuffer {
        values,
        values_type,
        validity: storage.validity()?,
    })
}

/// Wraps native values in the primitive or wide integer encoding of their stored width.
pub(crate) fn from_buffer(buffer: IntegerBuffer) -> VortexResult<ArrayRef> {
    if buffer.values_type > DecimalType::I64 {
        return WideIntegerArray::try_new_handle(
            buffer.values,
            buffer.values_type,
            buffer.validity,
        )
        .map(IntoArray::into_array);
    }

    let ptype = match buffer.values_type {
        DecimalType::I8 => PType::I8,
        DecimalType::I16 => PType::I16,
        DecimalType::I32 => PType::I32,
        DecimalType::I64 => PType::I64,
        _ => unreachable!("Wide integer types were handled above"),
    };
    Ok(PrimitiveArray::from_buffer_handle(buffer.values, ptype, buffer.validity).into_array())
}

/// Reads a non-null signed integer scalar without applying a decimal scale.
pub(crate) fn scalar_value(scalar: &Scalar) -> VortexResult<DecimalValue> {
    vortex_ensure!(!scalar.is_null(), "Expected a non-null integer scalar");
    if let Some(width) = WideInteger::width(scalar.dtype()) {
        return Ok(DecimalValue::I256(WideInteger::unpack_value(
            width,
            scalar
                .value()
                .vortex_expect("Integer scalar was checked non-null"),
        )?));
    }

    let scalar = scalar.as_primitive();
    Ok(match scalar.ptype() {
        PType::I8 => DecimalValue::from(
            scalar
                .typed_value::<i8>()
                .vortex_expect("Integer scalar is non-null"),
        ),
        PType::I16 => DecimalValue::from(
            scalar
                .typed_value::<i16>()
                .vortex_expect("Integer scalar is non-null"),
        ),
        PType::I32 => DecimalValue::from(
            scalar
                .typed_value::<i32>()
                .vortex_expect("Integer scalar is non-null"),
        ),
        PType::I64 => DecimalValue::from(
            scalar
                .typed_value::<i64>()
                .vortex_expect("Integer scalar is non-null"),
        ),
        ptype => vortex_bail!("Expected a signed integer scalar, got {ptype}"),
    })
}

/// Creates an integer scalar, rejecting values outside the requested signed width.
pub(crate) fn scalar_from_integer(value: DecimalValue, dtype: &DType) -> VortexResult<Scalar> {
    let values_type = signed_integer_type(dtype)
        .ok_or_else(|| vortex_err!("Expected a signed integer dtype, got {dtype}"))?;
    if let Some(width) = WideInteger::width(dtype) {
        let bytes = match width {
            IntegerWidth::I128 => value
                .cast::<i128>()
                .ok_or_else(|| vortex_err!("Integer does not fit {dtype}, got {value}"))?
                .to_le_bytes()
                .to_vec(),
            IntegerWidth::I256 => value
                .cast::<i256>()
                .vortex_expect("Every signed integer fits i256")
                .to_le_bytes()
                .to_vec(),
        };
        let values = bytes
            .into_iter()
            .map(|byte| Some(ScalarValue::Primitive(PValue::U8(byte))))
            .collect();
        return Scalar::try_new(dtype.clone(), Some(ScalarValue::Tuple(values)));
    }

    macro_rules! primitive {
        ($T:ty) => {
            Scalar::primitive(
                value
                    .cast::<$T>()
                    .ok_or_else(|| vortex_err!("Integer does not fit {dtype}, got {value}"))?,
                dtype.nullability(),
            )
        };
    }

    Ok(match values_type {
        DecimalType::I8 => primitive!(i8),
        DecimalType::I16 => primitive!(i16),
        DecimalType::I32 => primitive!(i32),
        DecimalType::I64 => primitive!(i64),
        _ => unreachable!("Wide integer scalars were handled above"),
    })
}

/// Casts a signed integer scalar with checked width and nullability.
pub(crate) fn cast_scalar(scalar: &Scalar, dtype: &DType) -> VortexResult<Scalar> {
    if scalar.is_null() {
        vortex_ensure!(dtype.is_nullable(), "Cannot cast null to {dtype}");
        return Ok(Scalar::null(dtype.clone()));
    }

    scalar_from_integer(scalar_value(scalar)?, dtype)
}

/// Materializes and casts valid signed integer rows, ignoring null payloads.
pub(crate) fn cast_array(
    array: &ArrayRef,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let target_type = signed_integer_type(dtype)
        .ok_or_else(|| vortex_err!("Expected a signed integer dtype, got {dtype}"))?;
    let source = materialize(array, ctx)?;
    let len = source.values.len() / source.values_type.byte_width();
    let validity = source
        .validity
        .cast_nullability(dtype.nullability(), len, ctx)?;
    if target_type == source.values_type {
        return from_buffer(IntegerBuffer { validity, ..source });
    }

    let mask = validity.execute_mask(len, ctx)?;
    let values = match_each_decimal_value_type!(source.values_type, |S| {
        let source = Buffer::<S>::from_byte_buffer(source.values.to_host_sync());
        match_each_decimal_value_type!(target_type, |T| {
            cast_buffer::<S, T>(&source, &mask, ctx)
        })
    })?;

    from_buffer(IntegerBuffer {
        values,
        values_type: target_type,
        validity,
    })
}

fn cast_buffer<S: NativeDecimalType, T: NativeDecimalType>(
    source: &[S],
    mask: &Mask,
    ctx: &ExecutionCtx,
) -> VortexResult<BufferHandle> {
    let mut values = BufferMut::<T>::zeroed_in(source.len(), ctx.allocator().clone());
    match mask {
        Mask::AllTrue(_) => {
            for (dst, &src) in values.iter_mut().zip(source) {
                *dst = <T as BigCast>::from(src).ok_or_else(|| {
                    vortex_err!("Integer does not fit {}, got {src}", T::DECIMAL_TYPE)
                })?;
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(mask) => {
            for &index in mask.indices() {
                values[index] = <T as BigCast>::from(source[index]).ok_or_else(|| {
                    vortex_err!(
                        "Integer does not fit {}, got {}",
                        T::DECIMAL_TYPE,
                        source[index]
                    )
                })?;
            }
        }
    }

    Ok(BufferHandle::new_host(values.freeze().into_byte_buffer()))
}

/// Bounds of non-null signed values, computed at the stored width.
pub(crate) fn bounds(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(i256, i256)>> {
    let buffer = materialize(array, ctx)?;
    let mask = buffer.validity.execute_mask(array.len(), ctx)?;
    match_each_decimal_value_type!(buffer.values_type, |T| {
        let values = Buffer::<T>::from_byte_buffer(buffer.values.to_host_sync());
        let bounds = match &mask {
            Mask::AllFalse(_) => MinMaxResult::NoElements,
            Mask::AllTrue(_) => values.iter().copied().minmax(),
            Mask::Values(mask) => values
                .iter()
                .copied()
                .zip(mask.bit_buffer().iter())
                .filter_map(|(value, valid)| valid.then_some(value))
                .minmax(),
        };
        Ok(match bounds {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(value) => {
                let value = DecimalValue::from(value).as_i256();
                Some((value, value))
            }
            MinMaxResult::MinMax(min, max) => Some((
                DecimalValue::from(min).as_i256(),
                DecimalValue::from(max).as_i256(),
            )),
        })
    })
}

/// Fills nulls at the smallest stored width that also fits the fill value.
pub(crate) fn fill_null(
    array: &ArrayRef,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let logical_type =
        signed_integer_type(array.dtype()).vortex_expect("Signed integer fill dtype");
    let value = scalar_value(fill_value)?;
    let buffer = materialize(array, ctx)?;
    let mask = buffer.validity.execute_mask(array.len(), ctx)?;
    let values_type = if mask.all_true() {
        buffer.values_type
    } else {
        fill_values_type(buffer.values_type, value)
    };
    let values = if mask.all_true() {
        buffer.values
    } else {
        match_each_decimal_value_type!(values_type, |T| {
            let fill_value = value
                .cast::<T>()
                .vortex_expect("Fill value fits stored width");
            let mut values = widened_buffer::<T>(&buffer).into_mut();
            for (index, value) in values.iter_mut().enumerate() {
                if !mask.value(index) {
                    *value = fill_value;
                }
            }
            BufferHandle::new_host(values.freeze().into_byte_buffer())
        })
    };
    let values = from_buffer(IntegerBuffer {
        values,
        values_type,
        validity: Validity::from(fill_value.dtype().nullability()),
    })?;
    if values_type < logical_type {
        return Ok(NarrowArray::try_new(
            values,
            array
                .dtype()
                .with_nullability(fill_value.dtype().nullability()),
        )?
        .into_array());
    }
    Ok(values)
}

/// Selects the smallest width that holds both the stored values and the fill constant.
fn fill_values_type(stored: DecimalType, value: DecimalValue) -> DecimalType {
    for candidate in [
        DecimalType::I8,
        DecimalType::I16,
        DecimalType::I32,
        DecimalType::I64,
        DecimalType::I128,
    ] {
        if candidate < stored {
            continue;
        }

        let fits = match_each_decimal_value_type!(candidate, |T| { value.cast::<T>().is_some() });
        if fits {
            return candidate;
        }
    }

    DecimalType::I256
}

/// Widens materialized signed storage to a common native width for bulk comparison.
pub(crate) fn widened_buffer<T: NativeDecimalType>(array: &IntegerBuffer) -> Buffer<T> {
    if array.values_type == T::DECIMAL_TYPE {
        return Buffer::from_byte_buffer(array.values.to_host_sync());
    }
    match_each_decimal_value_type!(array.values_type, |S| {
        Buffer::<S>::from_byte_buffer(array.values.to_host_sync())
            .into_iter()
            .map(|value| {
                <T as BigCast>::from(value).vortex_expect("Widening an integer cannot fail")
            })
            .collect()
    })
}
