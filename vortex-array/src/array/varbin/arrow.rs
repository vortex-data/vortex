use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, Datum, LargeBinaryArray, LargeStringArray, StringArray};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::array::VarBinArray;
use crate::arrow::wrappers::as_offset_buffer;
use crate::compute::unary::try_cast;
use crate::validity::ArrayValidity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, IntoArrayVariant, ToArrayData};

/// Convert the array to Arrow variable length binary array type.
pub(crate) fn varbin_to_arrow(varbin_array: &VarBinArray) -> VortexResult<ArrayRef> {
    let offsets = varbin_array
        .offsets()
        .into_primitive()
        .map_err(|err| err.with_context("Failed to canonicalize offsets"))?;
    let offsets = match offsets.ptype() {
        PType::I32 | PType::I64 => offsets,
        PType::U64 => try_cast(offsets, PType::I64.into())?.into_primitive()?,
        PType::U32 => try_cast(offsets, PType::I32.into())?.into_primitive()?,

        // Unless it's u64, everything else can be converted into an i32.
        _ => try_cast(offsets.to_array(), PType::I32.into())
            .and_then(|a| a.into_primitive())
            .map_err(|err| err.with_context("Failed to cast offsets to PrimitiveArray of i32"))?,
    };
    let nulls = varbin_array
        .logical_validity()
        .to_null_buffer()
        .map_err(|err| err.with_context("Failed to get null buffer from logical validity"))?;

    let data = varbin_array
        .bytes()
        .into_primitive()
        .map_err(|err| err.with_context("Failed to canonicalize bytes"))?;
    if data.dtype() != &DType::BYTES {
        vortex_bail!("Expected bytes to be of type U8, got {}", data.ptype());
    }
    let data = data.buffer();

    // Switch on Arrow DType.
    Ok(match varbin_array.dtype() {
        DType::Binary(_) => match offsets.ptype() {
            PType::I32 => Arc::new(unsafe {
                BinaryArray::new_unchecked(
                    as_offset_buffer::<i32>(offsets),
                    data.clone().into_arrow(),
                    nulls,
                )
            }),
            PType::I64 => Arc::new(unsafe {
                LargeBinaryArray::new_unchecked(
                    as_offset_buffer::<i64>(offsets),
                    data.clone().into_arrow(),
                    nulls,
                )
            }),
            _ => vortex_bail!("Invalid offsets type {}", offsets.ptype()),
        },
        DType::Utf8(_) => match offsets.ptype() {
            PType::I32 => Arc::new(unsafe {
                StringArray::new_unchecked(
                    as_offset_buffer::<i32>(offsets),
                    data.clone().into_arrow(),
                    nulls,
                )
            }),
            PType::I64 => Arc::new(unsafe {
                LargeStringArray::new_unchecked(
                    as_offset_buffer::<i64>(offsets),
                    data.clone().into_arrow(),
                    nulls,
                )
            }),
            _ => vortex_bail!("Invalid offsets type {}", offsets.ptype()),
        },
        _ => vortex_bail!(
            "expected utf8 or binary instead of {}",
            varbin_array.dtype()
        ),
    })
}

/// Create a [`Datum`] from a Utf8 or Binary scalar.
pub(crate) fn varbin_datum(scalar: Scalar) -> VortexResult<Arc<dyn Datum>> {
    match scalar.dtype() {
        DType::Utf8(_) => Ok(Arc::new(
            scalar
                .value()
                .as_buffer_string()?
                .map(StringArray::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(StringArray::new_null(1))),
        )),
        DType::Binary(_) => Ok(Arc::new(
            scalar
                .value()
                .as_buffer()?
                .map(BinaryArray::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(BinaryArray::new_null(1))),
        )),

        other => vortex_bail!("Expected Utf8 or Binary scalar, found {other}"),
    }
}
