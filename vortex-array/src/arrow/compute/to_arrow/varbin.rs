use std::sync::Arc;

use arrow_array::{
    ArrayRef as ArrowArrayRef, GenericBinaryArray, GenericStringArray, OffsetSizeTrait,
};
use arrow_schema::DataType;
use vortex_dtype::{DType, NativePType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::arrow::compute::{ToArrowKernel, ToArrowKernelAdapter};
use crate::compute::cast;
use crate::{Array, ToCanonical, register_kernel};

impl ToArrowKernel for VarBinEncoding {
    fn to_arrow(
        &self,
        array: &VarBinArray,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        let offsets_ptype = PType::try_from(array.offsets().dtype())?;

        match arrow_type {
            // Emit out preferred Arrow VarBin array.
            None => match array.dtype() {
                DType::Binary(_) => match offsets_ptype {
                    PType::I64 | PType::U64 => to_arrow::<i64>(array),
                    _ => to_arrow::<i32>(array),
                },
                DType::Utf8(_) => match offsets_ptype {
                    PType::I64 | PType::U64 => to_arrow::<i64>(array),
                    _ => to_arrow::<i32>(array),
                },
                _ => unreachable!("Unsupported DType"),
            },
            // Emit the requested Arrow array.
            Some(DataType::Binary) if array.dtype().is_binary() => to_arrow::<i32>(array),
            Some(DataType::LargeBinary) if array.dtype().is_binary() => to_arrow::<i64>(array),
            Some(DataType::Utf8) if array.dtype().is_utf8() => to_arrow::<i32>(array),
            Some(DataType::LargeUtf8) if array.dtype().is_utf8() => to_arrow::<i64>(array),
            // Allow fallback to canonicalize to a VarBinView and try again.
            Some(DataType::BinaryView) | Some(DataType::Utf8View) => {
                return Ok(None);
            }
            // Any other type is not supported.
            Some(_) => {
                vortex_bail!("Cannot convert VarBin to Arrow type {arrow_type:?}");
            }
        }
        .map(Some)
    }
}

register_kernel!(ToArrowKernelAdapter(VarBinEncoding).lift());

fn to_arrow<O: NativePType + OffsetSizeTrait>(array: &VarBinArray) -> VortexResult<ArrowArrayRef> {
    let offsets = cast(
        array.offsets(),
        &DType::Primitive(O::PTYPE, Nullability::NonNullable),
    )?
    .to_primitive()
    .map_err(|err| err.with_context("Failed to canonicalize offsets"))?;

    let nulls = array.validity_mask()?.to_null_buffer();
    let data = array.bytes().clone();

    // Switch on DType.
    Ok(match array.dtype() {
        DType::Binary(_) => Arc::new(unsafe {
            GenericBinaryArray::new_unchecked(
                offsets.buffer::<O>().into_arrow_offset_buffer(),
                data.into_arrow_buffer(),
                nulls,
            )
        }),
        DType::Utf8(_) => Arc::new(unsafe {
            GenericStringArray::new_unchecked(
                offsets.buffer::<O>().into_arrow_offset_buffer(),
                data.into_arrow_buffer(),
                nulls,
            )
        }),
        _ => unreachable!("expected utf8 or binary instead of {}", array.dtype()),
    })
}
