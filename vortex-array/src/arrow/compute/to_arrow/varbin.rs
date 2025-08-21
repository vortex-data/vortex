// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{
    ArrayRef as ArrowArrayRef, GenericBinaryArray, GenericStringArray, OffsetSizeTrait,
};
use arrow_schema::DataType;
use vortex_dtype::{DType, NativePType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};

use crate::arrays::{VarBinArray, VarBinVTable};
use crate::arrow::compute::{ToArrowKernel, ToArrowKernelAdapter};
use crate::compute::cast;
use crate::{Array, ToCanonical, register_kernel};

impl ToArrowKernel for VarBinVTable {
    fn to_arrow(
        &self,
        array: &VarBinArray,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        use DType::*;
        use PType::*;

        let offsets_ptype = PType::try_from(array.offsets().dtype())?;

        match arrow_type {
            // Emit out preferred Arrow VarBin array.
            None => match array.dtype() {
                Binary(_) => match offsets_ptype {
                    I64 | U64 => to_arrow::<i64>(array),
                    U8 | U16 | U32 | I8 | I16 | I32 => to_arrow::<i32>(array),
                    F16 | F32 | F64 => vortex_panic!("offsets array were somehow floating point"),
                },
                Utf8(_) => match offsets_ptype {
                    I64 | U64 => to_arrow::<i64>(array),
                    U8 | U16 | U32 | I8 | I16 | I32 => to_arrow::<i32>(array),
                    F16 | F32 | F64 => vortex_panic!("offsets array were somehow floating point"),
                },
                Null | Bool(_) | Primitive(..) | Decimal(..) | List(..) | Struct(..)
                | Extension(_) => unreachable!("Unsupported DType"),
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

register_kernel!(ToArrowKernelAdapter(VarBinVTable).lift());

fn to_arrow<O: NativePType + OffsetSizeTrait>(array: &VarBinArray) -> VortexResult<ArrowArrayRef> {
    use DType::*;

    let offsets = cast(
        array.offsets(),
        &Primitive(O::PTYPE, Nullability::NonNullable),
    )?
    .to_primitive()
    .map_err(|err| err.with_context("Failed to canonicalize offsets"))?;

    let nulls = array.validity_mask()?.to_null_buffer();
    let data = array.bytes().clone();

    // Match on the `DType`.
    Ok(match array.dtype() {
        Binary(_) => Arc::new(unsafe {
            GenericBinaryArray::new_unchecked(
                offsets.buffer::<O>().into_arrow_offset_buffer(),
                data.into_arrow_buffer(),
                nulls,
            )
        }),
        Utf8(_) => Arc::new(unsafe {
            GenericStringArray::new_unchecked(
                offsets.buffer::<O>().into_arrow_offset_buffer(),
                data.into_arrow_buffer(),
                nulls,
            )
        }),
        Null | Bool(_) | Primitive(..) | Decimal(..) | List(..) | Struct(..) | Extension(_) => {
            unreachable!("expected utf8 or binary instead of {}", array.dtype())
        }
    })
}
