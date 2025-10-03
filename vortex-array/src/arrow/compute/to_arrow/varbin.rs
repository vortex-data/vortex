// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{
    ArrayRef as ArrowArrayRef, GenericBinaryArray, GenericStringArray, OffsetSizeTrait,
};
use arrow_schema::DataType;
use vortex_dtype::{DType, IntegerPType, Nullability, PType};
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
        let offsets_ptype = PType::try_from(array.offsets().dtype())?;

        match arrow_type {
            // Emit out preferred Arrow VarBin array.
            None => match array.dtype() {
                DType::Binary(_) => match offsets_ptype {
                    PType::I64 | PType::U64 => to_arrow::<i64>(array),
                    PType::U8 | PType::U16 | PType::U32 | PType::I8 | PType::I16 | PType::I32 => {
                        to_arrow::<i32>(array)
                    }
                    PType::F16 | PType::F32 | PType::F64 => {
                        vortex_panic!("offsets array were somehow floating point")
                    }
                },
                DType::Utf8(_) => match offsets_ptype {
                    PType::I64 | PType::U64 => to_arrow::<i64>(array),
                    PType::U8 | PType::U16 | PType::U32 | PType::I8 | PType::I16 | PType::I32 => {
                        to_arrow::<i32>(array)
                    }
                    PType::F16 | PType::F32 | PType::F64 => {
                        vortex_panic!("offsets array were somehow floating point")
                    }
                },
                dtype => unreachable!("Unsupported DType {dtype}"),
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

fn to_arrow<O: IntegerPType + OffsetSizeTrait>(array: &VarBinArray) -> VortexResult<ArrowArrayRef> {
    let offsets = cast(
        array.offsets(),
        &DType::Primitive(O::PTYPE, Nullability::NonNullable),
    )?
    .to_primitive();

    let nulls = array.validity_mask().to_null_buffer();
    let data = array.bytes().clone();

    // Match on the `DType`.
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
        dtype => {
            unreachable!("expected utf8 or binary instead of {dtype}")
        }
    })
}
