// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::Array;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::BoolArrayParts;
use vortex::array::arrays::BoolVTable;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::PrimitiveArrayParts;
use vortex::array::arrays::SliceArrayParts;
use vortex::array::arrays::SliceVTable;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::StructArrayParts;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;

/// Slice a BoolArray that may have device-resident buffers.
///
/// Unlike the standard BoolArray::slice which calls to_bit_buffer() (requires host access),
/// this works at the BufferHandle level: it byte-slices the buffer to absorb full bytes
/// and keeps only the sub-byte remainder as the bit offset.
fn slice_bool_device_safe(bool_array: BoolArray, range: Range<usize>) -> VortexResult<BoolArray> {
    let BoolArrayParts {
        bits,
        offset,
        validity,
        ..
    } = bool_array.into_parts();

    let abs_start = offset + range.start;
    let new_len = range.len();

    // Slice the buffer to absorb full bytes, keep sub-byte remainder as offset
    let byte_start = abs_start / 8;
    let new_offset = abs_start % 8;
    let byte_end = (abs_start + new_len).div_ceil(8);

    let sliced_bits = bits.slice(byte_start..byte_end);
    let sliced_validity = slice_validity_device_safe(validity, range)?;

    Ok(BoolArray::new_handle(
        sliced_bits,
        new_offset,
        new_len,
        sliced_validity,
    ))
}

/// Slice validity in a device-buffer-safe way.
fn slice_validity_device_safe(
    validity: Validity,
    range: Range<usize>,
) -> VortexResult<Validity> {
    match &validity {
        Validity::NonNullable | Validity::AllValid | Validity::AllInvalid => Ok(validity),
        Validity::Array(a) => {
            if let Some(bool_array) = a.as_opt::<BoolVTable>() {
                Ok(Validity::Array(
                    slice_bool_device_safe(bool_array.clone(), range)?.into_array(),
                ))
            } else {
                Ok(Validity::Array(a.slice(range)?))
            }
        }
    }
}

/// Slice a canonical array that may have device-resident buffers.
fn slice_canonical(canonical: Canonical, range: Range<usize>) -> VortexResult<Canonical> {
    match canonical {
        Canonical::Null(null_array) => null_array.slice(range)?.to_canonical(),
        Canonical::Bool(bool_array) => {
            Ok(Canonical::Bool(slice_bool_device_safe(bool_array, range)?))
        }
        Canonical::Primitive(prim_array) => {
            let PrimitiveArrayParts {
                ptype,
                buffer,
                validity,
                ..
            } = prim_array.into_parts();
            let byte_start = range.start * ptype.byte_width();
            let byte_end = range.end * ptype.byte_width();
            let sliced_buf = buffer.slice(byte_start..byte_end);
            let sliced_validity = slice_validity_device_safe(validity, range)?;
            Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
                sliced_buf,
                ptype,
                sliced_validity,
            )))
        }
        Canonical::Struct(struct_array) => {
            let len = range.len();
            let StructArrayParts {
                fields,
                struct_fields,
                validity,
                ..
            } = struct_array.into_parts();
            let sliced_fields: Vec<ArrayRef> = fields
                .iter()
                .map(|f| {
                    let canonical = f.to_canonical()?;
                    Ok(slice_canonical(canonical, range.clone())?.into_array())
                })
                .collect::<VortexResult<_>>()?;
            let sliced_validity = slice_validity_device_safe(validity, range)?;
            Ok(Canonical::Struct(StructArray::new(
                struct_fields.names().clone(),
                sliced_fields,
                len,
                sliced_validity,
            )))
        }
        Canonical::Decimal(decimal_array) => decimal_array.slice(range)?.to_canonical(),
        Canonical::VarBinView(varbinview) => varbinview.slice(range)?.to_canonical(),
        Canonical::Extension(extension_array) => extension_array.slice(range)?.to_canonical(),
        c => todo!("Device-aware slice not implemented for {}", c.dtype()),
    }
}

#[derive(Debug)]
pub struct SliceExecutor;

#[async_trait]
impl CudaExecute for SliceExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let slice_array = array.try_into::<SliceVTable>().map_err(|array| {
            vortex_err!(
                "SliceExecutor requires input of SliceArray, was {}",
                array.encoding_id()
            )
        })?;

        let SliceArrayParts { child, range } = slice_array.into_parts();
        let child = child.execute_cuda(ctx).await?;

        slice_canonical(child, range)
    }
}
