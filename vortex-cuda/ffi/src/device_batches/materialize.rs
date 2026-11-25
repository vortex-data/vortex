// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::Dict;
use vortex::array::arrays::NullArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::DictionaryExport;
use vortex_cuda::arrow::ArrowDeviceArrayWithSchema;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::dynamic_dispatch::ForeignDecodeError;
use vortex_cuda::dynamic_dispatch::decode_into_foreign;
use vortex_cuda_driver::sys::CUstream;

use super::destinations::Destinations;
use super::destinations::vx_cuda_column_destination;

impl Destinations {
    pub(super) async fn export(
        &mut self,
        array: ArrayRef,
        total_rows: u64,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArrayWithSchema> {
        let len = array.len();
        let end = self
            .rows_written
            .checked_add(len)
            .ok_or_else(|| vortex_err!("CUDA scan row count overflow"))?;
        vortex_ensure!(
            end as u64 <= total_rows,
            "CUDA batch exceeds declared row count"
        );

        let mut decoded_ctx;
        let (array, ctx) = if matches!(array.dtype(), DType::Struct(..)) {
            let mut parts = array
                .execute::<StructArray>(ctx.execution_ctx())?
                .into_data_parts();
            vortex_ensure_eq!(parts.fields.len(), self.slots.len());
            // Decide before export: a cuDF consumer must not have to reopen the scan if this
            // batch contains a nested dictionary or binary keys it cannot import.
            let keep_dicts = self.preserve_dictionaries
                && parts.fields.iter().zip(&self.slots).all(|(field, slot)| {
                    slot.is_some()
                        || (!field.dtype().is_nested()
                            && field.as_opt::<Dict>().is_none_or(|dict| {
                                !field.dtype().is_binary() && !dict.values().is::<Dict>()
                            }))
                });
            let ctx = if keep_dicts {
                ctx
            } else {
                decoded_ctx = ctx.fork()?.with_dictionary_export(DictionaryExport::Decode);
                &mut decoded_ctx
            };
            for (field, slot) in parts.fields.iter_mut().zip(&self.slots) {
                if let Some(dest) = slot {
                    write_field(
                        field,
                        dest,
                        self.rows_written,
                        self.consumer_stream.cast(),
                        ctx,
                    )
                    .await?;
                    *field = NullArray::new(len).into_array();
                }
            }
            (
                StructArray::try_new(
                    parts.struct_fields.names().clone(),
                    parts.fields,
                    len,
                    parts.validity,
                )?
                .into_array(),
                ctx,
            )
        } else {
            // Scalar roots have no registerable table fields.
            decoded_ctx = ctx.fork()?.with_dictionary_export(DictionaryExport::Decode);
            (array, &mut decoded_ctx)
        };
        // Recording the struct's Arrow event on this context covers the preceding writes as
        // well as ordinary child exports. Null placeholders themselves need no device buffers.
        let output = array.export_device_array_with_schema(ctx).await?;
        self.rows_written = end;
        Ok(output)
    }
}

async fn write_field(
    field: &ArrayRef,
    dest: &vx_cuda_column_destination,
    row: usize,
    consumer: CUstream,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let width = field.dtype().as_ptype().byte_width();
    // Registration checked total_rows * width and the whole address range; export checked
    // row + field.len(). Bound each write to this batch, not just to the entire allocation.
    let mut offset = row * width;
    let end = offset + field.len() * width;
    vortex_ensure!(
        end <= dest.capacity_bytes,
        "CUDA batch exceeds destination capacity"
    );
    let mut pending = Vec::new();
    let mut next = Some(field.clone());
    while let Some(chunk) = next.take() {
        if let Some(children) = chunk.as_opt::<Chunked>() {
            let start = pending.len();
            pending.extend(children.non_empty_chunks().cloned());
            pending[start..].reverse();
            next = pending.pop();
            continue;
        }
        vortex_ensure!(
            chunk.len() <= (end - offset) / width,
            "chunk exceeds CUDA batch"
        );
        let bytes = chunk.len() * width;
        // SAFETY: The registration contract retains exclusive, non-overlapping storage and a
        // live consumer stream. The bridge borrows both and orders the return, even on error.
        match unsafe {
            decode_into_foreign(
                &chunk,
                (dest.data.addr() + offset) as u64,
                bytes,
                consumer,
                ctx,
            )
            .await
        } {
            Ok(()) => offset += bytes,
            Err(ForeignDecodeError::Ordered(err)) => return Err(err),
            // The C caller can release its allocation on any returned error. This is the
            // unrecoverable case where neither a fence nor stream synchronization worked.
            Err(ForeignDecodeError::OrderingLost(_)) => std::process::abort(),
        }
        next = pending.pop();
    }
    vortex_ensure_eq!(offset, end);
    Ok(())
}
