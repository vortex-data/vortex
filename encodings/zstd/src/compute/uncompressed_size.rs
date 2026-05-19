// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::FixedWidthUncompressedSizeInBytesKernel;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::validity_uncompressed_size_in_bytes;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::Zstd;

#[derive(Debug)]
pub(crate) struct ZstdUncompressedSizeInBytesKernel;

impl DynAggregateKernel for ZstdUncompressedSizeInBytesKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        if let Some(size) =
            FixedWidthUncompressedSizeInBytesKernel.aggregate(aggregate_fn, batch, ctx)?
        {
            return Ok(Some(size));
        }

        let Some(array) = batch.as_opt::<Zstd>() else {
            return Ok(None);
        };

        if !matches!(array.dtype(), DType::Binary(_) | DType::Utf8(_)) {
            return Ok(None);
        }

        let views_size = u64::try_from(array.len())
            .map_err(|e| vortex_err!("array length does not fit in u64: {e}"))?
            .checked_mul(size_of::<BinaryView>() as u64)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        let data_size = selected_frame_uncompressed_size(array, ctx)?;
        let validity_size = validity_uncompressed_size_in_bytes(
            array
                .as_ref()
                .validity()?
                .execute_mask(array.as_ref().len(), ctx)?,
        )?;

        let size = views_size
            .checked_add(data_size)
            .and_then(|size| size.checked_add(validity_size))
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

        Ok(Some(Scalar::from(size)))
    }
}

fn selected_frame_uncompressed_size(
    array: ArrayView<'_, Zstd>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let unsliced_validity =
        child_to_validity(array.slots()[0].as_ref(), array.dtype().nullability());
    let slice_value_indices = unsliced_validity
        .execute_mask(array.unsliced_n_rows(), ctx)?
        .valid_counts_for_indices(&[array.slice_start(), array.slice_stop()]);
    let slice_value_idx_start = slice_value_indices[0];
    let slice_value_idx_stop = slice_value_indices[1];

    let mut value_idx_start = 0;
    let mut size = 0u64;
    for frame_meta in &array.metadata().frames {
        if value_idx_start >= slice_value_idx_stop {
            break;
        }

        let frame_uncompressed_size = usize::try_from(frame_meta.uncompressed_size)
            .vortex_expect("uncompressed size must fit in usize");
        let frame_n_values = if frame_meta.n_values == 0 {
            // Older metadata omitted n_values; fixed-width arrays return above, so byte count is
            // the best available slice unit here.
            frame_uncompressed_size
        } else {
            usize::try_from(frame_meta.n_values).vortex_expect("frame size must fit usize")
        };

        let value_idx_stop = value_idx_start + frame_n_values;
        if value_idx_stop > slice_value_idx_start {
            size = size
                .checked_add(frame_meta.uncompressed_size)
                .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        }
        value_idx_start = value_idx_stop;
    }

    Ok(size)
}
