// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::Parts;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPairView;
use crate::OnPairViewArraySlotsExt;
use crate::decode::code_boundary_at;
use crate::decode::collect_widened;

impl OperationsVTable<OnPairView> for OnPairView {
    fn scalar_at(
        array: ArrayView<'_, OnPairView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // A single row's tokens are always the contiguous window
        // `codes[offset..offset + size]` — overlap/out-of-order only matters
        // *across* rows. Point-look up this row's offset and size and slice the
        // shared `codes` to exactly that window before decoding.
        let offset = code_boundary_at(array.codes_offsets(), index, ctx)?;
        let size = code_boundary_at(array.codes_sizes(), index, ctx)?;

        let codes = collect_widened::<u16>(&array.codes().slice(offset..offset + size)?, ctx)?;
        let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;
        let parts = Parts {
            dict_bytes: array.dict_bytes().as_slice(),
            dict_offsets: dict_offsets.as_slice(),
            bits: array.bits(),
            codes: codes.as_slice(),
        };

        let len = array
            .uncompressed_lengths()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("OnPairView uncompressed_lengths[{index}] is null"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written = onpair::decompress_into(parts, buf.spare_capacity_mut());
        debug_assert_eq!(written, len);
        // SAFETY: `decompress_into` initialised `written` bytes of the spare
        // capacity reserved above.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
