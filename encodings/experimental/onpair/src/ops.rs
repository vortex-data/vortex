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

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::decode::code_boundary_at;
use crate::decode::collect_widened;

impl OperationsVTable<OnPair> for OnPair {
    fn scalar_at(
        array: ArrayView<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // A row owns a variable-length run of the flat `codes` stream; the
        // per-row `codes_offsets` boundaries map the row index to that run.
        // Read just this row's two boundaries (point lookups that decode at
        // most one chunk of `codes_offsets`) and decode only that run — never
        // the whole column.
        let codes_offsets = array.codes_offsets();
        let row_start = code_boundary_at(codes_offsets, index, ctx)?;
        let row_end = code_boundary_at(codes_offsets, index + 1, ctx)?;

        let codes = collect_widened::<u16>(&array.codes().slice(row_start..row_end)?, ctx)?;
        let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;
        let parts = Parts {
            dict_bytes: array.dict_bytes().as_slice(),
            dict_offsets: dict_offsets.as_slice(),
            bits: array.bits(),
            codes: codes.as_slice(),
        };

        // The per-row decoded length is recorded in the `uncompressed_lengths`
        // child, so read it directly instead of asking the decoder to compute it.
        let len = array
            .uncompressed_lengths()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("OnPair uncompressed_lengths[{index}] is null"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written = onpair::decompress_into(parts, buf.spare_capacity_mut());
        debug_assert_eq!(written, len);
        // SAFETY: `decompress_into` initialised `written` bytes of the spare
        // capacity reserved above.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
