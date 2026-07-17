// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::CompactDictionaryView;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

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
        let dict =
            CompactDictionaryView::validate(array.dict_bytes().as_slice(), dict_offsets.as_slice())
                .map_err(|e| vortex_err!(InvalidArgument: "Invalid OnPair dictionary: {e}"))?;

        // The per-row decoded length is recorded in the `uncompressed_lengths`
        // child, so read it directly instead of asking the decoder to compute it.
        let len = array
            .uncompressed_lengths()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("OnPair uncompressed_lengths[{index}] is null"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written =
            match onpair::try_decode_into(codes.as_slice(), dict, buf.spare_capacity_mut()) {
                Ok(written) => written,
                Err(_) => vortex_panic!(
                    "OnPair row {index} decodes to more bytes than uncompressed_lengths records"
                ),
            };
        if written != len {
            vortex_panic!(
                "OnPair row {index} decoded to {written} bytes but uncompressed_lengths records {len}"
            );
        }
        // SAFETY: `try_decode_into` initialised exactly `written` bytes.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
