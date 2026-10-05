// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::array::dict_view;
use crate::decode::code_boundary_at;
use crate::decode::collect_widened;

impl OperationsVTable<OnPair> for OnPair {
    type ProbeState = ();

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
        let dict = dict_view(array, ctx)?;

        // The per-row decoded length is recorded in the `uncompressed_lengths`
        // child, so read it directly instead of asking the decoder to compute it.
        let len = array
            .uncompressed_lengths()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_opt::<usize>()
            .flatten()
            .ok_or_else(|| {
                vortex_err!("OnPair uncompressed_lengths[{index}] is null, negative, or too large")
            })?;
        // The stored length controls allocation; each code emits 1 to MAX_TOKEN_SIZE bytes.
        vortex_ensure!(
            codes.len() <= len && len <= codes.len().saturating_mul(onpair::MAX_TOKEN_SIZE),
            "OnPair row {index} recorded length {len} is impossible for {} codes",
            codes.len()
        );
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written =
            match onpair::try_decode_into(codes.as_slice(), dict, buf.spare_capacity_mut()) {
                Ok(written) => written,
                Err(_) => vortex_bail!("OnPair row {index} exceeds its recorded length"),
            };
        vortex_ensure_eq!(written, len, "OnPair row {index} decoded length mismatch");
        // SAFETY: `try_decode_into` initialised exactly `written` bytes.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
