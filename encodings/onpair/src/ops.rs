// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::OnPairSlots;
use crate::array::dict_view;
use crate::decode::code_boundary;
use crate::decode::collect_widened_range;

/// What a repeated probe keeps between reads of an [`OnPair`] array.
///
/// The dictionary is memoized in `OnPairData`, and the children keep their own probes. The probe
/// keeps the code range of the last row, because the end of one row is the start of the next.
#[derive(Default)]
pub struct OnPairProbeState {
    last_row: Option<RowBounds>,
}

/// The code range of one row.
#[derive(Clone, Copy)]
struct RowBounds {
    index: usize,
    start: usize,
    end: usize,
}

impl OperationsVTable<OnPair> for OnPair {
    type ProbeState = OnPairProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        // A row owns a variable-length run of the flat `codes` stream; the
        // per-row `codes_offsets` boundaries map the row index to that run.
        // Read just this row's two boundaries and decode only that run — never
        // the whole column.
        let (retained, mut children) = state.split();
        let mut codes_offsets = children
            .slot(OnPairSlots::CODES_OFFSETS)?
            .ok_or_else(|| vortex_err!("OnPair has no codes_offsets child"))?;
        let mut boundary =
            |row: usize| code_boundary(&codes_offsets.execute_scalar(row, ctx)?, row);

        // A repeated read of the same row needs no boundary, and a read of the next row starts
        // where the last row ended.
        let (row_start, row_end) = match retained.as_ref().and_then(|state| state.last_row) {
            Some(last) if last.index == index => (last.start, last.end),
            Some(last) if last.index + 1 == index => (last.end, boundary(index + 1)?),
            _ => (boundary(index)?, boundary(index + 1)?),
        };

        if let Some(state) = retained {
            state.last_row = Some(RowBounds {
                index,
                start: row_start,
                end: row_end,
            });
        }

        // The per-row decoded length is recorded in the `uncompressed_lengths`
        // child, so read it directly instead of asking the decoder to compute it.
        let len = state
            .slot(OnPairSlots::UNCOMPRESSED_LENGTHS)?
            .ok_or_else(|| vortex_err!("OnPair has no uncompressed_lengths child"))?
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_opt::<usize>()
            .flatten()
            .ok_or_else(|| {
                vortex_err!("OnPair uncompressed_lengths[{index}] is null, negative, or too large")
            })?;

        // The codes are a range read, which a scalar probe cannot serve.
        let codes = collect_widened_range::<u16>(array.codes(), row_start..row_end, ctx)?;
        let dict = dict_view(array, ctx)?;

        // MAX_TOKEN_SIZE bytes per code always hold the row, and decode it fastest.
        let mut buf: Vec<u8> = Vec::with_capacity(codes.len() * onpair::MAX_TOKEN_SIZE);
        let written =
            match onpair::decompress_into(codes.as_slice(), dict, buf.spare_capacity_mut()) {
                Ok(decompressed) => decompressed.len(),
                Err(_) => vortex_bail!("OnPair row {index} exceeds MAX_TOKEN_SIZE bytes per code"),
            };
        vortex_ensure_eq!(written, len, "OnPair row {index} decoded length mismatch");
        // SAFETY: `decompress_into` initialised exactly `written` bytes.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }

    fn scalar_at(
        array: ArrayView<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
