// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Delta scalar access by decoding the chunk that holds the row.
//!
//! A value depends on every earlier delta of its lane in its chunk, so a read decodes the whole
//! 1024-value chunk. Repeated probes keep the last decoded chunk, so reads that stay within a chunk
//! decode it once.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ProbeState;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::lane_count;

/// The last chunk a repeated probe decoded, and its values in logical order.
#[derive(Default)]
pub struct DeltaProbeState {
    chunk: Option<(usize, PrimitiveArray)>,
}

pub(crate) fn scalar_at(
    state: &mut ProbeState<'_, Delta>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array();
    if !state.is_valid(index, ctx)? {
        return Ok(Scalar::null(array.dtype().clone()));
    }
    let position = array.offset() + index;
    let chunk = position / FL_CHUNK_SIZE;
    let within = position % FL_CHUNK_SIZE;
    match state.retained() {
        Some(retained) => {
            let decoded = match &mut retained.chunk {
                Some((cached, decoded)) if *cached == chunk => decoded,
                slot => &mut slot.insert((chunk, decode_chunk(array, chunk, ctx)?)).1,
            };
            Ok(value_at(array, decoded, within))
        }
        None => Ok(value_at(array, &decode_chunk(array, chunk, ctx)?, within)),
    }
}

/// Decode chunk `chunk` of `array` into its 1024 values, ignoring validity.
fn decode_chunk(
    array: ArrayView<'_, Delta>,
    chunk: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let lanes = lane_count(array.dtype().as_ptype());
    let bases = array.bases().slice(chunk * lanes..(chunk + 1) * lanes)?;
    let deltas = array
        .deltas()
        .slice(chunk * FL_CHUNK_SIZE..(chunk + 1) * FL_CHUNK_SIZE)?;
    Delta::try_new(bases, deltas, Validity::NonNullable, 0, FL_CHUNK_SIZE)?
        .into_array()
        .execute::<PrimitiveArray>(ctx)
}

/// The value at `within` of a decoded chunk, with the nullability of `array`.
fn value_at(array: ArrayView<'_, Delta>, decoded: &PrimitiveArray, within: usize) -> Scalar {
    let nullability = array.dtype().nullability();
    match_each_integer_ptype!(decoded.ptype(), |P| {
        Scalar::primitive::<P>(decoded.as_slice::<P>()[within], nullability)
    })
}

#[cfg(test)]
mod tests;
