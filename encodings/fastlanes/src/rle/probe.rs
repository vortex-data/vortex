// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! RLE scalar access through the indices, offsets, and values slots.

use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FL_CHUNK_SIZE;
use crate::RLE;
use crate::rle::RLEArrayExt;
use crate::rle::RLESlots;

/// The slice's base value offset, retained across repeated lookups.
#[derive(Default)]
pub struct RleProbeState {
    base: Option<usize>,
}

pub(crate) fn scalar_at(
    state: &mut ProbeState<'_, RLE>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array();
    let logical_index = array.offset() + index;
    // Hold the retained state while reading children: the base offset is cached across rows.
    let (mut retained, mut children) = state.split();
    let code = children
        .slot(RLESlots::INDICES)?
        .ok_or_else(|| vortex_err!("RLE indices slot is missing"))?
        .execute_scalar(logical_index, ctx)?;
    let Some(code) = code.as_primitive().as_::<usize>() else {
        return Ok(Scalar::null(array.dtype().clone()));
    };

    let chunk = logical_index / FL_CHUNK_SIZE;
    let offset = if chunk == 0 {
        0
    } else {
        let base = match retained.as_deref().and_then(|state| state.base) {
            Some(base) => base,
            None => {
                let value = read_offset(
                    children
                        .slot(RLESlots::VALUES_IDX_OFFSETS)?
                        .ok_or_else(|| vortex_err!("RLE offsets slot is missing"))?
                        .execute_scalar(0, ctx)?,
                )?;
                if let Some(state) = retained.as_mut() {
                    state.base = Some(value);
                }
                value
            }
        };
        read_offset(
            children
                .slot(RLESlots::VALUES_IDX_OFFSETS)?
                .ok_or_else(|| vortex_err!("RLE offsets slot is missing"))?
                .execute_scalar(chunk, ctx)?,
        )?
        .checked_sub(base)
        .ok_or_else(|| vortex_err!("RLE offsets precede the slice base"))?
    };
    let value_index = offset
        .checked_add(code)
        .ok_or_else(|| vortex_err!("RLE value index overflow"))?;
    let scalar = children
        .slot(RLESlots::VALUES)?
        .ok_or_else(|| vortex_err!("RLE values slot is missing"))?
        .execute_scalar(value_index, ctx)?;
    Scalar::try_new(array.dtype().clone(), scalar.into_value())
}

fn read_offset(scalar: Scalar) -> VortexResult<usize> {
    scalar
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("RLE offset must be a non-null usize"))
}

#[cfg(test)]
mod tests;
