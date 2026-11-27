// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::array::ValidityVTable;
use crate::arrays::Primitive;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::indices::chunk_start;
use crate::arrays::patched::indices::local_position;
use crate::arrays::scalar_fn::ValidityArray;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl OperationsVTable<Patched> for Patched {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let row = array.offset + index;
        let chunk = row / PATCH_CHUNK_SIZE - array.offset / PATCH_CHUNK_SIZE;
        let local = local_position(row as u64);
        let offsets = array.patch_chunk_offsets();
        let start = chunk_start(offsets, chunk, array.patch_indices().len(), ctx)?;
        let end = chunk_start(offsets, chunk + 1, array.patch_indices().len(), ctx)?;
        let position = if let Some(indices) = array.patch_indices().as_opt::<Primitive>()
            && indices.buffer_handle().is_on_host()
        {
            indices.as_slice::<u16>()[start..end]
                .binary_search(&local)
                .ok()
                .map(|i| start + i)
        } else {
            let mut indices = state
                .slot(PatchedSlots::PATCH_INDICES)?
                .vortex_expect("required patch indices");
            let mut range = start..end;
            let mut found = None;
            while !range.is_empty() {
                let mid = range.start + range.len() / 2;
                let value = u16::try_from(&indices.execute_scalar(mid, ctx)?)?;
                match value.cmp(&local) {
                    Ordering::Less => range.start = mid + 1,
                    Ordering::Greater => range.end = mid,
                    Ordering::Equal => {
                        found = Some(mid);
                        break;
                    }
                }
            }
            found
        };
        if let Some(position) = position {
            return state
                .slot(PatchedSlots::PATCH_VALUES)?
                .vortex_expect("required patch values")
                .execute_scalar(position, ctx);
        }
        state
            .slot(PatchedSlots::INNER)?
            .vortex_expect("required base")
            .execute_scalar(index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl ValidityVTable<Patched> for Patched {
    fn validity(array: ArrayView<'_, Self>) -> VortexResult<Validity> {
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        match (array.inner().validity()?, array.patch_values().validity()?) {
            (Validity::AllValid, Validity::AllValid) => Ok(Validity::AllValid),
            (Validity::AllInvalid, Validity::AllInvalid) => Ok(Validity::AllInvalid),
            _ => Ok(Validity::Array(
                ValidityArray::new(array.array().clone()).into_array(),
            )),
        }
    }
}
