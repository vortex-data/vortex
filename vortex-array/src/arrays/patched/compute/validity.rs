// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::indices::visit_chunks;
use crate::arrays::scalar_fn::ScalarFnValidity;
use crate::kernel::ExecuteParentKernel;

/// Project replacement validity directly from the base and patch validity masks.
#[derive(Debug)]
pub(in crate::arrays::patched) struct PatchValidity;

impl ExecuteParentKernel<Patched> for PatchValidity {
    type Parent = ScalarFnValidity;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Patched>,
        _parent: ArrayView<'_, ScalarFnValidity>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let base = array
            .inner()
            .validity()?
            .execute_mask(array.len(), ctx)?
            .into_bit_buffer();
        let mut bits = base
            .try_into_mut()
            .unwrap_or_else(|b| BitBufferMut::copy_from(&b));
        let valid = array
            .patch_values()
            .validity()?
            .execute_mask(array.patch_values().len(), ctx)?;
        visit_chunks::<false>(
            array.patch_indices(),
            array.patch_chunk_offsets(),
            array.offset,
            array.len(),
            ctx,
            |base, first, local| {
                for (i, &row) in local.iter().enumerate() {
                    bits.set_to(
                        base + usize::from(row) - array.offset,
                        valid.value(first + i),
                    );
                }
            },
        )?;
        Ok(Some(BoolArray::from(bits.freeze()).into_array()))
    }
}
