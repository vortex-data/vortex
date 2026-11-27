// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(super) mod kernels;
mod operations;
mod plugin;
mod slice;
#[cfg(test)]
mod tests;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::Array;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::Primitive;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::block::Block;
use crate::block::BlockOp;
use crate::block::BlockPatches;
use crate::block::Values;
use crate::block::Written;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::require_child;
use crate::require_spine;

impl VTable for Patched {
    type TypedArrayData = PatchedData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.patched");
        *ID
    }

    fn validate(
        &self,
        data: &PatchedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), PatchedSlots::COUNT);
        vortex_ensure!(slots.iter().all(Option::is_some));
        let slots = PatchedSlotsView::from_slots(slots);
        data.strategy.validate(dtype)?;
        vortex_ensure_eq!(slots.inner.dtype(), dtype);
        vortex_ensure_eq!(slots.inner.len(), len);
        vortex_ensure_eq!(slots.patch_values.dtype(), dtype);
        vortex_ensure_eq!(slots.patch_indices.len(), slots.patch_values.len());
        vortex_ensure!((1..=len).contains(&slots.patch_indices.len()));
        for positions in [slots.patch_indices, slots.patch_chunk_offsets] {
            vortex_ensure!(positions.dtype().is_unsigned_int() && !positions.dtype().is_nullable());
        }
        let offset = data.patches.offset();
        let end = offset
            .checked_add(len)
            .ok_or_else(|| vortex_err!("Patch range overflow"))?;
        let chunks = end.div_ceil(PATCH_CHUNK_SIZE) - offset / PATCH_CHUNK_SIZE;
        vortex_ensure_eq!(slots.patch_chunk_offsets.len(), chunks);
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Invalid Patched buffer index {idx}")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PatchedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let array =
            require_child!(array, array.patch_indices(), PatchedSlots::PATCH_INDICES => Primitive);
        let array =
            require_child!(array, array.patch_values(), PatchedSlots::PATCH_VALUES => Primitive);
        let array = require_spine!(array, array.inner(), PatchedSlots::INNER => Primitive);
        let patches = array.patches();
        let inner = array.inner().clone().downcast::<Primitive>();
        drop(array);
        Ok(ExecutionResult::done(
            inner.patch(&patches, ctx)?.into_array(),
        ))
    }

    fn block_op(
        array: ArrayView<'_, Self>,
        slot_idx: usize,
        _child_dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn BlockOp>>> {
        if slot_idx != PatchedSlots::INNER {
            return Ok(None);
        }
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let patches = array.patches();
        match_each_native_ptype!(array.dtype().as_ptype(), |T| {
            let patches = BlockPatches::try_new(&patches, ctx, |_, value: T| value)?;
            Ok(Some(Box::new(Replace(patches)) as Box<dyn BlockOp>))
        })
    }
}

struct Replace<T>(BlockPatches<T>);

impl<T: NativePType> BlockOp for Replace<T> {
    fn process<'a>(&mut self, mut block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let mut written = match block.values {
            Values::InPlace(values) => values.written(),
            values => values.map(|v: T| v),
        };
        self.0
            .apply(start, written.as_mut_slice::<T>(), &mut block.validity);
        Ok(written)
    }
}
