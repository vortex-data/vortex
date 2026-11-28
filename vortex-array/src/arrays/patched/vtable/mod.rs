// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(super) mod kernels;
mod operations;
mod plugin;
mod slice;
#[cfg(test)]
mod tests;

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
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
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::patched::BoolPatchStrategy;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::arrays::patched::indices::visit_chunks;
use crate::block::Block;
use crate::block::BlockOp;
use crate::block::Values;
use crate::block::Written;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::require_child;
use crate::require_spine;
use crate::search_sorted::decode_sorted_positions;

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
        vortex_ensure!(
            slots.patch_indices.dtype().is_unsigned_int()
                && slots.patch_indices.dtype().as_ptype() == PType::U16
        );
        for positions in [slots.patch_indices, slots.patch_chunk_offsets] {
            vortex_ensure!(positions.dtype().is_unsigned_int() && !positions.dtype().is_nullable());
        }
        let offset = data.offset;
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
        let (PatchStrategy::Bool(BoolPatchStrategy::Replace)
        | PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let array =
            require_child!(array, array.patch_indices(), PatchedSlots::PATCH_INDICES => Primitive);
        if array.dtype().is_boolean() {
            let array =
                require_child!(array, array.patch_values(), PatchedSlots::PATCH_VALUES => Bool);
            let array = require_spine!(array, array.inner(), PatchedSlots::INNER => Bool);
            let validity = array.validity()?;
            let values = array.patch_values().as_::<Bool>();
            let replacement_bits = values.bit_buffer_view();
            let mut first = 0;
            let base = array.inner().clone().downcast::<Bool>().into_bit_buffer();
            let mut output = base
                .try_into_mut()
                .unwrap_or_else(|bits| BitBufferMut::copy_from(&bits));
            // Accessing typed array data performs a downcast. Resolve it once before the scatter.
            let offset = array.offset;
            let bit_offset = output.offset();
            let bytes = output.as_mut_slice();
            visit_chunks::<false>(
                array.patch_indices(),
                array.patch_chunk_offsets(),
                offset,
                array.len(),
                ctx,
                |base, local| {
                    let replacements = replacement_bits.slice(first..first + local.len());
                    first += local.len();
                    for (&position, value) in local.iter().zip(replacements.iter()) {
                        let bit = base + usize::from(position) - offset + bit_offset;
                        let shift = bit % 8;
                        let byte = &mut bytes[bit / 8];
                        *byte = (*byte & !(1 << shift)) | (u8::from(value) << shift);
                    }
                },
            )?;
            return Ok(ExecutionResult::done(
                BoolArray::new(output.freeze(), validity).into_array(),
            ));
        }
        let array =
            require_child!(array, array.patch_values(), PatchedSlots::PATCH_VALUES => Primitive);
        let array = require_spine!(array, array.inner(), PatchedSlots::INNER => Primitive);
        let patches = array.absolute_patches(ctx)?;
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
        if slot_idx != PatchedSlots::INNER || array.dtype().is_boolean() {
            return Ok(None);
        }
        let (PatchStrategy::Bool(BoolPatchStrategy::Replace)
        | PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let indices = array
            .patch_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .to_buffer::<u16>();
        let offsets = decode_sorted_positions(array.patch_chunk_offsets(), ctx)?;
        let values = array
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let valid = values.validity()?.execute_mask(values.len(), ctx)?;
        match_each_native_ptype!(array.dtype().as_ptype(), |T| {
            Ok(Some(Box::new(Replace {
                indices,
                offsets,
                values: values.to_buffer::<T>(),
                valid,
                offset: array.offset,
                chunk: 0,
                position: 0,
            }) as Box<dyn BlockOp>))
        })
    }
}

struct Replace<T: NativePType> {
    indices: Buffer<u16>,
    offsets: Buffer<u64>,
    values: Buffer<T>,
    valid: Mask,
    offset: usize,
    chunk: usize,
    position: usize,
}

impl<T: NativePType> BlockOp for Replace<T> {
    fn process<'a>(&mut self, mut block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let mut written = match block.values {
            Values::InPlace(values) => values.written(),
            values => values.map(|v: T| v),
        };
        let values = written.as_mut_slice::<T>();
        let begin = self.offset + start;
        let end = begin + values.len();
        while self.position < self.indices.len() {
            while self
                .offsets
                .get(self.chunk + 1)
                .is_some_and(|&next| next <= self.position as u64)
            {
                self.chunk += 1;
            }
            let row = (self.offset / PATCH_CHUNK_SIZE + self.chunk) * PATCH_CHUNK_SIZE
                + usize::from(self.indices[self.position]);
            if row >= end {
                break;
            }
            if row >= begin {
                values[row - begin] = self.values[self.position];
                block
                    .validity
                    .set(row - begin, self.valid.value(self.position));
            }
            self.position += 1;
        }
        Ok(written)
    }
}
