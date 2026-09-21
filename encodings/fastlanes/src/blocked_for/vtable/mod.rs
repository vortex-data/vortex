// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BlockedFoRData;
use crate::blocked_for::array::BLOCK_SIZE;
use crate::blocked_for::array::BlockedFoRArrayExt;
use crate::blocked_for::array::BlockedFoRSlots;
use crate::blocked_for::array::BlockedFoRSlotsView;
use crate::blocked_for::array::blocked_for_decompress::decompress;
use crate::blocked_for::array::num_blocks;
use crate::blocked_for::vtable::rules::PARENT_RULES;

mod operations;
mod rules;
mod slice;
mod validity;

/// A [`BlockedFoR`]-encoded Vortex array.
pub type BlockedFoRArray = Array<BlockedFoR>;

#[derive(Clone, prost::Message)]
pub struct BlockedFoRMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) offset: u32, // must be < BLOCK_SIZE
}

impl ArrayHash for BlockedFoRData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for BlockedFoRData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for BlockedFoR {
    type TypedArrayData = BlockedFoRData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.blockedfor");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = BlockedFoRSlotsView::from_slots(slots);
        validate_parts(slots.encoded, slots.references, data.offset, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("BlockedFoRArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        BlockedFoRSlots::NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            BlockedFoRMetadata {
                offset: array.offset() as u32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "BlockedFoRArray expects 0 buffers, got {}",
            buffers.len()
        );
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for BlockedFoR encoding, found {}",
                children.len()
            )
        }

        let metadata = BlockedFoRMetadata::decode(metadata)?;
        let offset = u16::try_from(metadata.offset)?;

        let encoded = children.get(0, dtype, len)?;
        let references = children.get(1, &dtype.as_nonnullable(), num_blocks(len, offset))?;
        let slots = smallvec![Some(encoded), Some(references)];

        let data = BlockedFoRData::try_new(offset)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(decompress(&array, ctx)?.into_array()))
    }
}

/// Block-wise frame-of-reference encoding.
#[derive(Clone, Debug)]
pub struct BlockedFoR;

impl BlockedFoR {
    /// Construct a new blocked FoR array from residuals and one reference per block.
    ///
    /// `offset` is the position of the first value within its block, so `references` must hold
    /// `(len + offset).div_ceil(BLOCK_SIZE)` values.
    pub fn try_new(
        encoded: ArrayRef,
        references: ArrayRef,
        offset: u16,
    ) -> VortexResult<BlockedFoRArray> {
        let dtype = encoded.dtype().clone();
        let len = encoded.len();
        validate_parts(&encoded, &references, offset, &dtype, len)?;
        let data = BlockedFoRData::try_new(offset)?;
        let slots = smallvec![Some(encoded), Some(references)];
        Array::try_from_parts(ArrayParts::new(BlockedFoR, dtype, len, data).with_slots(slots))
    }

    /// Encode a primitive array, subtracting a per-block minimum from every block of
    /// [`BLOCK_SIZE`] values.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<BlockedFoRArray> {
        BlockedFoRData::encode(array, ctx)
    }
}

fn validate_parts(
    encoded: &ArrayRef,
    references: &ArrayRef,
    offset: u16,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        dtype.is_int(),
        "BlockedFoR requires an integer dtype, got {dtype}"
    );
    vortex_ensure!(
        (offset as usize) < BLOCK_SIZE,
        "BlockedFoR offset must be less than {BLOCK_SIZE}, got {offset}"
    );
    vortex_ensure!(
        encoded.dtype() == dtype,
        "BlockedFoR encoded dtype mismatch: expected {dtype}, got {}",
        encoded.dtype()
    );
    vortex_ensure!(
        encoded.len() == len,
        "BlockedFoR encoded length mismatch: expected {len}, got {}",
        encoded.len()
    );
    vortex_ensure!(
        references.dtype().nullability() == Nullability::NonNullable,
        "BlockedFoR references must be non-nullable, got {}",
        references.dtype()
    );
    vortex_ensure!(
        references.dtype() == &dtype.as_nonnullable(),
        "BlockedFoR references dtype mismatch: expected {}, got {}",
        dtype.as_nonnullable(),
        references.dtype()
    );
    let expected_blocks = num_blocks(len, offset);
    vortex_ensure!(
        references.len() == expected_blocks,
        "BlockedFoR expects {expected_blocks} references for {len} values at offset {offset}, \
         got {}",
        references.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::test_harness::check_metadata;

    use super::BlockedFoRMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_blocked_for_metadata() {
        check_metadata(
            "blockedfor.metadata",
            &BlockedFoRMetadata { offset: 1023 }.encode_to_vec(),
        );
    }
}
