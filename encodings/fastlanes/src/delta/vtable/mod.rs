// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use fastlanes::FastLanes;
use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::DeltaData;
use crate::delta::array::NUM_SLOTS;
use crate::delta::array::SLOT_NAMES;
use crate::delta::array::delta_decompress::delta_decompress;

mod operations;
mod rules;
mod slice;
mod validity;

vtable!(Delta, Delta, DeltaData);

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl VTable for Delta {
    type ArrayData = DeltaData;

    type Metadata = ProstMetadata<DeltaMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &DeltaData) -> &Self {
        &Delta
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &DeltaData) -> usize {
        array.len()
    }

    fn dtype(array: &DeltaData) -> &DType {
        array.dtype()
    }

    fn stats(array: &DeltaData) -> &ArrayStats {
        array.stats_set()
    }

    fn array_hash<H: std::hash::Hasher>(array: &DeltaData, state: &mut H, precision: Precision) {
        array.offset().hash(state);
        array.bases().array_hash(state, precision);
        array.deltas().array_hash(state, precision);
    }

    fn array_eq(array: &DeltaData, other: &DeltaData, precision: Precision) -> bool {
        array.offset() == other.offset()
            && array.bases().array_eq(other.bases(), precision)
            && array.deltas().array_eq(other.deltas(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        rules::RULES.evaluate(array, parent, child_idx)
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "DeltaArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata {
            deltas_len: array.deltas().len() as u64,
            offset: array.offset() as u32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata::decode(bytes)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DeltaData> {
        assert_eq!(children.len(), 2);
        let ptype = PType::try_from(dtype)?;
        let lanes = match_each_unsigned_integer_ptype!(ptype, |T| { <T as FastLanes>::LANES });

        // Compute the length of the bases array
        let deltas_len = usize::try_from(metadata.0.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.0.deltas_len))?;
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if deltas_len % 1024 > 0 { 1 } else { 0 };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        DeltaData::try_new(bases, deltas, metadata.0.offset as usize, len)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            delta_decompress(&array, ctx)?.into_array(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct Delta;

impl Delta {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.delta");

    /// Compress a primitive array using Delta encoding.
    pub fn try_from_primitive_array(
        array: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DeltaArray> {
        Array::try_from_data(DeltaData::try_from_primitive_array(array, ctx)?)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;
    use super::ProstMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            ProstMetadata(DeltaMetadata {
                offset: u32::MAX,
                deltas_len: u64::MAX,
            }),
        );
    }
}
