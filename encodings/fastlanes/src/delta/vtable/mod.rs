// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use fastlanes::FastLanes;
use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::VTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::DeltaData;
use crate::delta::array::BASES_SLOT;
use crate::delta::array::DELTAS_SLOT;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::SLOT_NAMES;
use crate::delta::array::delta_decompress::delta_decompress;
use crate::delta::array::lane_count;

mod operations;
mod rules;
mod slice;
mod validity;

/// A [`Delta`]-encoded Vortex array.
pub type DeltaArray = Array<Delta>;

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl ArrayHash for DeltaData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.offset.hash(state);
    }
}

impl ArrayEq for DeltaData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.offset == other.offset
    }
}

impl VTable for Delta {
    type ArrayData = DeltaData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.delta");
        *ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let bases = slots[BASES_SLOT]
            .as_ref()
            .vortex_expect("DeltaArray bases slot");
        let deltas = slots[DELTAS_SLOT]
            .as_ref()
            .vortex_expect("DeltaArray deltas slot");
        validate_parts(bases, deltas, data.offset, dtype, len)
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DeltaMetadata {
                deltas_len: array.deltas().len() as u64,
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
            "DeltaArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == 2,
            "DeltaArray expects 2 children, got {}",
            children.len()
        );
        let metadata = DeltaMetadata::decode(metadata)?;
        let ptype = PType::try_from(dtype)?;
        let lanes = match_each_unsigned_integer_ptype!(ptype, |T| { <T as FastLanes>::LANES });

        // Compute the length of the bases array
        let deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if deltas_len % 1024 > 0 { 1 } else { 0 };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        let data = DeltaData::try_new(metadata.offset as usize)?;
        let slots = vec![Some(bases), Some(deltas)];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
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
    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        len: usize,
    ) -> VortexResult<DeltaArray> {
        let dtype = bases.dtype().with_nullability(deltas.dtype().nullability());
        let data = DeltaData::try_new(offset)?;
        let slots = vec![Some(bases), Some(deltas)];
        Array::try_from_parts(ArrayParts::new(Delta, dtype, len, data).with_slots(slots))
    }

    /// Compress a primitive array using Delta encoding.
    pub fn try_from_primitive_array(
        array: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DeltaArray> {
        let logical_len = array.len();
        let (bases, deltas) = crate::delta::array::delta_compress::delta_compress(array, ctx)?;
        Self::try_new(bases.into_array(), deltas.into_array(), 0, logical_len)
    }
}

fn validate_parts(
    bases: &ArrayRef,
    deltas: &ArrayRef,
    offset: usize,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        offset + len <= deltas.len(),
        "offset + len, {offset} + {len}, must be less than or equal to the size of deltas: {}",
        deltas.len()
    );
    vortex_ensure!(
        bases.dtype().eq_ignore_nullability(deltas.dtype()),
        "DeltaArray: bases and deltas must have the same dtype, got {} and {}",
        bases.dtype(),
        deltas.dtype()
    );

    vortex_ensure!(
        bases.dtype().is_unsigned_int(),
        "DeltaArray: dtype must be an unsigned integer, got {}",
        bases.dtype()
    );

    let expected_dtype = bases.dtype().with_nullability(deltas.dtype().nullability());
    vortex_ensure!(
        dtype == &expected_dtype,
        "DeltaArray dtype mismatch: expected {expected_dtype}, got {dtype}"
    );

    let lanes = lane_count(bases.dtype().as_ptype());

    vortex_ensure!(
        deltas.len().is_multiple_of(1024),
        "deltas length ({}) must be a multiple of 1024",
        deltas.len(),
    );
    vortex_ensure!(
        bases.len().is_multiple_of(lanes),
        "bases length ({}) must be a multiple of LANES ({lanes})",
        bases.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            &DeltaMetadata {
                offset: u32::MAX,
                deltas_len: u64::MAX,
            }
            .encode_to_vec(),
        );
    }
}
