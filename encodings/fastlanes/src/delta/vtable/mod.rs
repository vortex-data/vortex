// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::Range;

use fastlanes::FastLanes;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChildSliceHelper;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::DeltaArray;

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

vtable!(Delta);

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl VTable for DeltaVTable {
    type Array = DeltaArray;

    type Metadata = ProstMetadata<DeltaMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("fastlanes.delta")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DeltaVTable.as_vtable()
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let physical_start = range.start + array.offset();
        let physical_stop = range.end + array.offset();

        let start_chunk = physical_start / 1024;
        let stop_chunk = physical_stop.div_ceil(1024);

        let bases = array.bases();
        let deltas = array.deltas();
        let lanes = array.lanes();

        let new_bases = bases.slice(
            min(start_chunk * lanes, array.bases_len())..min(stop_chunk * lanes, array.bases_len()),
        );

        let new_deltas = deltas.slice(
            min(start_chunk * 1024, array.deltas_len())..min(stop_chunk * 1024, array.deltas_len()),
        );

        // SAFETY: slicing valid bases/deltas preserves correctness
        Ok(Some(unsafe {
            DeltaArray::new_unchecked(new_bases, new_deltas, physical_start % 1024, range.len())
                .into_array()
        }))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // DeltaArray children order (from visit_children):
        // 1. bases
        // 2. deltas

        vortex_ensure!(
            children.len() == 2,
            "Expected 2 children for Delta encoding, got {}",
            children.len()
        );

        array.bases = children[0].clone();
        array.deltas = children[1].clone();

        Ok(())
    }

    fn metadata(array: &DeltaArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata {
            deltas_len: array.deltas().len() as u64,
            offset: array.offset() as u32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata::decode(buffer)?))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DeltaArray> {
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

        DeltaArray::try_new(bases, deltas, metadata.0.offset as usize, len)
    }

    // TODO(joe): impl execute without canonical
}

#[derive(Debug)]
pub struct DeltaVTable;

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
