use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::DeltaEncoding;
use crate::{DeltaArray, DeltaVTable};

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl SerdeVTable<DeltaVTable> for DeltaVTable {
    type Metadata = ProstMetadata<DeltaMetadata>;

    fn metadata(array: &DeltaArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(DeltaMetadata {
            deltas_len: array.deltas().len() as u64,
            offset: array.offset() as u32,
        })))
    }

    fn build(
        _encoding: &DeltaEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DeltaArray> {
        let validity = if children.len() == 2 {
            Validity::from(dtype.nullability())
        } else if children.len() == 3 {
            let validity = children.get(2, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "DeltaArray: expected 2 or 3 children, got {}",
                children.len()
            );
        };

        let ptype = PType::try_from(dtype)?;
        let lanes = match_each_unsigned_integer_ptype!(ptype, |$T| {
            <$T as fastlanes::FastLanes>::LANES
        });

        // Compute the length of the bases array
        let deltas_len = usize::try_from(metadata.deltas_len)
            .vortex_expect("DeltaArray: deltas_len must be a valid usize");
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if deltas_len % 1024 > 0 { 1 } else { 0 };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        DeltaArray::try_new(bases, deltas, validity, metadata.offset as usize, len)
    }
}

impl VisitorVTable<DeltaVTable> for DeltaVTable {
    fn visit_buffers(_array: &DeltaArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DeltaArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("bases", array.bases());
        visitor.visit_child("deltas", array.deltas());
        visitor.visit_validity(array.validity(), array.len());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;

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
