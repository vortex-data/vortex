use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    EncodingId, ProstMetadata,
};
use vortex_dtype::{DType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::DeltaEncoding;
use crate::DeltaArray;

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl EncodingVTable for DeltaEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("fastlanes.delta")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = ProstMetadata::<DeltaMetadata>::deserialize(parts.metadata())?;

        let validity = if parts.nchildren() == 2 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 3 {
            let validity = parts.child(2).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "DeltaArray: expected 2 or 3 children, got {}",
                parts.nchildren()
            );
        };

        let ptype = PType::try_from(&dtype)?;
        let lanes = match_each_unsigned_integer_ptype!(ptype, |$T| {
            <$T as fastlanes::FastLanes>::LANES
        });

        // Compute the length of the bases array
        let deltas_len = usize::try_from(metadata.deltas_len)
            .vortex_expect("DeltaArray: deltas_len must be a valid usize");
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if deltas_len % 1024 > 0 { 1 } else { 0 };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = parts.child(0).decode(ctx, dtype.clone(), bases_len)?;
        let deltas = parts.child(1).decode(ctx, dtype, deltas_len)?;

        Ok(
            DeltaArray::try_new(bases, deltas, validity, metadata.offset as usize, len)?
                .into_array(),
        )
    }
}

impl ArrayVisitorImpl<ProstMetadata<DeltaMetadata>> for DeltaArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("bases", self.bases());
        visitor.visit_child("deltas", self.deltas());
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> ProstMetadata<DeltaMetadata> {
        ProstMetadata(DeltaMetadata {
            deltas_len: self.deltas().len() as u64,
            offset: self.offset() as u32,
        })
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
