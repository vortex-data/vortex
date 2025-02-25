use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, DeserializeMetadata,
    RkyvMetadata,
};
use vortex_dtype::{DType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{DeltaArray, DeltaEncoding};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct DeltaMetadata {
    // TODO(ngates): do we need any of this?
    deltas_len: u64,
    offset: u16, // must be <1024
}

impl ArrayVisitorImpl<RkyvMetadata<DeltaMetadata>> for DeltaArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("bases", self.bases());
        visitor.visit_child("deltas", self.deltas());
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<DeltaMetadata> {
        RkyvMetadata(DeltaMetadata {
            deltas_len: self.deltas().len() as u64,
            offset: self.offset() as u16,
        })
    }
}

impl SerdeVTable<&DeltaArray> for DeltaEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = RkyvMetadata::<DeltaMetadata>::deserialize(parts.metadata())?;

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

#[cfg(test)]
mod test {
    use vortex_array::RkyvMetadata;
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            RkyvMetadata(DeltaMetadata {
                offset: u16::MAX,
                deltas_len: u64::MAX,
            }),
        );
    }
}
