use vortex_array::{Array, ArrayChildVisitor, ArrayVisitorImpl, RkyvMetadata};

use crate::DeltaArray;

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct DeltaMetadata {
    // TODO(ngates): do we need any of this?
    pub(crate) deltas_len: u64,
    pub(crate) offset: u16, // must be <1024
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
