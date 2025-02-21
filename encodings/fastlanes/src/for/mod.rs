use std::fmt::Debug;
use std::sync::{Arc, RwLock};

pub use compress::*;
use serde::{Deserialize, Serialize};
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{StatisticsVTable, VTableRef};
use vortex_array::{
    encoding_ids, Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariantsImpl, ArrayVisitorImpl, Canonical, EmptyMetadata, Encoding,
    EncodingId,
};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::{PValue, Scalar};

mod compress;
mod compute;

#[derive(Clone, Debug)]
pub struct FoRArray {
    encoded: ArrayRef,
    reference: Scalar,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct FoREncoding;
impl Encoding for FoREncoding {
    const ID: EncodingId = EncodingId::new("fastlanes.for", encoding_ids::FL_FOR);
    type Array = FoRArray;
    type Metadata = EmptyMetadata;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct FoRMetadata {
    reference: PValue,
}

impl FoRArray {
    pub fn try_new(encoded: ArrayRef, reference: Scalar) -> VortexResult<Self> {
        if reference.is_null() {
            vortex_bail!("Reference value cannot be null");
        }
        let reference = reference.cast(
            &reference
                .dtype()
                .with_nullability(encoded.dtype().nullability()),
        )?;
        Ok(Self {
            encoded,
            reference,
            stats_set: Default::default(),
        })
    }

    #[inline]
    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }

    #[inline]
    pub fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }
}

impl ArrayImpl for FoRArray {
    type Encoding = FoREncoding;

    fn _len(&self) -> usize {
        self.encoded().len()
    }

    fn _dtype(&self) -> &DType {
        self.reference_scalar().dtype()
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::from_static(&FoREncoding)
    }
}

impl ArrayCanonicalImpl for FoRArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        decompress(self).map(Canonical::Primitive)
    }
}

impl ArrayStatisticsImpl for FoRArray {
    fn stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

impl ArrayValidityImpl for FoRArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.encoded().is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.encoded().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.encoded().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.encoded().validity_mask()
    }
}

impl ArrayVariantsImpl for FoRArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl ArrayVisitorImpl for FoRArray {
    fn _accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", self.encoded())
    }
}

impl StatisticsVTable<'_, FoRArray> for FoREncoding {}

impl PrimitiveArrayTrait for FoRArray {}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
    use vortex_scalar::PValue;

    use crate::FoRMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_for_metadata() {
        check_metadata(
            "for.metadata",
            SerdeMetadata(FoRMetadata {
                reference: PValue::I64(i64::MAX),
            }),
        );
    }
}
