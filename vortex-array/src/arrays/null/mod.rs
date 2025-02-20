use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::encoding::encoding_ids;
use crate::stats::{Precision, Stat, StatsSet};
use crate::validity::Validity;
use crate::variants::NullArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::{
    ArrayCanonicalImpl, ArrayValidityImpl, ArrayVariantsImpl, ArrayVisitorImpl, Canonical,
    EmptyMetadata, Encoding, EncodingId,
};

// mod compute;

#[derive(Clone, Debug)]
pub struct NullArray {
    len: usize,
}

pub struct NullEncoding;
impl Encoding for NullEncoding {
    const ID: EncodingId = EncodingId("vortex.null", encoding_ids::NULL);
    type Array = NullArray;
    type Metadata = EmptyMetadata;
}

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ArrayCanonicalImpl for NullArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Null(self.clone()))
    }
}

impl ArrayValidityImpl for NullArray {
    fn _is_valid(&self, _index: usize) -> VortexResult<bool> {
        Ok(false)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        Ok(self.len == 0)
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        Ok(self.len > 0)
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        Ok(Mask::AllFalse(self.len))
    }
}

// impl StatisticsVTable<NullArray> for NullEncoding {
//     fn compute_statistics(&self, array: &NullArray, stat: Stat) -> VortexResult<StatsSet> {
//         if stat == Stat::UncompressedSizeInBytes {
//             return Ok(StatsSet::of(stat, Precision::exact(array.nbytes())));
//         }
//
//         Ok(StatsSet::nulls(array.len(), &DType::Null))
//     }
// }

impl ArrayVisitorImpl for NullArray {
    fn _accept(&self, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        // No children.
        Ok(())
    }
}

impl ArrayVariantsImpl for NullArray {
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for NullArray {}
