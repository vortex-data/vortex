use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::encoding::encoding_ids;
use crate::nbytes::NBytes;
use crate::stats::{ArrayStats, Precision, Stat, StatsSet, StatsSetRef};
use crate::variants::NullArrayTrait;
use crate::vtable::{StatisticsVTable, VTableRef};
use crate::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, EmptyMetadata, Encoding, EncodingId,
};

mod compute;
mod serde;

#[derive(Clone, Debug)]
pub struct NullArray {
    len: usize,
    stats_set: ArrayStats,
}

pub struct NullEncoding;
impl Encoding for NullEncoding {
    const ID: EncodingId = EncodingId::new("vortex.null", encoding_ids::NULL);
    type Array = NullArray;
    type Metadata = EmptyMetadata;
}

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            stats_set: Default::default(),
        }
    }
}

impl ArrayImpl for NullArray {
    type Encoding = NullEncoding;

    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        &DType::Null
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&NullEncoding)
    }
}

impl ArrayStatisticsImpl for NullArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
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
        Ok(self.is_empty())
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        Ok(!self.is_empty())
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        Ok(Mask::AllFalse(self.len))
    }
}

impl StatisticsVTable<&NullArray> for NullEncoding {
    fn compute_statistics(&self, array: &NullArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, Precision::exact(array.nbytes())));
        }

        Ok(StatsSet::nulls(array.len()))
    }
}

impl ArrayVariantsImpl for NullArray {
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for NullArray {}
