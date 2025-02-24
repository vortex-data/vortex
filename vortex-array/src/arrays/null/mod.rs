use std::sync::{Arc, RwLock};

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::encoding::encoding_ids;
use crate::nbytes::NBytes;
use crate::stats::{Precision, Stat, StatsSet};
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
    stats_set: Arc<RwLock<StatsSet>>,
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
        VTableRef::from_static(&NullEncoding)
    }
}

impl ArrayStatisticsImpl for NullArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
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

        Ok(StatsSet::nulls(array.len(), &DType::Null))
    }
}

impl ArrayVariantsImpl for NullArray {
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for NullArray {}
