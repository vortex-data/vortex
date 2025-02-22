use std::sync::{Arc, RwLock};

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::encoding::encoding_ids;
use crate::stats::{Stat, StatsSet};
use crate::vtable::{StatisticsVTable, VTableRef};
use crate::{
    Array, ArrayBufferVisitor, ArrayImpl, ArrayStatistics, ArrayStatisticsImpl, ArrayVariantsImpl,
    ArrayVisitorImpl, EmptyMetadata, Encoding, EncodingId,
};

mod canonical;
mod compute;
mod serde;
mod variants;

#[derive(Clone, Debug)]
pub struct ConstantArray {
    scalar: Scalar,
    len: usize,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct ConstantEncoding;
impl Encoding for ConstantEncoding {
    const ID: EncodingId = EncodingId::new("vortex.constant", encoding_ids::CONSTANT);
    type Array = ConstantArray;
    type Metadata = EmptyMetadata;
}

impl ConstantArray {
    pub fn new<S>(scalar: S, len: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        let stats = StatsSet::constant(scalar.clone(), len);
        Self {
            scalar,
            len,
            stats_set: Arc::new(RwLock::new(stats)),
        }
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }
}

impl ArrayImpl for ConstantArray {
    type Encoding = ConstantEncoding;

    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::from_static(&ConstantEncoding)
    }
}

impl ArrayValidityImpl for ConstantArray {
    fn _is_valid(&self, _index: usize) -> VortexResult<bool> {
        Ok(!self.scalar().is_null())
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        Ok(!self.scalar().is_null())
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        Ok(self.scalar().is_null())
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        Ok(match self.scalar().is_null() {
            true => Mask::AllFalse(self.len()),
            false => Mask::AllTrue(self.len()),
        })
    }
}

impl ArrayStatisticsImpl for ConstantArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

impl StatisticsVTable<&ConstantArray> for ConstantEncoding {}
