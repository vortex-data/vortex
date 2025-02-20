use std::sync::{Arc, RwLock};

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::encoding::encoding_ids;
use crate::stats::{ArrayStatistics, Stat, StatsSet};
use crate::visitor::ArrayVisitor;
use crate::{
    Array, ArrayImpl, ArrayVariantsImpl, ArrayVisitorImpl, EmptyMetadata, Encoding, EncodingId,
};

mod canonical;
// mod compute;
mod variants;

#[derive(Clone)]
pub struct ConstantArray {
    scalar: Scalar,
    len: usize,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct ConstantEncoding;
impl Encoding for ConstantEncoding {
    const ID: EncodingId = EncodingId("vortex.constant", encoding_ids::CONSTANT);
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
    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        self.scalar.dtype()
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

impl ArrayStatistics for ConstantArray {
    fn stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }

    fn compute_statistic(&self, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::constant(self.scalar.clone(), self.len()))
    }
}

impl ArrayVisitorImpl for ConstantArray {
    fn _accept(&self, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        // visitor.visit_buffer(array.byte_buffer(0).vortex_expect("missing scalar buffer"))
        Ok(())
    }
}
