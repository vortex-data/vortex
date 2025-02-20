use std::sync::{Arc, RwLock};

use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::encoding::encoding_ids;
use crate::stats::{ArrayStatistics, Stat, StatsSet};
use crate::variants::ExtensionArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::{
    Array, ArrayImpl, ArrayRef, ArrayVariantsImpl, ArrayVisitorImpl, Canonical, EmptyMetadata,
    Encoding, EncodingId,
};
// mod compute;

#[derive(Clone, Debug)]
pub struct ExtensionArray {
    dtype: DType,
    storage: ArrayRef,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct ExtensionEncoding;
impl Encoding for ExtensionEncoding {
    const ID: EncodingId = EncodingId::new("vortex.ext", encoding_ids::EXTENSION);
    type Array = ExtensionArray;
    type Metadata = EmptyMetadata;
}

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: ArrayRef) -> Self {
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );
        Self {
            dtype: DType::Extension(ext_dtype),
            storage,
            stats_set: Arc::new(RwLock::new(StatsSet::default())),
        }
    }

    pub fn storage(&self) -> &ArrayRef {
        &self.storage
    }

    #[allow(dead_code)]
    #[inline]
    pub fn id(&self) -> &ExtID {
        self.ext_dtype().id()
    }
}

impl ArrayImpl for ExtensionArray {
    fn _len(&self) -> usize {
        self.storage.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }
}

impl ArrayCanonicalImpl for ExtensionArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(self.clone()))
    }
}

impl ArrayValidityImpl for ExtensionArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.storage.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.storage.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.storage.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.storage.validity_mask()
    }
}

impl ArrayVariantsImpl for ExtensionArray {
    fn _as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        Some(self)
    }
}

impl ExtensionArrayTrait for ExtensionArray {
    fn storage_data(&self) -> ArrayRef {
        self.storage().clone()
    }
}

impl ArrayVisitorImpl for ExtensionArray {
    fn _accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("storage", self.storage())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::PType;

    use super::*;
    use crate::stats::Precision;
    use crate::IntoArray;

    #[test]
    fn compute_statistics() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("timestamp".into()),
            DType::from(PType::I64).into(),
            None,
        ));
        let array = ExtensionArray::new(ext_dtype, buffer![1i64, 2, 3, 4, 5].into_array());

        let stats = array
            .statistics()
            .compute_all(&[Stat::Min, Stat::Max, Stat::NullCount])
            .unwrap();
        let num_stats = stats.clone().into_iter().count();
        assert!(
            num_stats >= 3,
            "Expected at least 3 stats, got {}",
            num_stats
        );

        assert_eq!(stats.get_as::<i64>(Stat::Min), Some(Precision::exact(1i64)));
        assert_eq!(
            stats.get_as::<i64>(Stat::Max),
            Some(Precision::exact(5_i64))
        );
        assert_eq!(
            stats.get_as::<usize>(Stat::NullCount),
            Some(Precision::exact(0usize))
        );
    }
}
