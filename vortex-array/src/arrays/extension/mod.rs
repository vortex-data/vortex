use std::sync::{Arc, RwLock};

use arrow_array::builder::ArrayBuilder;
use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::encoding::encoding_ids;
use crate::stats::{Stat, StatsSet};
use crate::variants::ExtensionArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayValidityImpl, ArrayVariantsImpl,
    Canonical, EmptyMetadata, Encoding, EncodingId,
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
    const ID: EncodingId = EncodingId("vortex.ext", encoding_ids::EXTENSION);
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

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        todo!()
    }
}

impl ArrayValidityImpl for ExtensionArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        todo!()
    }
}

impl ArrayVariantsImpl for ExtensionArray {}

impl ValidateVTable<ExtensionArray> for ExtensionEncoding {}

impl VariantsVTable<ExtensionArray> for ExtensionEncoding {
    fn as_extension_array<'a>(
        &self,
        array: &'a ExtensionArray,
    ) -> Option<&'a dyn ExtensionArrayTrait> {
        Some(array)
    }
}

impl ExtensionArrayTrait for ExtensionArray {
    fn storage_data(&self) -> ArrayRef {
        self.storage()
    }
}

impl CanonicalVTable<ExtensionArray> for ExtensionEncoding {
    fn into_canonical(&self, array: ExtensionArray) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(array))
    }
}

impl ValidityVTable<ExtensionArray> for ExtensionEncoding {
    fn is_valid(&self, array: &ExtensionArray, index: usize) -> VortexResult<bool> {
        array.storage().is_valid(index)
    }

    fn all_valid(&self, array: &ExtensionArray) -> VortexResult<bool> {
        array.storage().all_valid()
    }

    fn all_invalid(&self, array: &ExtensionArray) -> VortexResult<bool> {
        array.storage().all_invalid()
    }

    fn validity_mask(&self, array: &ExtensionArray) -> VortexResult<Mask> {
        array.storage().validity_mask()
    }
}

impl VisitorVTable<ExtensionArray> for ExtensionEncoding {
    fn accept(&self, array: &ExtensionArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("storage", &array.storage())
    }
}

impl StatisticsVTable<ExtensionArray> for ExtensionEncoding {
    fn compute_statistics(&self, array: &ExtensionArray, stat: Stat) -> VortexResult<StatsSet> {
        array.storage().statistics().compute_all(&[stat])
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
