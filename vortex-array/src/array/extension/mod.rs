use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::encoding::encoding_ids;
use crate::stats::{Stat, StatsSet};
use crate::variants::ExtensionArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::vtable::{
    CanonicalVTable, StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable,
    VisitorVTable,
};
use crate::{impl_encoding, Array, Canonical, EmptyMetadata};

mod compute;

impl_encoding!(
    "vortex.ext",
    encoding_ids::EXTENSION,
    Extension,
    EmptyMetadata
);

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: Array) -> Self {
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );

        Self::try_from_parts(
            DType::Extension(ext_dtype),
            storage.len(),
            EmptyMetadata,
            None,
            Some([storage].into()),
            StatsSet::default(),
        )
        .vortex_expect("Invalid ExtensionArray")
    }

    pub fn storage(&self) -> Array {
        self.as_ref()
            .child(0, self.ext_dtype().storage_dtype(), self.len())
            .vortex_expect("Missing storage array for ExtensionArray")
    }

    #[allow(dead_code)]
    #[inline]
    pub fn id(&self) -> &ExtID {
        self.ext_dtype().id()
    }
}

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
    fn storage_data(&self) -> Array {
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
