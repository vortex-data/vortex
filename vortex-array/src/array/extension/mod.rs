use std::fmt::{Debug, Display};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::stats::{ArrayStatistics as _, Stat, StatisticsVTable, StatsSet};
use crate::validate::ValidateVTable;
use crate::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use crate::variants::{ExtensionArrayTrait, VariantsVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, Canonical, EmptyMetadata, IntoCanonical,
};
mod compute;

impl_encoding!("vortex.ext", ids::EXTENSION, Extension, EmptyMetadata);

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: ArrayData) -> Self {
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

    pub fn storage(&self) -> ArrayData {
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
    fn storage_data(&self) -> ArrayData {
        self.storage()
    }
}

impl IntoCanonical for ExtensionArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(self))
    }
}

impl ValidityVTable<ExtensionArray> for ExtensionEncoding {
    fn is_valid(&self, array: &ExtensionArray, index: usize) -> VortexResult<bool> {
        array.storage().is_valid(index)
    }

    fn logical_validity(&self, array: &ExtensionArray) -> VortexResult<LogicalValidity> {
        array.storage().logical_validity()
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
    use crate::IntoArrayData;

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

        assert_eq!(stats.get_as::<i64>(Stat::Min), Some(1i64));
        assert_eq!(stats.get_as::<i64>(Stat::Max), Some(5_i64));
        assert_eq!(stats.get_as::<usize>(Stat::NullCount), Some(0));
    }
}
