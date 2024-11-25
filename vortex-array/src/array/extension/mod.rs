use std::fmt::{Debug, Display};
use std::sync::Arc;

use enum_iterator::all;
use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::stats::{ArrayStatistics as _, Stat, StatisticsVTable, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use crate::variants::{ArrayVariants, ExtensionArrayTrait};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoCanonical};

mod compute;

impl_encoding!("vortex.ext", ids::EXTENSION, Extension);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMetadata;

impl Display for ExtensionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

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
            ExtensionMetadata,
            [storage].into(),
            Default::default(),
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

impl ArrayTrait for ExtensionArray {}

impl ArrayVariants for ExtensionArray {
    fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        Some(self)
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
    fn is_valid(&self, array: &ExtensionArray, index: usize) -> bool {
        array.storage().is_valid(index)
    }

    fn logical_validity(&self, array: &ExtensionArray) -> LogicalValidity {
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
        let mut stats = array.storage().statistics().compute_all(&[stat])?;

        // for e.g., min/max, we want to cast to the extension array's dtype
        // for other stats, we don't need to change anything
        for stat in all::<Stat>().filter(|s| s.has_same_dtype_as_array()) {
            if let Some(value) = stats.get(stat) {
                stats.set(stat, value.cast(array.dtype())?);
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PType;
    use vortex_scalar::Scalar;

    use super::*;
    use crate::array::PrimitiveArray;
    use crate::validity::Validity;
    use crate::IntoArrayData as _;

    #[test]
    fn compute_statistics() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("timestamp".into()),
            DType::from(PType::I64).into(),
            None,
        ));
        let array = ExtensionArray::new(
            ext_dtype.clone(),
            PrimitiveArray::from_vec(vec![1i64, 2, 3, 4, 5], Validity::NonNullable).into_array(),
        );

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

        assert_eq!(
            stats.get(Stat::Min),
            Some(&Scalar::extension(ext_dtype.clone(), Scalar::from(1_i64)))
        );
        assert_eq!(
            stats.get(Stat::Max),
            Some(&Scalar::extension(ext_dtype, Scalar::from(5_i64)))
        );
        assert_eq!(stats.get(Stat::NullCount), Some(&0u64.into()));
    }
}
