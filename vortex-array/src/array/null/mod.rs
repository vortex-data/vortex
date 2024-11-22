use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::nbytes::ArrayNBytes;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::validity::{LogicalValidity, Validity, ValidityVTable};
use crate::variants::{ArrayVariants, NullArrayTrait};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{impl_encoding, ArrayLen, ArrayTrait, Canonical, IntoCanonical};

mod compute;

impl_encoding!("vortex.null", ids::NULL, Null);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullMetadata;

impl Display for NullMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NullMetadata")
    }
}

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self::try_from_parts(
            DType::Null,
            len,
            NullMetadata,
            [].into(),
            StatsSet::nulls(len, &DType::Null),
        )
        .vortex_expect("NullArray::new should never fail!")
    }
}

impl IntoCanonical for NullArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Null(self))
    }
}

impl ValidityVTable<NullArray> for NullEncoding {
    fn is_valid(&self, _array: &NullArray, _idx: usize) -> bool {
        false
    }

    fn logical_validity(&self, array: &NullArray) -> LogicalValidity {
        LogicalValidity::AllInvalid(array.len())
    }
}

impl StatisticsVTable<NullArray> for NullEncoding {
    fn compute_statistics(&self, array: &NullArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, array.nbytes()));
        }

        Ok(StatsSet::nulls(array.len(), &DType::Null))
    }
}

impl VisitorVTable<NullArray> for NullEncoding {
    fn accept(&self, _array: &NullArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_validity(&Validity::AllInvalid)
    }
}

impl ArrayTrait for NullArray {}

impl ArrayVariants for NullArray {
    fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for NullArray {}
