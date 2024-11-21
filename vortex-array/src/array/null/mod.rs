use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};

use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::encoding::ids;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity, Validity};
use crate::variants::{ArrayVariants, NullArrayTrait};
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

impl ArrayValidity for NullArray {
    fn is_valid(&self, _: usize) -> bool {
        false
    }

    fn logical_validity(&self) -> LogicalValidity {
        LogicalValidity::AllInvalid(self.len())
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

impl AcceptArrayVisitor for NullArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_validity(&Validity::AllInvalid)
    }
}

impl ArrayTrait for NullArray {
    fn nbytes(&self) -> usize {
        0
    }
}

impl ArrayVariants for NullArray {
    fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for NullArray {}
