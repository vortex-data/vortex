use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::encoding::encoding_ids;
use crate::stats::{Stat, StatsSet};
use crate::validity::Validity;
use crate::variants::NullArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::vtable::{
    CanonicalVTable, StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable,
    VisitorVTable,
};
use crate::{impl_encoding, Canonical, EmptyMetadata};

mod compute;

impl_encoding!("vortex.null", encoding_ids::NULL, Null, EmptyMetadata);

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self::try_from_parts(
            DType::Null,
            len,
            EmptyMetadata,
            None,
            None,
            StatsSet::nulls(len, &DType::Null),
        )
        .vortex_expect("NullArray::new should never fail!")
    }
}

impl CanonicalVTable<NullArray> for NullEncoding {
    fn into_canonical(&self, array: NullArray) -> VortexResult<Canonical> {
        Ok(Canonical::Null(array))
    }
}

impl ValidityVTable<NullArray> for NullEncoding {
    fn is_valid(&self, _array: &NullArray, _idx: usize) -> VortexResult<bool> {
        Ok(false)
    }

    fn all_valid(&self, array: &NullArray) -> VortexResult<bool> {
        Ok(array.len() == 0)
    }

    fn validity_mask(&self, array: &NullArray) -> VortexResult<Mask> {
        Ok(Mask::AllFalse(array.len()))
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

impl ValidateVTable<NullArray> for NullEncoding {}

impl VariantsVTable<NullArray> for NullEncoding {
    fn as_null_array<'a>(&self, array: &'a NullArray) -> Option<&'a dyn NullArrayTrait> {
        Some(array)
    }
}

impl NullArrayTrait for NullArray {}
