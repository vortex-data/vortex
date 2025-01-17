use std::fmt::{Debug, Display};

pub use compress::*;
use croaring::{Bitmap, Portable};
use serde::{Deserialize, Serialize};
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::try_cast;
use vortex_array::encoding::ids;
use vortex_array::stats::{ArrayStatistics, Stat, StatisticsVTable, StatsSet};
use vortex_array::validate::ValidateVTable;
use vortex_array::validity::{LogicalValidity, Validity, ValidityVTable};
use vortex_array::variants::{PrimitiveArrayTrait, VariantsVTable};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType as _, ArrayData, ArrayLen, Canonical, IntoArrayData, IntoCanonical,
};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

mod compress;
mod compute;

impl_encoding!("vortex.roaring_int", ids::ROARING_INT, RoaringInt);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoaringIntMetadata {
    ptype: PType,
}

impl Display for RoaringIntMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl RoaringIntArray {
    pub fn try_new(bitmap: Bitmap, ptype: PType) -> VortexResult<Self> {
        if !ptype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", ptype);
        }

        let length = bitmap.statistics().cardinality as usize;
        let max = bitmap.maximum();
        if max
            .map(|mv| mv as u64 > ptype.max_value_as_u64())
            .unwrap_or(false)
        {
            vortex_bail!(
                "Bitmap's maximum value ({}) is greater than the maximum value for the primitive type ({})",
                max.vortex_expect("Bitmap has no maximum value despite having just checked"),
                ptype
            );
        }

        let mut stats = StatsSet::default();
        stats.set(Stat::NullCount, 0);
        stats.set(Stat::Max, max);
        stats.set(Stat::Min, bitmap.minimum());
        stats.set(Stat::IsConstant, length <= 1);
        stats.set(Stat::IsSorted, true);
        stats.set(Stat::IsStrictSorted, true);

        Self::try_from_parts(
            DType::Primitive(ptype, NonNullable),
            length,
            RoaringIntMetadata { ptype },
            Some([ByteBuffer::from(bitmap.serialize::<Portable>())].into()),
            None,
            stats,
        )
    }

    pub fn owned_bitmap(&self) -> Bitmap {
        Bitmap::deserialize::<Portable>(
            self.as_ref()
                .byte_buffer(0)
                .vortex_expect("RoaringBoolArray buffer is missing")
                .as_ref(),
        )
    }

    pub fn cached_ptype(&self) -> PType {
        self.metadata().ptype
    }

    pub fn encode(array: ArrayData) -> VortexResult<ArrayData> {
        if let Ok(parray) = PrimitiveArray::try_from(array) {
            Ok(roaring_int_encode(parray)?.into_array())
        } else {
            vortex_bail!("RoaringInt can only encode primitive arrays")
        }
    }
}

impl ValidateVTable<RoaringIntArray> for RoaringIntEncoding {}

impl VariantsVTable<RoaringIntArray> for RoaringIntEncoding {
    fn as_primitive_array<'a>(
        &self,
        array: &'a RoaringIntArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for RoaringIntArray {}

impl ValidityVTable<RoaringIntArray> for RoaringIntEncoding {
    fn is_valid(&self, _array: &RoaringIntArray, _index: usize) -> bool {
        true
    }

    fn logical_validity(&self, array: &RoaringIntArray) -> LogicalValidity {
        LogicalValidity::AllValid(array.len())
    }
}

impl IntoCanonical for RoaringIntArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        try_cast(
            PrimitiveArray::new(
                // TODO(ngates): we may well care about this copy.
                Buffer::copy_from(self.owned_bitmap().to_vec()),
                Validity::NonNullable,
            ),
            self.dtype(),
        )
        .and_then(ArrayData::into_canonical)
    }
}

impl VisitorVTable<RoaringIntArray> for RoaringIntEncoding {
    fn accept(&self, array: &RoaringIntArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(
            array
                .as_ref()
                .byte_buffer(0)
                .vortex_expect("Missing buffer in RoaringIntArray"),
        )
    }
}

impl StatisticsVTable<RoaringIntArray> for RoaringIntEncoding {
    fn compute_statistics(&self, array: &RoaringIntArray, stat: Stat) -> VortexResult<StatsSet> {
        // possibly faster to write an accumulator over the iterator, though not necessarily
        if stat == Stat::TrailingZeroFreq || stat == Stat::BitWidthFreq || stat == Stat::RunCount {
            let primitive = PrimitiveArray::new(
                // TODO(ngates): can we change owned_bitmap to avoid the copy?
                Buffer::copy_from(array.owned_bitmap().to_vec()),
                Validity::NonNullable,
            );
            primitive.statistics().compute_all(&[
                Stat::TrailingZeroFreq,
                Stat::BitWidthFreq,
                Stat::RunCount,
            ])
        } else {
            Ok(StatsSet::default())
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::RoaringIntMetadata;

    #[test]
    fn test_roaring_int_metadata() {
        check_metadata(
            "roaring_int.metadata",
            RoaringIntMetadata { ptype: PType::U64 },
        );
    }
}
