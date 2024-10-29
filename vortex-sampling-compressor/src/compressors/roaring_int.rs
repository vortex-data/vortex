use vortex::aliases::hash_set::HashSet;
use vortex::encoding::EncodingRef;
use vortex::stats::ArrayStatistics;
use vortex::{Array, ArrayDType, ArrayDef, IntoArray, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_roaring::{roaring_int_encode, RoaringInt, RoaringIntEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct RoaringIntCompressor;

impl EncodingCompressor for RoaringIntCompressor {
    fn id(&self) -> &str {
        RoaringInt::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::ROARING_INT_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        // Only support non-nullable uint arrays
        if !array.dtype().is_unsigned_int() || array.dtype().is_nullable() {
            return None;
        }

        // Only support sorted unique arrays
        if !array
            .statistics()
            .compute_is_strict_sorted()
            .unwrap_or(false)
        {
            return None;
        }

        if array.statistics().compute_max().unwrap_or(0) > u32::MAX as usize {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        _like: Option<CompressionTree<'a>>,
        _ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        Ok(CompressedArray::new(
            roaring_int_encode(array.clone().into_primitive()?)?.into_array(),
            Some(CompressionTree::flat(self)),
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&RoaringIntEncoding as EncodingRef])
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::PrimitiveArray;
    use vortex::validity::Validity;
    use vortex::IntoArray;
    use vortex_roaring::RoaringIntArray;

    use crate::compressors::roaring_int::RoaringIntCompressor;
    use crate::compressors::EncodingCompressor as _;
    use crate::SamplingCompressor;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_roaring_int_compressor() {
        let array =
            PrimitiveArray::from_vec(vec![1u32, 2, 3, 4, 5], Validity::NonNullable).into_array();
        assert!(RoaringIntCompressor.can_compress(&array).is_some());
        let compressed = RoaringIntCompressor
            .compress(&array, None, SamplingCompressor::default())
            .unwrap();
        assert_eq!(compressed.array.len(), 5);
        assert!(compressed.path.is_some());

        let roaring = RoaringIntArray::try_from(compressed.array).unwrap();
        assert!(roaring.owned_bitmap().contains_range(1..=5));
    }
}
