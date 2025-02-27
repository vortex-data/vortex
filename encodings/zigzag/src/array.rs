use vortex_array::arrays::PrimitiveArray;
use vortex_array::stats::{ArrayStats, Precision, Stat, StatsSet, StatsSetRef};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::{StatisticsVTable, VTableRef};
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, EmptyMetadata, Encoding, EncodingId, ToCanonical,
    try_from_array_ref,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use zigzag::ZigZag as ExternalZigZag;

use crate::compress::zigzag_encode;
use crate::zigzag_decode;

#[derive(Clone, Debug)]
pub struct ZigZagArray {
    dtype: DType,
    encoded: ArrayRef,
    stats_set: ArrayStats,
}

try_from_array_ref!(ZigZagArray);

pub struct ZigZagEncoding;
impl Encoding for ZigZagEncoding {
    const ID: EncodingId = EncodingId::new_ref("vortex.zigzag");
    type Array = ZigZagArray;
    type Metadata = EmptyMetadata;
}

impl ZigZagArray {
    pub fn try_new(encoded: ArrayRef) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        Ok(Self {
            dtype,
            encoded,
            stats_set: Default::default(),
        })
    }

    pub fn encode(array: &dyn Array) -> VortexResult<ZigZagArray> {
        PrimitiveArray::try_from(array.to_array())
            .map_err(|_| vortex_err!("ZigZag can only encoding primitive arrays"))
            .and_then(zigzag_encode)
    }

    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }
}

impl ArrayImpl for ZigZagArray {
    type Encoding = ZigZagEncoding;

    fn _len(&self) -> usize {
        self.encoded.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&ZigZagEncoding)
    }
}

impl ArrayCanonicalImpl for ZigZagArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        zigzag_decode(self.encoded().to_primitive()?).map(Canonical::Primitive)
    }
}

impl ArrayStatisticsImpl for ZigZagArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for ZigZagArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.encoded.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.encoded.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.encoded.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.encoded.validity_mask()
    }
}

impl ArrayVariantsImpl for ZigZagArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for ZigZagArray {}

impl StatisticsVTable<&ZigZagArray> for ZigZagEncoding {
    fn compute_statistics(&self, array: &ZigZagArray, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::default();

        // these stats are the same for array and array.encoded()
        if matches!(stat, Stat::IsConstant | Stat::NullCount) {
            if let Some(val) = array.encoded().statistics().compute_stat(stat)? {
                stats.set(stat, Precision::exact(val));
            }
        } else if matches!(stat, Stat::Min | Stat::Max) {
            let encoded_max = array.encoded().statistics().compute_as::<u64>(Stat::Max);
            if let Some(val) = encoded_max {
                // the max of the encoded array is the element with the highest absolute value (so either min if negative, or max if positive)
                let decoded = <i64 as ExternalZigZag>::decode(val);
                let decoded_stat = if decoded < 0 { Stat::Min } else { Stat::Max };
                stats.set(decoded_stat, Precision::exact(decoded));
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::compute::{scalar_at, slice};
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_compute_statistics() {
        let array = buffer![1i32, -5i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        let zigzag = ZigZagArray::encode(&array).unwrap();

        assert_eq!(
            zigzag.statistics().compute_max::<i32>(),
            array.statistics().compute_max::<i32>()
        );
        assert_eq!(
            zigzag.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            zigzag.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );

        let sliced = ZigZagArray::try_from(slice(&zigzag, 0, 2).unwrap()).unwrap();
        assert_eq!(
            scalar_at(&sliced, sliced.len() - 1).unwrap(),
            Scalar::from(-5i32)
        );

        assert_eq!(
            sliced.statistics().compute_min::<i32>(),
            array.statistics().compute_min::<i32>()
        );
        assert_eq!(
            sliced.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            sliced.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );
    }
}
