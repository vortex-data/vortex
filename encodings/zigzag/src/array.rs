use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::stats::{ArrayStatistics, Stat, StatsSet};
use vortex_array::validity::ArrayValidity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{
    CanonicalVTable, StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable,
    VisitorVTable,
};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, Canonical, EmptyMetadata, IntoArrayVariant,
};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::ScalarValue;
use zigzag::ZigZag as ExternalZigZag;

use crate::compress::zigzag_encode;
use crate::zigzag_decode;

impl_encoding!("vortex.zigzag", ids::ZIGZAG, ZigZag, EmptyMetadata);

impl ZigZagArray {
    pub fn try_new(encoded: ArrayData) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        let len = encoded.len();
        let children = [encoded];

        Self::try_from_parts(
            dtype,
            len,
            EmptyMetadata,
            None,
            Some(children.into()),
            StatsSet::default(),
        )
    }

    pub fn encode(array: &ArrayData) -> VortexResult<ZigZagArray> {
        PrimitiveArray::try_from(array.clone())
            .map_err(|_| vortex_err!("ZigZag can only encoding primitive arrays"))
            .and_then(zigzag_encode)
    }

    pub fn encoded(&self) -> ArrayData {
        let ptype = PType::try_from(self.dtype()).unwrap_or_else(|err| {
            vortex_panic!(err, "Failed to convert DType {} to PType", self.dtype())
        });
        let encoded = DType::from(ptype.to_unsigned()).with_nullability(self.dtype().nullability());
        self.as_ref()
            .child(0, &encoded, self.len())
            .vortex_expect("ZigZagArray is missing its encoded child array")
    }
}

impl ValidateVTable<ZigZagArray> for ZigZagEncoding {}

impl VariantsVTable<ZigZagArray> for ZigZagEncoding {
    fn as_primitive_array<'a>(
        &self,
        array: &'a ZigZagArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for ZigZagArray {}

impl ValidityVTable<ZigZagArray> for ZigZagEncoding {
    fn is_valid(&self, array: &ZigZagArray, index: usize) -> VortexResult<bool> {
        array.encoded().is_valid(index)
    }

    fn logical_validity(&self, array: &ZigZagArray) -> VortexResult<Mask> {
        array.encoded().logical_validity()
    }
}

impl VisitorVTable<ZigZagArray> for ZigZagEncoding {
    fn accept(&self, array: &ZigZagArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", &array.encoded())
    }
}

impl StatisticsVTable<ZigZagArray> for ZigZagEncoding {
    fn compute_statistics(&self, array: &ZigZagArray, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::default();

        // these stats are the same for array and array.encoded()
        if matches!(stat, Stat::IsConstant | Stat::NullCount) {
            if let Some(val) = array.encoded().statistics().compute(stat) {
                stats.set(stat, val);
            }
        } else if matches!(stat, Stat::Min | Stat::Max) {
            let encoded_max = array.encoded().statistics().compute_as::<u64>(Stat::Max);
            if let Some(val) = encoded_max {
                // the max of the encoded array is the element with the highest absolute value (so either min if negative, or max if positive)
                let decoded = <i64 as ExternalZigZag>::decode(val);
                let decoded_stat = if decoded < 0 { Stat::Min } else { Stat::Max };
                stats.set(decoded_stat, ScalarValue::from(decoded));
            }
        }

        Ok(stats)
    }
}

impl CanonicalVTable<ZigZagArray> for ZigZagEncoding {
    fn into_canonical(&self, array: ZigZagArray) -> VortexResult<Canonical> {
        zigzag_decode(array.encoded().into_primitive()?).map(Canonical::Primitive)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::compute::{scalar_at, slice};
    use vortex_array::IntoArrayData;
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

        let sliced = ZigZagArray::try_from(slice(zigzag, 0, 2).unwrap()).unwrap();
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
