use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_array::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::stats::{ArrayStatistics, ArrayStatisticsCompute, Stat, StatsSet};
use vortex_array::validity::{ArrayValidity, LogicalValidity};
use vortex_array::variants::{ArrayVariants, PrimitiveArrayTrait};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayTrait, Canonical, IntoArrayVariant, IntoCanonical,
};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;
use zigzag::ZigZag as ExternalZigZag;

use crate::compress::zigzag_encode;
use crate::zigzag_decode;

impl_encoding!("vortex.zigzag", ids::ZIGZAG, ZigZag);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZigZagMetadata;

impl Display for ZigZagMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZigZagMetadata")
    }
}

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

        Self::try_from_parts(dtype, len, ZigZagMetadata, children.into(), StatsSet::new())
    }

    pub fn encode(array: &ArrayData) -> VortexResult<ZigZagArray> {
        PrimitiveArray::try_from(array)
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

impl ArrayTrait for ZigZagArray {}

impl ArrayVariants for ZigZagArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for ZigZagArray {}

impl ArrayValidity for ZigZagArray {
    fn is_valid(&self, index: usize) -> bool {
        self.encoded().with_dyn(|a| a.is_valid(index))
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.encoded().with_dyn(|a| a.logical_validity())
    }
}

impl AcceptArrayVisitor for ZigZagArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", &self.encoded())
    }
}

impl ArrayStatisticsCompute for ZigZagArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::new();

        // these stats are the same for self and self.encoded()
        if matches!(stat, Stat::IsConstant | Stat::NullCount) {
            if let Some(val) = self.encoded().statistics().compute(stat) {
                stats.set(stat, val);
            }
        } else if matches!(stat, Stat::Min | Stat::Max) {
            let encoded_max = self
                .encoded()
                .statistics()
                .compute_as_cast::<u64>(Stat::Max);
            if let Some(val) = encoded_max {
                // the max of the encoded array is the element with the highest absolute value (so either min if negative, or max if positive)
                let decoded = <i64 as ExternalZigZag>::decode(val);
                let decoded_stat = if decoded < 0 { Stat::Min } else { Stat::Max };
                stats.set(decoded_stat, Scalar::from(decoded).cast(self.dtype())?);
            }
        }

        Ok(stats)
    }
}

impl IntoCanonical for ZigZagArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        zigzag_decode(self.encoded().into_primitive()?).map(Canonical::Primitive)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::compute::slice;
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::IntoArrayData;

    use super::*;

    #[test]
    fn test_compute_statistics() {
        let array =
            PrimitiveArray::from(vec![1i32, -5i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]).into_array();
        let zigzag = ZigZagArray::encode(&array).unwrap();

        for stat in [Stat::Max, Stat::NullCount, Stat::IsConstant] {
            let stats = zigzag.compute_statistics(stat).unwrap();
            assert_eq!(stats.get(stat), array.statistics().compute(stat).as_ref());
        }

        let sliced = ZigZagArray::try_from(slice(zigzag.clone(), 0, 2).unwrap()).unwrap();
        assert_eq!(
            scalar_at(&sliced, sliced.len() - 1).unwrap(),
            Scalar::from(-5i32)
        );
        for stat in [Stat::Min, Stat::NullCount, Stat::IsConstant] {
            let stats = sliced.compute_statistics(stat).unwrap();
            assert_eq!(stats.get(stat), array.statistics().compute(stat).as_ref());
        }
    }
}
