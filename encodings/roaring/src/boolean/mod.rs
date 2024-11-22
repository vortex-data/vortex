use std::fmt::{Debug, Display};
use std::sync::Arc;

use arrow_buffer::{BooleanBuffer, MutableBuffer};
pub use compress::*;
use croaring::Native;
pub use croaring::{Bitmap, Portable};
use serde::{Deserialize, Serialize};
use vortex_array::array::BoolArray;
use vortex_array::encoding::ids;
use vortex_array::stats::StatsSet;
use vortex_array::validity::{LogicalValidity, ValidityVTable};
use vortex_array::variants::{ArrayVariants, BoolArrayTrait};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData, IntoCanonical,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};

mod compress;
mod compute;
mod stats;

impl_encoding!("vortex.roaring_bool", ids::ROARING_BOOL, RoaringBool);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoaringBoolMetadata;

impl Display for RoaringBoolMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl RoaringBoolArray {
    pub fn try_new(bitmap: Bitmap, length: usize) -> VortexResult<Self> {
        let max_set = bitmap.maximum().unwrap_or(0) as usize;
        if length < max_set {
            vortex_bail!(
                "RoaringBoolArray length is less than bitmap maximum {}",
                max_set
            )
        }

        let stats = StatsSet::bools_with_true_and_null_count(
            bitmap.statistics().cardinality as usize,
            0,
            length,
        );

        ArrayData::try_new_owned(
            &RoaringBoolEncoding,
            DType::Bool(Nullability::NonNullable),
            length,
            Arc::new(RoaringBoolMetadata),
            Some(Buffer::from(bitmap.serialize::<Native>())),
            vec![].into(),
            stats,
        )?
        .try_into()
    }

    pub fn bitmap(&self) -> Bitmap {
        //TODO(@jdcasale): figure out a way to avoid this deserialization per-call
        Bitmap::deserialize::<Native>(self.buffer().as_ref())
    }

    pub fn encode(array: ArrayData) -> VortexResult<ArrayData> {
        if let Ok(bools) = BoolArray::try_from(array) {
            roaring_bool_encode(bools).map(|a| a.into_array())
        } else {
            vortex_bail!("RoaringBool can only encode boolean arrays")
        }
    }

    pub fn buffer(&self) -> &Buffer {
        self.as_ref()
            .buffer()
            .vortex_expect("Missing buffer in PrimitiveArray")
    }
}

impl ArrayTrait for RoaringBoolArray {}

impl ArrayVariants for RoaringBoolArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for RoaringBoolArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        RoaringBoolArray::try_new(self.bitmap().flip(0..(self.len() as u32)), self.len())
            .map(|a| a.into_array())
    }
}

impl VisitorVTable<RoaringBoolArray> for RoaringBoolEncoding {
    fn accept(&self, array: &RoaringBoolArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        // TODO(ngates): should we store a buffer in memory? Or delay serialization?
        //  Or serialize into metadata? The only reason we support buffers is so we can write to
        //  the wire without copying into FlatBuffers. But if we need to allocate to serialize
        //  the bitmap anyway, then may as well shove it into metadata.
        visitor.visit_buffer(array.buffer())
    }
}

impl ValidityVTable<RoaringBoolArray> for RoaringBoolEncoding {
    fn is_valid(&self, _array: &RoaringBoolArray, _index: usize) -> bool {
        true
    }

    fn logical_validity(&self, array: &RoaringBoolArray) -> LogicalValidity {
        LogicalValidity::AllValid(array.len())
    }
}

impl IntoCanonical for RoaringBoolArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        // TODO(ngates): benchmark the fastest conversion from BitMap.
        //  Via bitset requires two copies.
        let bitset = self
            .bitmap()
            .to_bitset()
            .ok_or_else(|| vortex_err!("Failed to convert RoaringBitmap to Bitset"))?;

        let byte_length = (self.len() + 7) / 8;
        let mut buffer = MutableBuffer::with_capacity(byte_length);
        buffer.extend_from_slice(bitset.as_slice());
        if byte_length > bitset.size_in_bytes() {
            buffer.extend_zeros(byte_length - bitset.size_in_bytes());
        }
        Ok(Canonical::Bool(BoolArray::new(
            BooleanBuffer::new(buffer.into(), 0, self.len()),
            Nullability::NonNullable,
        )))
    }
}

#[cfg(test)]
mod test {
    use std::iter;

    use vortex_array::array::BoolArray;
    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};

    use crate::RoaringBoolArray;

    #[test]
    #[cfg_attr(miri, ignore)]
    pub fn iter() {
        let bool: BoolArray = BoolArray::from_iter([true, false, true, true]);
        let array = RoaringBoolArray::encode(bool.into_array()).unwrap();
        let round_trip = RoaringBoolArray::try_from(array).unwrap();
        let values = round_trip.bitmap().to_vec();
        assert_eq!(values, vec![0, 2, 3]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    pub fn trailing_false() {
        let bool: BoolArray = BoolArray::from_iter(
            [true, true]
                .into_iter()
                .chain(iter::repeat(false).take(100)),
        );
        let array = RoaringBoolArray::encode(bool.into_array()).unwrap();
        let round_trip = RoaringBoolArray::try_from(array).unwrap();
        let bool_arr = round_trip.into_bool().unwrap();
        assert_eq!(bool_arr.len(), 102);
    }
}
