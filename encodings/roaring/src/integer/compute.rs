use croaring::Bitmap;
use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{ArrayCompute, SliceFn};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::PType;
use vortex_error::{vortex_err, VortexResult, VortexUnwrap as _};
use vortex_scalar::Scalar;

use crate::RoaringIntArray;

impl ArrayCompute for RoaringIntArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }
}

impl ScalarAtFn for RoaringIntArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let bitmap_value = self
            .owned_bitmap()
            .select(index as u32)
            .ok_or_else(|| vortex_err!(OutOfBounds: index, 0, self.len()))?;
        let scalar: Scalar = match self.metadata().ptype {
            PType::U8 => (bitmap_value as u8).into(),
            PType::U16 => (bitmap_value as u16).into(),
            PType::U32 => bitmap_value.into(),
            PType::U64 => (bitmap_value as u64).into(),
            _ => unreachable!("RoaringIntArray constructor should have disallowed this type"),
        };
        Ok(scalar)
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        <Self as ScalarAtFn>::scalar_at(self, index).vortex_unwrap()
    }
}

impl SliceFn for RoaringIntArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let mut bitmap = self.owned_bitmap();
        let start = bitmap
            .select(start as u32)
            .ok_or_else(|| vortex_err!(OutOfBounds: start, 0, self.len()))?;
        let stop_inclusive = if stop == self.len() {
            bitmap.maximum().unwrap_or(0)
        } else {
            bitmap
                .select(stop.saturating_sub(1) as u32)
                .ok_or_else(|| vortex_err!(OutOfBounds: stop, 0, self.len()))?
        };

        bitmap.and_inplace(&Bitmap::from_range(start..=stop_inclusive));
        Self::try_new(bitmap, self.cached_ptype()).map(IntoArrayData::into_array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::slice;
    use vortex_array::compute::unary::scalar_at;

    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    pub fn test_scalar_at() {
        let ints = PrimitiveArray::from(vec![2u32, 12, 22, 32]).into_array();
        let array = RoaringIntArray::encode(ints).unwrap();

        assert_eq!(scalar_at(&array, 0).unwrap(), 2u32.into());
        assert_eq!(scalar_at(&array, 1).unwrap(), 12u32.into());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_slice() {
        let array = RoaringIntArray::try_new(Bitmap::from_range(10..20), PType::U32).unwrap();

        let sliced = slice(&array, 0, 5).unwrap();
        assert_eq!(sliced.len(), 5);
        assert_eq!(scalar_at(&sliced, 0).unwrap(), 10u32.into());
        assert_eq!(scalar_at(&sliced, 4).unwrap(), 14u32.into());

        let sliced = slice(&array, 5, 10).unwrap();
        assert_eq!(sliced.len(), 5);
        assert_eq!(scalar_at(&sliced, 0).unwrap(), 15u32.into());
        assert_eq!(scalar_at(&sliced, 4).unwrap(), 19u32.into());

        let sliced = slice(&sliced, 3, 5).unwrap();
        assert_eq!(sliced.len(), 2);
        assert_eq!(scalar_at(&sliced, 0).unwrap(), 18u32.into());
        assert_eq!(scalar_at(&sliced, 1).unwrap(), 19u32.into());
    }
}
