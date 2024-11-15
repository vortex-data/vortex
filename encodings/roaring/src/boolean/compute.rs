use croaring::Bitmap;
use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{ArrayCompute, SliceFn};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::RoaringBoolArray;

impl ArrayCompute for RoaringBoolArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }
}

impl ScalarAtFn for RoaringBoolArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(<Self as ScalarAtFn>::scalar_at_unchecked(self, index))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        self.bitmap().contains(index as u32).into()
    }
}

impl SliceFn for RoaringBoolArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let slice_bitmap = Bitmap::from_range(start as u32..stop as u32);
        let bitmap = self.bitmap().and(&slice_bitmap).add_offset(-(start as i64));

        Self::try_new(bitmap, stop - start).map(IntoArrayData::into_array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::BoolArray;
    use vortex_array::compute::slice;
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_scalar::Scalar;

    use crate::RoaringBoolArray;

    #[test]
    #[cfg_attr(miri, ignore)]
    pub fn test_scalar_at() {
        let bool = BoolArray::from(vec![true, false, true, true]);
        let array = RoaringBoolArray::encode(bool.into_array()).unwrap();

        let truthy: Scalar = true.into();
        let falsy: Scalar = false.into();

        assert_eq!(scalar_at(&array, 0).unwrap(), truthy);
        assert_eq!(scalar_at(&array, 1).unwrap(), falsy);
        assert_eq!(scalar_at(&array, 2).unwrap(), truthy);
        assert_eq!(scalar_at(&array, 3).unwrap(), truthy);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    pub fn test_slice() {
        let bool = BoolArray::from(vec![true, false, true, true]);
        let array = RoaringBoolArray::encode(bool.into_array()).unwrap();
        let sliced = slice(&array, 1, 3).unwrap();

        assert_eq!(
            sliced
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            &[false, true]
        );
    }
}
