use vortex_dtype::NativePType;

use crate::accessor::ArrayAccessor;
use crate::array::primitive::PrimitiveArray;
use crate::validity::ArrayValidity;

impl<T: NativePType> ArrayAccessor<T> for PrimitiveArray {
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut (dyn Iterator<Item = Option<&'a T>>)) -> R,
    {
        match self.logical_validity().to_null_buffer() {
            None => {
                let mut iter = self.as_slice::<T>().iter().map(Some);
                f(&mut iter)
            }
            Some(nulls) => {
                let mut iter = self
                    .as_slice::<T>()
                    .iter()
                    .zip(nulls.iter())
                    .map(|(value, valid)| valid.then_some(value));
                f(&mut iter)
            }
        }
    }
}
