use std::iter;

use vortex_dtype::NativePType;
use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::primitive::PrimitiveArray;
use crate::validity::Validity;
use crate::IntoArrayVariant;

impl<T: NativePType> ArrayAccessor<T> for PrimitiveArray {
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut (dyn Iterator<Item = Option<&'a T>>)) -> R,
    {
        match self.validity() {
            Validity::NonNullable | Validity::AllValid => {
                let mut iter = self.as_slice::<T>().iter().map(Some);
                Ok(f(&mut iter))
            }
            Validity::AllInvalid => Ok(f(&mut iter::repeat_n(None, self.len()))),
            Validity::Array(v) => {
                let validity = v.into_bool()?.boolean_buffer();
                let mut iter = self
                    .as_slice::<T>()
                    .iter()
                    .zip(validity.iter())
                    .map(|(value, valid)| valid.then_some(value));
                Ok(f(&mut iter))
            }
        }
    }
}
