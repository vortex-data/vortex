// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use crate::ToCanonical;
use crate::accessor::ArrayAccessor;
use crate::arrays::primitive::PrimitiveArray;
use crate::dtype::NativePType;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl<T: NativePType> ArrayAccessor<T> for PrimitiveArray {
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a T>>) -> R,
    {
        match self.validity() {
            Validity::NonNullable | Validity::AllValid => {
                let mut iter = self.as_slice::<T>().iter().map(Some);
                f(&mut iter)
            }
            Validity::AllInvalid => f(&mut iter::repeat_n(None, self.len())),
            Validity::Array(v) => {
                let validity = v.to_bool().into_bit_buffer();
                let mut iter = self
                    .as_slice::<T>()
                    .iter()
                    .zip(validity.iter())
                    .map(|(value, valid)| valid.then_some(value));
                f(&mut iter)
            }
        }
    }
}
