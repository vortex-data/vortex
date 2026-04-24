// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::accessor::ArrayAccessor;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::validity::Validity;

impl<T: NativePType> ArrayAccessor<T> for PrimitiveArray {
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a T>>) -> R,
    {
        match self
            .validity()
            .vortex_expect("primitive validity should be derivable")
        {
            Validity::NonNullable | Validity::AllValid => {
                let mut iter = self.as_slice::<T>().iter().map(Some);
                f(&mut iter)
            }
            Validity::AllInvalid => f(&mut iter::repeat_n(None, self.len())),
            Validity::Array(v) => {
                #[expect(deprecated)]
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
