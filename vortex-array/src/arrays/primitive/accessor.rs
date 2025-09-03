// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;

use crate::arrays::primitive::PrimitiveArray;
use crate::validity::Validity;

pub struct Iter<'a, T> {
    index: usize,
    buffer: &'a [T],
    validity: &'a Validity,
}

impl<'a, T: NativePType> Iterator for Iter<'a, T> {
    type Item = Option<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len() {
            return None;
        }

        let result = self
            .validity
            .is_valid(self.index)
            .then(|| self.buffer[self.index]);

        self.index += 1;

        Some(result)
    }
}

impl PrimitiveArray {
    #[inline]
    pub fn typed_iter<T: NativePType>(&self) -> Iter<'_, T> {
        Iter {
            index: 0,
            buffer: self.as_slice::<T>(),
            validity: &self.validity,
        }
    }
}
