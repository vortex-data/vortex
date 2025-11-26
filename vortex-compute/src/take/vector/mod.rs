// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::match_each_unsigned_pvector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::take::Take;

mod bool;
mod primitive;
mod pvector;

#[cfg(test)]
mod tests;

impl<T> Take<PrimitiveVector> for &T
where
    for<'a> &'a T: Take<PVector<u8>, Output = T>,
    for<'a> &'a T: Take<PVector<u16>, Output = T>,
    for<'a> &'a T: Take<PVector<u32>, Output = T>,
    for<'a> &'a T: Take<PVector<u64>, Output = T>,
{
    type Output = T;

    fn take(self, indices: &PrimitiveVector) -> T {
        match_each_unsigned_pvector!(indices, |iv| { self.take(iv) })
    }
}
