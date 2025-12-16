// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::UnsignedPType;
use vortex_vector::Vector;
use vortex_vector::match_each_unsigned_pvector;
use vortex_vector::match_each_vector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::take::Take;

mod binaryview;
mod bool;
mod decimal;

pub use self::bool::default_take;
pub use self::bool::optimized_take;
mod dvector;
mod fixed_size_list;
mod listview;
mod null;
mod primitive;
mod pvector;
mod struct_;

#[cfg(test)]
mod tests;

impl<I: UnsignedPType> Take<PVector<I>> for Vector {
    type Output = Vector;

    fn take(self, indices: &PVector<I>) -> Vector {
        (&self).take(indices)
    }
}

impl<I: UnsignedPType> Take<[I]> for Vector {
    type Output = Vector;

    fn take(self, indices: &[I]) -> Vector {
        (&self).take(indices)
    }
}

impl<I: UnsignedPType> Take<PVector<I>> for &Vector {
    type Output = Vector;

    fn take(self, indices: &PVector<I>) -> Vector {
        match_each_vector!(self, |v| { v.take(indices).into() })
    }
}

impl<I: UnsignedPType> Take<[I]> for &Vector {
    type Output = Vector;

    fn take(self, indices: &[I]) -> Vector {
        match_each_vector!(self, |v| { v.take(indices).into() })
    }
}

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

impl Take<PrimitiveVector> for Vector {
    type Output = Vector;

    fn take(self, indices: &PrimitiveVector) -> Vector {
        (&self).take(indices)
    }
}
